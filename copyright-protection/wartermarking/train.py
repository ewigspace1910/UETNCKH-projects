import sys
import os
sys.path.insert(0, os.path.join(os.path.__file__))

import torch
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
import argparse
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms
from torch.autograd import Variable
import torch.autograd as autograd

from data_loader import create_watermark
from models.stargan import Discriminator
from models.stargan import GeneratorResNet as Generator
from models import VAEWrapper
from losses import gan_loss, adversarial_loss, perturbation_loss
from data_loader import create_dataloader
# from evaluate import evaluate_adversarial_quality
from utils.meter import AverageMeter

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--name", type=str, default="trial")
    args.add_argument("--checkpoint", type=str, default=None)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--iheight", type=int, default=256, help="size of image height")
    args.add_argument("--iwidth", type=int, default=256, help="size of image width")
    args.add_argument("--lam", type=float, default=1)
    args.add_argument("--alpha", type=float, default=1)
    args.add_argument("--beta", type=float, default=10)
    args.add_argument("--c", type=float, default=10/255)
    args.add_argument("--watermark_region", type=float, default=4.0)
    args.add_argument("--input_channels", type=int, default=3)
    args.add_argument("--vae_path", type=str, default="runwayml/stable-diffusion-v1-5")
    args.add_argument("--lr", type=float, default=0.0002)
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--save_dir", type=str, default="../saves")
    args.add_argument("--beta1", type=float, default=0.5)
    args.add_argument("--beta2", type=float, default=0.999)
    args.add_argument("--device", type=str, default="cuda")

    args.add_argument("--train_dir", type=str, default="data/wikiart")
    args.add_argument("--train_classes", type=str, default="data/wikiart/train_classes.csv")
    args.add_argument("--eval_dir", type=str, default="data/wikiart")
    args.add_argument("--eval_classes", type=str, default="data/wikiart/eval_classes.csv")

    args.add_argument("--n-critic", type=int, default=5, help="number of training iterations for WGAN discriminator")
    return args.parse_args()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    

def train_step(G, D, vae, optimizer_G, optimizer_D, real_images, watermark, config, is_train_G=True):
    device = real_images.device
    current_batch_size = real_images.size(0)
    valid = torch.full((current_batch_size, 1), 0.9, device=device)
    fake = torch.full((current_batch_size, 1), 0.1, device=device)


    def _reset_grad():
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
    _reset_grad()
    # Train Discriminator
    _, fake_images = G(real_images, watermark)
    real_validity = D(real_images)
    fake_validity = D(fake_images.detach())
    gradient_penalty = compute_gradient_penalty(D, real_images.data, fake_images.data) * 10
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) +  gradient_penalty
    # d_loss = gan_loss(D(real_images), valid) + gan_loss(D(fake_images.detach()), fake)
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    if is_train_G:
        _reset_grad()
        perturbation, fake_images = G(real_images, watermark)
        adv_loss_ = adversarial_loss(vae, fake_images, watermark)
        gan_loss_ = -torch.mean(D(fake_images)) #gan_loss(D(fake_images), valid)
        perturbation_loss_ = perturbation_loss(perturbation, watermark, config.c, config.watermark_region)

        g_loss = config.lam*adv_loss_ + config.alpha * gan_loss_ + config.beta * perturbation_loss_
        g_loss.backward()
        optimizer_G.step()
        g_loss_ = {'g':g_loss, 'adv_loss': adv_loss_.item(), 'gan_loss': gan_loss_.item(), 'perturbation_loss': perturbation_loss_.item()} 
    else:g_loss_ = {'g':0,'adv_loss': 0, 'gan_loss': 0, 'perturbation_loss': 0} 

    return d_loss.item(), g_loss_, fake_images


def main(args, pipe):
    test_image = Image.open("./data/imagenet/IMAGENET_DOG/n02109961_48_n02109961.JPEG").convert("RGB")
    test_image_size = test_image.size[::-1]

    # Load dataset
    train_image_dir = args.train_dir
    train_classes_csv = args.train_classes

    eval_image_dir = args.eval_dir
    eval_classes_csv = args.eval_classes
    # image_dir = "data/imagenet"
    # classes_csv = "data/imagenet/image_artist.csv"

    # Create dataloader
    train_dataloader, metadata = create_dataloader(
        image_dir=train_image_dir,
        classes_csv=train_classes_csv,
        batch_size=args.batch_size,
        image_size=(args.iwidth, args.iheight)
    )

    eval_dataloader, metadata = create_dataloader(
        image_dir=eval_image_dir,
        classes_csv=eval_classes_csv,
        batch_size=args.batch_size,
        image_size=(args.iwidth, args.iheight)
    )

    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(img_shape=(6, args.iwidth, args.iheight)).to(device)
    D = Discriminator(img_shape=(3, args.iwidth, args.iheight)).to(device)
    vae = VAEWrapper(args.vae_path).to(device)
    for param in vae.vae.parameters(): param.detach_()

    # Setup optimizers
    optimizer_G = torch.optim.Adam(
        G.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        betas=(args.beta1, args.beta2)
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        betas=(args.beta1, args.beta2)
    )

    # Training loop
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch']
        G.load_state_dict(checkpoint['generator_state_dict'])
        D.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    timestamp         = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir          = f"{args.save_dir}/{args.name}/{timestamp}"
    save_dir_advimage = f"{save_dir}/adv_image"
    save_dir_difimage    = f"{save_dir}/dif_image"
    for p in [save_dir, save_dir_advimage, save_dir_difimage]:   os.makedirs(p, exist_ok=True)
    

    for epoch in range(start_epoch, args.num_epochs+1):
        G.train()
        D.train()
        meters = {
            "loss_D": AverageMeter(),
            "loss_G": AverageMeter(),
            "loss_G_adv": AverageMeter(),
            "loss_G_gan": AverageMeter(),
            "loss_G_per": AverageMeter()}
            
        for batch_idx, (real_images, watermark, _) in enumerate(train_dataloader):
            real_images = real_images.to(device)
            watermark = watermark.to(device)

            d_loss, g_loss, fake_images = train_step(G, D, vae, optimizer_G, optimizer_D,
                                                    real_images, watermark, args,
                                                    is_train_G=((batch_idx % args.n_critic) == 0))
            meters['loss_D'].update(d_loss, real_images.shape[0])
            meters['loss_G'].update(g_loss['g'], real_images.shape[0])
            meters['loss_G_adv'].update(g_loss['adv_loss'], real_images.shape[0])
            meters['loss_G_gan'].update(g_loss['gan_loss'], real_images.shape[0])
            meters['loss_G_per'].update(g_loss['perturbation_loss'], real_images.shape[0])

            if batch_idx % 10 == 0:
                info_str= f"Epoch [{epoch}/{args.num_epochs}] | Batch [{batch_idx}] \t"
                for k, v in meters.items():
                    if (v.avg > 0):
                        info_str += f", {k}: {v.avg:.3f}"
                print(info_str)
                
        ### save test image every epoch
        G.eval()
        watermark = create_watermark("ANIMAL", test_image_size).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((args.iwidth, args.iheight)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-m / s for m, s in zip((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))], std=[1 / s for s in (0.5, 0.5, 0.5)]),
            transforms.ToPILImage(),
        ])
        image = transform(test_image).unsqueeze(0).to(device)
        watermark = transform(watermark).unsqueeze(0).to(device)
        perturbation, adv_image = G(image, watermark)
        
        adv_image_ = reverse_transform(adv_image.squeeze(0).cpu())
        adv_image_.save(f"{save_dir_advimage}/adv_image_epoch_{epoch}.png")
        diffusion_image = pipe(prompt="A painting",image=adv_image, strength=0.1,).images[0]
        diffusion_image.save(f"{save_dir_difimage}/diffusion_image_epoch_{epoch}.png")
        
        # ### 
        # if epoch % 5 == 0 and epoch != 0:
        #     G.eval()
        #     adv_metrics = evaluate_adversarial_quality(G, eval_dataloader, device)
        #     print("\nAdversarial Example Quality Metrics:")
        #     print(f"MSE: {adv_metrics['mse']:.8f}")
        #     print(f"PSNR: {adv_metrics['psnr']:.8f} dB")
        #     print(f"SSIM: {adv_metrics['ssim']:.8f}")
        # # Save models every 10 epochs
        # if (epoch) % 10 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'generator_state_dict': G.state_dict(),
        #         'discriminator_state_dict': D.state_dict(),
        #         'optimizer_G_state_dict': optimizer_G.state_dict(),
        #         'optimizer_D_state_dict': optimizer_D.state_dict(),
        #     }, f"{save_dir}/checkpoint_epoch_{epoch}.pth")


if __name__ == "__main__":
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker = None,
        requires_safety_checker = False,
    ).to('cuda')
    pipe.enable_model_cpu_offload()
    
    args = get_args()
    main(args, pipe)
