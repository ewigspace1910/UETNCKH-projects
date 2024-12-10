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
from evaluate import evaluate_adversarial_quality
from utils.meter import AverageMeter

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--name", type=str, default="trial")
    args.add_argument("--checkpoint", type=str, default=None)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--iheight", type=int, default=256, help="size of image height")
    args.add_argument("--iwidth", type=int, default=256, help="size of image width")
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

    args.add_argument("--n_critic", type=int, default=5, help="number of training iterations for WGAN discriminator")
    return args.parse_args()


def main(args, pipe):
    test_image = Image.open("./data/imagenet/IMAGENET_DOG/n02109961_48_n02109961.JPEG").convert("RGB")
    test_image_size = test_image.size[::-1]

    watermark = create_watermark("ANIMAL", (args.iwidth, args.iheight)).convert("RGB")
    watermark.save("masking_sample.png")
    # transform = transforms.Compose([
    #     transforms.Resize((args.iwidth, args.iheight)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # reverse_transform = transforms.Compose([
    #     transforms.Normalize(mean=[-m / s for m, s in zip((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))], std=[1 / s for s in (0.5, 0.5, 0.5)]),
    #     transforms.ToPILImage(),
    # ])
    # image = transform(test_image).unsqueeze(0).to(device)
    # watermark = transform(watermark).unsqueeze(0).to(device)
    

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
