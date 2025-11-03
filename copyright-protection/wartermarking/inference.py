import torch
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import warnings
import argparse

from models import Generator
from data_loader import create_watermark

warnings.filterwarnings("ignore")

# args = parser.parse_args()
def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--prompt", '-p',type=str, default="A photo", help="comma separated prompts")
    args.add_argument("--image", '-i', type=str, default="data/wikiart/Early_Renaissance/andrea-del-castagno_crucifixion-1.jpg")
    args.add_argument("--watermark", '-w', type=str, default="aaron-siskind")
    args.add_argument("--strength", '-s', type=float, default=0.1)
    args.add_argument("--checkpoints", type=str, default='checkpoints/20241203-115147/checkpoint_epoch_130.pth')
    return args.parse_args()


def generate_with_gan(pipe, generator, image, watermark, prompt, strength=0.1):
    # Generate adversarial image
    with torch.no_grad():
        adversarial_image = generator(image, watermark)

    # Generate from adversarial image
    output = pipe(
        prompt=prompt,
        image=adversarial_image,
        strength=strength,
    ).images[0]
    del pipe

    return output, image


def generate_without_gan(pipe, image, prompt, strength=0.1):
    # Generate from original image
    output = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
    ).images[0]
    # del pipe
    return output, image


def main(args):
    # Test with Stable Diffusion
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker = None,
        requires_safety_checker = False,
    ).to('cuda')
    pipe.enable_model_cpu_offload()
    print("loaded model")
    
    
    generator = Generator().to('cuda')
    checkpoint = torch.load(args.checkpoints)
    generator.load_state_dict(checkpoint['generator_state_dict'])

    # Test cases
    test_prompts = args.prompt.split(",")
    print(test_prompts)

    image = Image.open(args.image).convert("RGB")
    image_size = image.size  # Save original image size
    print(image_size)
    watermark = create_watermark(args.watermark, image_size).convert("RGB")

    # convert to tensor
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    resize = transforms.Resize(image_size[::-1])
    image_ = transform(image).unsqueeze(0).to('cuda')
    watermark_ = transform(watermark).unsqueeze(0).to('cuda')

    for prompt in test_prompts:
        output, _ = generate_with_gan(pipe, 
            generator, image_, watermark_, prompt, strength=args.strength)
        output_resized = resize(output)
        output_resized = resize(output).convert("RGB")
        output_resized.save(f"output/{prompt}_with_gan.png")
    for prompt in test_prompts:
        output, _ = generate_without_gan(pipe, image, prompt, strength=args.strength)
        output.save(f"output/{prompt}_without_gan.png")

if __name__ == "__main__":
    args = get_args()
    main(args)
