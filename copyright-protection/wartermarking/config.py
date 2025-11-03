# from dataclasses import dataclass

# @dataclass
# class TrainingConfig:
#     batch_size: int = 8
#     num_epochs: int = 200
#     lr: float = 0.001
#     beta1: float = 0.5
#     beta2: float = 0.999
#     alpha: float = 1.0  # GAN loss weight
#     beta: float = 10   # Perturbation loss weight
#     c: float = 10/255      # Perturbation bound
#     watermark_region: float = 4.0
#     checkpoint: str = None #"checkpoints/20241106-151538/checkpoint_epoch_90.pth"
#     save_dir: str = "checkpoints"
    
# @dataclass
# class ModelConfig:
#     input_channels: int = 3
#     vae_path: str = "runwayml/stable-diffusion-v1-5"