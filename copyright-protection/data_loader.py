import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from typing import List, Tuple, Optional, Dict
from PIL import ImageDraw, ImageFont, Image
import random
import numpy as np

class WatermarkDataset(Dataset):
    """Custom Dataset for watermark training that uses artist names as watermarks"""
    
    def __init__(
        self,
        image_dir: str,
        classes_csv: str,
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
        is_test=False
    ):
        """
        Args:
            image_dir (str): Directory with all the images
            classes_csv (str): Path to CSV file containing filename, artist, genre
            transform (transforms.Compose, optional): Optional transform to be applied on images
            image_size (tuple): Size to resize images to
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        
        # Load and process the classes CSV
        self.metadata_df = pd.read_csv(classes_csv)
        
        # Verify that the images exist
        self.valid_files = []
        for idx, row in self.metadata_df.iterrows():
            filepath = os.path.join(image_dir, row['filename'])
            if os.path.exists(filepath):
                self.valid_files.append(idx)
 
        # Create mappings for artists and genres
        self.artists = sorted(self.metadata_df['artist'].unique())
        # self.genres = sorted(self.metadata_df['genre'].unique())
        self.artist_to_idx = {artist: idx for idx, artist in enumerate(self.artists)}
        # self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        
        # Default transform if none provided
        if self.transform is None:
            if is_test:
                self.transform = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Get metadata for this index
        metadata_idx = self.valid_files[idx]
        row = self.metadata_df.iloc[metadata_idx]
        
        # Load and transform image
        try:
            img_path = os.path.join(self.image_dir, row['filename'])
            image = Image.open(img_path).convert('RGB')
        except:
            # Skip if image is not found
            return self.__getitem__(idx + 1)

        artist_name = row['artist']
        watermark = create_watermark(artist_name, self.image_size).convert('RGB')

        if self.transform:
            image = self.transform(image)
            watermark = self.transform(watermark)
        
        # Create metadata dictionary
        metadata = {
            'filename': row['filename'],
            'artist': artist_name,
            # 'genre': row['genre'],
            'artist_idx': self.artist_to_idx[artist_name],
            # 'genre_idx': self.genre_to_idx[row['genre']],
            # 'watermark': watermark.shape,
        }
            
        return image, watermark, metadata

def create_dataloader(
    image_dir: str,
    classes_csv: str,
    batch_size: int = 1,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    shuffle: bool = True,
    is_test=False,
) -> Tuple[DataLoader, Dict]:
    """
    Creates a DataLoader for the watermark training pipeline
    
    Args:
        image_dir (str): Directory containing training images
        classes_csv (str): Path to CSV file containing filename, artist, genre
        batch_size (int): Number of samples per batch
        num_workers (int): Number of subprocesses for data loading
        image_size (tuple): Size to resize images to
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        Tuple[DataLoader, Dict]: PyTorch DataLoader object and metadata dictionary
    """
    dataset = WatermarkDataset(
        image_dir=image_dir,
        classes_csv=classes_csv,
        image_size=image_size,
        is_test=is_test
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=custom_collate
    )
    
    # Create metadata dictionary with mappings
    metadata = {
        'num_artists': len(dataset.artists),
        # 'num_genres': len(dataset.genres),
        'artist_to_idx': dataset.artist_to_idx,
        # 'genre_to_idx': dataset.genre_to_idx,
        'idx_to_artist': {v: k for k, v in dataset.artist_to_idx.items()},
        # 'idx_to_genre': {v: k for k, v in dataset.genre_to_idx.items()},
        'artists': dataset.artists,
        # 'genres': dataset.genres
    }
    
    return dataloader, metadata

def create_watermark(
    text: str,
    image_size: Tuple[int, int] = (512, 512),
    font_size: int = 30,
    line_spacing: float = 1.25,
    noise_level: float = 0.1,
) -> Image.Image:
    """
    Creates a watermark image with repeated text using semi-transparent gray
    
    Args:
        text (str): Text to use as watermark
        image_size (tuple): Size of output image (width, height)
        font_size (int): Font size for watermark text
        line_spacing (float): Spacing between text lines
        
    Returns:
        Image.Image: PIL Image containing the watermark
    """
    
    text = text.upper()
    img_width, img_height = image_size
    
    # Create new transparent image
    watermark_image = Image.new("RGBA", (img_width, img_height), (2, 2, 2, 255))
    draw = ImageDraw.Draw(watermark_image)
    
    # Calculate font size based on image width and text length
    adjusted_font_size = int(img_width / (len(text) * 0.65))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", adjusted_font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculate text dimensions and spacing
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_height = text_bbox[3] - text_bbox[1]
    line_height = text_height * line_spacing
    num_lines = int(img_height / line_height)
    
    # Draw watermark text lines with semi-transparent dark gray
    for i in range(num_lines):
        y = i * line_height
        draw.text((2, y), text, fill=(255, 255, 255, 255), font=font)
    
    # add small noise to watermark
    watermark_image = watermark_image.rotate(1)
    return watermark_image
        # Random horizontal offset
    #     x_offset = random.uniform(-20, 20)
        
    #     # Random opacity (alpha)
    #     base_alpha = 180  # Semi-transparent base
    #     alpha_noise = int(base_alpha * random.uniform(1 - noise_level, 1 + noise_level))
    #     alpha_noise = max(0, min(255, alpha_noise))
        
    #     # Random color variation
    #     color_noise = tuple(int(c * random.uniform(1 - noise_level, 1 + noise_level)) for c in (255, 255, 255))
        
    #     draw.text(
    #         (x_offset + 2, y), 
    #         text, 
    #         fill=color_noise + (alpha_noise,), 
    #         font=font
    #     )
    # watermark_array = np.array(watermark_image)

    # # Add random noise to the alpha channel
    # noise = np.random.normal(
    #     loc=0, 
    #     scale=noise_level * 50, 
    #     size=watermark_array.shape[:2]
    # )
    
    # # Modify alpha channel with noise
    # noise_mask = noise.astype(np.int16)
    # watermark_array[:, :, 3] = np.clip(
    #     watermark_array[:, :, 3].astype(np.int16) + noise_mask, 
    #     0, 
    #     255
    # )
    
    # # Slight color variation in background
    # background_noise = np.random.normal(
    #     loc=0, 
    #     scale=noise_level * 10, 
    #     size=watermark_array.shape[:2] + (3,)
    # )
    
    # watermark_array[:, :, :3] = np.clip(
    #     watermark_array[:, :, :3].astype(np.int16) + background_noise.astype(np.int16), 
    #     0, 
    #     255
    # )
    
    # return Image.fromarray(watermark_array)

# Example usage:
if __name__ == "__main__":
    # Example parameters
    image_dir = "data/wikiart"
    classes_csv = "data/wikiart/train_classes.csv"
    
    # Create dataloader
    dataloader, metadata = create_dataloader(
        image_dir=image_dir,
        classes_csv=classes_csv,
        batch_size=32
    )
    
    # Print dataset information
    print(f"Number of artists: {metadata['num_artists']}")
    # print(f"Number of genres: {metadata['num_genres']}")
    print(f"Available artists: {metadata['artists'][:5]}...")  # Show first 5 artists
    # print(f'Watermark shape: {metadata["watermark"]}')
    
    # Test the dataloader
    # for images, watermarks, batch_metadata in dataloader:
    #     print(f"Batch shapes: Images {images.shape}, Watermarks {watermarks.shape}")
    #     print(f"Sample metadata: {batch_metadata['filename'][0]}, "
    #           f"Artist: {batch_metadata['artist'][0]}, "
    #           f"Genre: {batch_metadata['genre'][0]}")
    #         #   f"Watemark shape: {batch_metadata['watermark']}")
    #     break  # Just test one batch

    for batch_idx, (images, watermarks, batch_metadata) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        # print(f"Images shape: {images.shape}")
        # print(f"Watermarks shape: {watermarks.shape}")
        # print(f"Metadata: {batch_metadata}")
        # break
