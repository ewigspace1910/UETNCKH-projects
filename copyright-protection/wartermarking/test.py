import pandas as pd
import numpy as np

def split_by_artist(csv_path, train_ratio=0.8, random_seed=42):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Initialize empty dataframes for train and eval
    train_df = pd.DataFrame(columns=df.columns)
    eval_df = pd.DataFrame(columns=df.columns)
    
    # Group by artist and split
    for artist, group in df.groupby('artist'):
        # Shuffle artist's images
        group = group.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # Calculate split index for this artist
        artist_split_idx = int(len(group) * train_ratio)
        
        # Split and append
        train_df = pd.concat([train_df, group[:artist_split_idx]])
        eval_df = pd.concat([eval_df, group[artist_split_idx:]])
    
    # Shuffle final datasets
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    eval_df = eval_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Save to files
    output_dir = '/'.join(csv_path.split('/')[:-1])
    train_path = f"{output_dir}/train_classes.csv"
    eval_path = f"{output_dir}/eval_classes.csv"
    
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    
    print(f"Total images: {len(df)}")
    print(f"Training set: {len(train_df)} images")
    print(f"Eval set: {len(eval_df)} images")
    
    # Print distribution per artist
    print("\nArtist distribution:")
    for artist in df['artist'].unique():
        train_count = len(train_df[train_df['artist'] == artist])
        eval_count = len(eval_df[eval_df['artist'] == artist])
        print(f"{artist}: train={train_count}, eval={eval_count}")

if __name__ == "__main__":
    split_by_artist("data/imagenet/image_artist.csv")