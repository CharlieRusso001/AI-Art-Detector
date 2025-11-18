"""
Script to organize unsorted art images into train/val structure for binary classification.
Organizes AI-generated art vs Real art.
"""
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def get_image_files(directory):
    """Get all image files from a directory recursively."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files


def organize_data(source_dir, output_dir, val_split=0.3, seed=42):
    """
    Organize images from source directory into train/val structure.
    
    Args:
        source_dir: Directory containing RealArt and AiArtData folders
        output_dir: Output directory for organized data (will create data/ folder)
        val_split: Fraction of data to use for validation (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Paths
    real_art_source = os.path.join(source_dir, 'RealArt', 'RealArt')
    ai_art_source = os.path.join(source_dir, 'AiArtData', 'AiArtData')
    
    # Output structure
    output_data_dir = os.path.join(output_dir, 'data')
    train_real_dir = os.path.join(output_data_dir, 'train', 'real')
    train_ai_dir = os.path.join(output_data_dir, 'train', 'ai')
    val_real_dir = os.path.join(output_data_dir, 'val', 'real')
    val_ai_dir = os.path.join(output_data_dir, 'val', 'ai')
    
    # Create directories
    for dir_path in [train_real_dir, train_ai_dir, val_real_dir, val_ai_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all image files
    print("Scanning for images...")
    real_images = get_image_files(real_art_source) if os.path.exists(real_art_source) else []
    ai_images = get_image_files(ai_art_source) if os.path.exists(ai_art_source) else []
    
    print(f"Found {len(real_images)} real art images")
    print(f"Found {len(ai_images)} AI art images")
    
    if len(real_images) == 0 and len(ai_images) == 0:
        print("Error: No images found! Check your source directory paths.")
        return
    
    # Shuffle and split
    random.shuffle(real_images)
    random.shuffle(ai_images)
    
    # Calculate split points
    real_val_count = int(len(real_images) * val_split)
    ai_val_count = int(len(ai_images) * val_split)
    
    # Split real art
    real_val = real_images[:real_val_count]
    real_train = real_images[real_val_count:]
    
    # Split AI art
    ai_val = ai_images[:ai_val_count]
    ai_train = ai_images[ai_val_count:]
    
    print(f"\nSplitting data:")
    print(f"  Real Art - Train: {len(real_train)}, Val: {len(real_val)}")
    print(f"  AI Art   - Train: {len(ai_train)}, Val: {len(ai_val)}")
    
    # Copy files
    print("\nCopying files...")
    
    # Copy real art
    print("Copying real art images...")
    for img_path in tqdm(real_train, desc="Real train"):
        filename = os.path.basename(img_path)
        dest = os.path.join(train_real_dir, filename)
        # Handle duplicate filenames
        counter = 1
        while os.path.exists(dest):
            name, ext = os.path.splitext(filename)
            dest = os.path.join(train_real_dir, f"{name}_{counter}{ext}")
            counter += 1
        shutil.copy2(img_path, dest)
    
    for img_path in tqdm(real_val, desc="Real val"):
        filename = os.path.basename(img_path)
        dest = os.path.join(val_real_dir, filename)
        counter = 1
        while os.path.exists(dest):
            name, ext = os.path.splitext(filename)
            dest = os.path.join(val_real_dir, f"{name}_{counter}{ext}")
            counter += 1
        shutil.copy2(img_path, dest)
    
    # Copy AI art
    print("Copying AI art images...")
    for img_path in tqdm(ai_train, desc="AI train"):
        filename = os.path.basename(img_path)
        dest = os.path.join(train_ai_dir, filename)
        counter = 1
        while os.path.exists(dest):
            name, ext = os.path.splitext(filename)
            dest = os.path.join(train_ai_dir, f"{name}_{counter}{ext}")
            counter += 1
        shutil.copy2(img_path, dest)
    
    for img_path in tqdm(ai_val, desc="AI val"):
        filename = os.path.basename(img_path)
        dest = os.path.join(val_ai_dir, filename)
        counter = 1
        while os.path.exists(dest):
            name, ext = os.path.splitext(filename)
            dest = os.path.join(val_ai_dir, f"{name}_{counter}{ext}")
            counter += 1
        shutil.copy2(img_path, dest)
    
    print(f"\n[SUCCESS] Data organized successfully!")
    print(f"  Output directory: {output_data_dir}")
    print(f"\n  Training set:")
    print(f"    - Real: {len(real_train)} images")
    print(f"    - AI:   {len(ai_train)} images")
    print(f"\n  Validation set:")
    print(f"    - Real: {len(real_val)} images")
    print(f"    - AI:   {len(ai_val)} images")
    print(f"\nYou can now run: python train.py --data-root {output_data_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize art images into train/val structure')
    parser.add_argument('--source', type=str, default='unsorteddata',
                        help='Source directory containing RealArt and AiArtData folders')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory (will create data/ folder here)')
    parser.add_argument('--val-split', type=float, default=0.3,
                        help='Fraction of data for validation (default: 0.3 = 30%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    organize_data(args.source, args.output, args.val_split, args.seed)

