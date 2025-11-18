"""
Data loading utilities for galaxy classification dataset.
Expects directory structure:
    data/
        train/
            spiral/
                image1.jpg
                image2.jpg
                ...
            elliptical/
                image1.jpg
                ...
            ...
        val/
            spiral/
                ...
            elliptical/
                ...
"""
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch


class GalaxyDataset(Dataset):
    """Dataset class for loading galaxy images."""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to directory containing class subdirectories
            transform: Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all class names from subdirectories
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_num_classes(self):
        return len(self.classes)


def get_data_loaders(data_root, batch_size=32, num_workers=4, img_size=224):
    """
    Create data loaders for training and validation.
    
    Args:
        data_root: Root directory containing 'train' and 'val' subdirectories
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        img_size: Target image size (will be resized to this)
    
    Returns:
        train_loader, val_loader, num_classes, class_names
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Only basic transforms for validation
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory not found: {val_dir}")
    
    train_dataset = GalaxyDataset(train_dir, transform=train_transform)
    val_dataset = GalaxyDataset(val_dir, transform=val_transform)
    
    # Ensure both datasets have the same classes
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Training and validation sets must have the same classes!")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    num_classes = train_dataset.get_num_classes()
    class_names = train_dataset.classes
    
    return train_loader, val_loader, num_classes, class_names

