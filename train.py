"""
Main training script for AI art vs real art classification using AlexNet.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
from pathlib import Path
from datetime import datetime
import csv

from model import create_alexnet
from data_loader import get_data_loaders
from config import Config


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, loss, acc, save_path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
    }, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train AlexNet for AI art vs real art classification')
    parser.add_argument('--data-root', type=str, default=Config.DATA_ROOT,
                        help='Root directory containing train/val folders')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', default=Config.PRETRAINED,
                        help='Use pretrained ImageNet weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Train from scratch')
    parser.add_argument('--save-dir', type=str, default=Config.SAVE_DIR,
                        help='Directory to save checkpoints')
    parser.add_argument('--img-size', type=int, default=Config.IMG_SIZE,
                        help='Input image size')
    parser.add_argument('--csv-file', type=str, default='training_log.csv',
                        help='CSV file to save training metrics (default: training_log.csv)')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping (train for full number of epochs)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize CSV logging
    csv_file_path = args.csv_file
    print(f"\nTraining metrics will be saved to: {csv_file_path}")
    print("The CSV file updates after each epoch - you can open it in Excel or any spreadsheet app!")
    
    # Create CSV file with headers
    csv_exists = os.path.exists(csv_file_path)
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer_csv = csv.writer(csvfile)
        if not csv_exists:
            # Write header row
            writer_csv.writerow([
                'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
                'learning_rate', 'timestamp'
            ])
            print("Created new CSV file with headers.")
        else:
            print("Appending to existing CSV file.")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, num_classes, class_names = get_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=Config.NUM_WORKERS,
        img_size=args.img_size
    )
    
    print(f"Found {num_classes} classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_alexnet(num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Use ReduceLROnPlateau for adaptive learning rate reduction
    # This reduces LR when validation loss plateaus, leading to faster convergence
    # This is more efficient than fixed step scheduling
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',  # Minimize validation loss
        factor=0.5,  # Reduce LR by half when plateau detected
        patience=5,  # Wait 5 epochs without improvement before reducing
        min_lr=1e-6,  # Minimum learning rate
        cooldown=2  # Wait 2 epochs after LR reduction before checking again
    )
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    if args.no_early_stop:
        print("Early stopping is DISABLED - training for full {} epochs".format(args.epochs))
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling - ReduceLROnPlateau uses validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)  # ReduceLROnPlateau adaptively reduces LR based on val loss
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print message if learning rate was reduced
        if current_lr < old_lr:
            print(f"  [LR REDUCED] Learning rate reduced from {old_lr:.8f} to {current_lr:.8f}")
        
        # Log to CSV file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([
                epoch, 
                f'{train_loss:.6f}', 
                f'{train_acc:.4f}', 
                f'{val_loss:.6f}', 
                f'{val_acc:.4f}', 
                f'{current_lr:.8f}',
                timestamp
            ])
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  CSV updated: {csv_file_path}")
        
        # Save checkpoint
        if epoch % Config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.save_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)
            patience_counter = 0
            print(f"  [NEW BEST] Validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping (disabled by default - train for full epochs)
        # Only stop early if explicitly enabled via config
        if Config.ENABLE_EARLY_STOPPING and not args.no_early_stop:
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"\nTraining metrics saved to: {csv_file_path}")
    print("You can open this CSV file in Excel, Google Sheets, or any spreadsheet app")
    print("to view and graph your training progress!")
    print("="*60)


if __name__ == "__main__":
    main()

