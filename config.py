"""
Configuration file for galaxy classification training.
"""
import os


class Config:
    """Training configuration parameters."""
    
    # Data paths
    DATA_ROOT = "data"  # Root directory containing 'train' and 'val' folders
    
    # Model parameters
    NUM_CLASSES = None  # Will be determined from data
    PRETRAINED = True  # Use ImageNet pretrained weights
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001  # Start with this, will be reduced adaptively
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # Image parameters
    IMG_SIZE = 224
    
    # Data loading
    NUM_WORKERS = 4
    
    # Device
    # Will be auto-detected in train.py based on torch.cuda.is_available()
    
    # TensorBoard
    LOG_DIR = "runs"  # Directory for TensorBoard logs
    
    # Checkpointing
    SAVE_DIR = "checkpoints"
    SAVE_EVERY = 5  # Save checkpoint every N epochs
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 999  # Effectively disabled (set very high)
    EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum change to qualify as improvement
    ENABLE_EARLY_STOPPING = False  # Set to False to disable early stopping completely

