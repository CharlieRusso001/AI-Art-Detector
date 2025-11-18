# AI Art vs Real Art Classification with AlexNet

A PyTorch-based training system for classifying AI-generated art vs real art using AlexNet architecture. Includes automatic data organization and Weights & Biases integration for training visualization.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Organize Your Data

You have two options:

#### Option A: Automatic Organization (Recommended)

If your images are in `unsorteddata/RealArt/` and `unsorteddata/AiArtData/`, run:

```bash
python organize_data.py --source unsorteddata
```

This will:
- Automatically find all images in the nested folders
- Split them into train/val sets (80/20 by default)
- Organize them into the proper structure:
  ```
  data/
  ├── train/
  │   ├── real/
  │   └── ai/
  └── val/
      ├── real/
      └── ai/
  ```

#### Option B: Manual Organization

Organize your images manually in the following structure:

```
data/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ai/
│       ├── image1.jpg
│       └── ...
└── val/
    ├── real/
    │   └── ...
    └── ai/
        └── ...
```

**Important:**
- Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`
- Both `train` and `val` folders must have the same class subdirectories (`real` and `ai`)
- For best results, try to balance the number of images in each class

### 3. Training Metrics (CSV Logging)

The training script automatically saves all metrics to a CSV file (`training_log.csv` by default):
- Training and validation loss
- Training and validation accuracy
- Learning rate
- Timestamp for each epoch

The CSV file updates after each epoch, so you can:
- Open it in Excel, Google Sheets, or any spreadsheet app
- Create graphs and charts of your training progress
- View metrics in real-time as training progresses

## Usage

### Step 1: Organize Data (if using unsorted data)

```bash
python organize_data.py --source unsorteddata --val-split 0.2
```

Options:
- `--source`: Source directory with RealArt and AiArtData folders (default: `unsorteddata`)
- `--output`: Output directory (default: current directory, creates `data/` folder)
- `--val-split`: Fraction for validation set (default: 0.2 = 20%)
- `--seed`: Random seed for reproducibility (default: 42)

### Step 2: Train the Model

```bash
python train.py --data-root data
```

### Advanced Training Options

```bash
python train.py \
    --data-root data \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001 \
    --pretrained \
    --run-name "experiment-1" \
    --img-size 224
```

### Command Line Arguments

- `--data-root`: Root directory containing train/val folders (default: `data`)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--pretrained`: Use ImageNet pretrained weights (default: True) - **Recommended**
- `--no-pretrained`: Train from scratch
- `--save-dir`: Directory to save checkpoints (default: `checkpoints`)
- `--img-size`: Input image size (default: 224)
- `--csv-file`: CSV file to save training metrics (default: `training_log.csv`)

## Output

- **Checkpoints**: Saved in `checkpoints/` directory
  - `best_model.pth`: Best model based on validation accuracy
  - `checkpoint_epoch_N.pth`: Checkpoints saved every 5 epochs
- **Training Metrics CSV**: `training_log.csv` (or custom filename)
  - Updates after each epoch with all training metrics
  - Can be opened in Excel, Google Sheets, or any spreadsheet app
  - Columns: epoch, train_loss, train_acc, val_loss, val_acc, learning_rate, timestamp

## Model Architecture

The system uses AlexNet architecture with:
- 5 convolutional layers
- 3 fully connected layers
- Adaptive average pooling
- Dropout for regularization
- Transfer learning from ImageNet pretrained weights (recommended)

## Data Augmentation

Training images are automatically augmented with:
- Random horizontal flips (50% probability)
- Random rotations (±15 degrees)
- Color jitter (brightness and contrast variations)
- Normalization using ImageNet statistics

This helps the model generalize better and work with smaller datasets.

## Monitoring Training Progress

Once training starts, you'll see:

1. **Console Output**: Real-time progress bars showing loss and accuracy for each batch
2. **CSV File**: `training_log.csv` updates after each epoch with:
   - Training and validation loss
   - Training and validation accuracy
   - Learning rate
   - Timestamp

**To view your training progress:**
- Open `training_log.csv` in Excel, Google Sheets, or any spreadsheet app
- Create charts/graphs from the data
- The file updates after each epoch, so you can refresh to see new data

**Example CSV structure:**
```
epoch,train_loss,train_acc,val_loss,val_acc,learning_rate,timestamp
1,0.623456,65.4321,0.589012,68.9012,0.00100000,2024-01-01 12:00:00
2,0.512345,72.3456,0.501234,75.2345,0.00100000,2024-01-01 12:05:00
...
```

## Configuration

You can modify training parameters in `config.py`:
- Learning rate, momentum, weight decay
- Early stopping patience (stops if no improvement for 10 epochs)
- Checkpoint saving frequency
- Image size and batch size

## Tips

1. **Data Balance**: Try to have roughly equal numbers of images per class for best results
2. **Image Quality**: Higher resolution images generally work better
3. **Validation Set**: 20% validation split is a good default
4. **Transfer Learning**: Using `--pretrained` (default) typically gives much better results
5. **Monitoring**: Use wandb dashboard to monitor training progress and tune hyperparameters
6. **Early Stopping**: The model automatically stops if validation accuracy doesn't improve for 10 epochs

## Example Workflow

1. **Organize data**:
   ```bash
   python organize_data.py --source unsorteddata
   ```

2. **Start training**:
   ```bash
   python train.py --data-root data
   ```

3. **Monitor progress**: 
   - Watch console for real-time updates
   - Open `training_log.csv` in Excel/Sheets to see graphs
   - The CSV file updates after each epoch

4. **Use trained model**: 
   ```bash
   python inference.py --model checkpoints/best_model.pth --image path/to/image.jpg --num-classes 2 --class-names real ai
   ```

## Inference

After training, classify new images:

```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --num-classes 2 \
    --class-names real ai
```

Or classify all images in a directory:

```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --image path/to/images/ \
    --num-classes 2 \
    --class-names real ai
```

## Loading a Trained Model in Python

```python
import torch
from model import create_alexnet

# Load model
checkpoint = torch.load('checkpoints/best_model.pth')
model = create_alexnet(num_classes=2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference...
```

## Requirements

See `requirements.txt` for full list. Main dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- Pillow >= 10.0.0
- tqdm >= 4.65.0

## Troubleshooting

- **"No images found"**: Check that your source directory has `RealArt/RealArt/` and `AiArtData/AiArtData/` subdirectories
- **CUDA out of memory**: Reduce `--batch-size` (try 16 or 8)
- **CSV file not updating**: Make sure the file isn't open in another program (Excel/Sheets may lock it)
