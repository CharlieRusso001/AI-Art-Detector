"""
Generate matplotlib graphs from training_log.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


def plot_training_metrics(csv_file='training_log.csv', output_dir='graphs', show_plot=True):
    """
    Generate graphs from training CSV file.
    
    Args:
        csv_file: Path to training_log.csv file
        output_dir: Directory to save graph images
        show_plot: Whether to display the plot interactively
    """
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        print(f"Make sure training has started and generated the CSV file.")
        return
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if we have data
    if len(df) == 0:
        print("CSV file is empty. No data to plot.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['train_acc'], label='Training Accuracy', marker='o', linewidth=2, color='green')
    ax2.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='s', linewidth=2, color='orange')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['learning_rate'], marker='o', linewidth=2, color='purple')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale for learning rate
    
    # Plot 4: Combined view (Loss and Accuracy on same plot with dual y-axis)
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linewidth=2, marker='o')
    line2 = ax4.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red', linewidth=2, marker='s')
    line3 = ax4_twin.plot(df['epoch'], df['train_acc'], label='Train Acc', color='green', linewidth=2, linestyle='--', marker='^')
    line4 = ax4_twin.plot(df['epoch'], df['val_acc'], label='Val Acc', color='orange', linewidth=2, linestyle='--', marker='v')
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12, color='black')
    ax4_twin.set_ylabel('Accuracy (%)', fontsize=12, color='black')
    ax4.set_title('Combined View', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'training_graphs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphs saved to: {output_path}")
    
    # Also save individual graphs
    # Loss only
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o', linewidth=2)
    ax_loss.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax_loss.legend(fontsize=11)
    ax_loss.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_loss)
    
    # Accuracy only
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    ax_acc.plot(df['epoch'], df['train_acc'], label='Training Accuracy', marker='o', linewidth=2, color='green')
    ax_acc.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='s', linewidth=2, color='orange')
    ax_acc.set_xlabel('Epoch', fontsize=12)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=12)
    ax_acc.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax_acc.legend(fontsize=11)
    ax_acc.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_acc)
    
    print(f"Individual graphs saved to: {output_dir}/")
    print(f"  - loss_curve.png")
    print(f"  - accuracy_curve.png")
    print(f"  - training_graphs.png (all graphs)")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Training Summary:")
    print("="*60)
    print(f"Total epochs: {len(df)}")
    print(f"Best validation accuracy: {df['val_acc'].max():.2f}% (epoch {df.loc[df['val_acc'].idxmax(), 'epoch']})")
    print(f"Best validation loss: {df['val_loss'].min():.4f} (epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})")
    print(f"Final training accuracy: {df['train_acc'].iloc[-1]:.2f}%")
    print(f"Final validation accuracy: {df['val_acc'].iloc[-1]:.2f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Generate graphs from training CSV file')
    parser.add_argument('--csv-file', type=str, default='training_log.csv',
                        help='Path to training CSV file (default: training_log.csv)')
    parser.add_argument('--output-dir', type=str, default='graphs',
                        help='Directory to save graph images (default: graphs)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the plot interactively (just save files)')
    
    args = parser.parse_args()
    
    plot_training_metrics(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        show_plot=not args.no_show
    )


if __name__ == "__main__":
    main()

