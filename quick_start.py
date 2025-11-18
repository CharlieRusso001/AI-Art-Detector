"""
Quick start script to organize data and start training.
Run this after organizing your data to begin training immediately.
"""
import subprocess
import sys
import os


def main():
    print("=" * 60)
    print("AI Art vs Real Art Classification - Quick Start")
    print("=" * 60)
    
    # Check if data is already organized
    if os.path.exists("data/train/real") and os.path.exists("data/train/ai"):
        print("\n[INFO] Data already organized in data/ folder")
        print("       Skipping data organization step...")
    else:
        print("\n[STEP 1] Organizing data...")
        if not os.path.exists("unsorteddata"):
            print("[ERROR] unsorteddata folder not found!")
            print("        Please ensure your images are in unsorteddata/RealArt/ and unsorteddata/AiArtData/")
            return
        
        result = subprocess.run([sys.executable, "organize_data.py", "--source", "unsorteddata"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("[ERROR] Data organization failed!")
            print(result.stderr)
            return
        print("[SUCCESS] Data organized!")
    
    # TensorBoard info
    print("\n[STEP 2] TensorBoard setup...")
    print("[INFO] TensorBoard is ready! After training starts, run:")
    print("       tensorboard --logdir runs")
    print("       Then open http://localhost:6006 in your browser")
    
    # Start training
    print("\n[STEP 3] Starting training...")
    print("=" * 60)
    print("\nTraining will begin now. You can:")
    print("  - Watch progress in the console")
    print("  - View graphs at the wandb dashboard (link will appear)")
    print("  - Press Ctrl+C to stop training early")
    print("\n" + "=" * 60 + "\n")
    
    # Run training
    subprocess.run([sys.executable, "train.py", "--data-root", "data"])


if __name__ == "__main__":
    main()

