"""
Simple script to test if an image is AI-generated or real art.

Usage:
    python test_image.py <path_to_your_image>
    python test_image.py <path_to_your_image> <path_to_model>

The script will automatically use 'best_model.pth' in the current directory if no model is specified.
"""
import sys
import os
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from model import create_alexnet

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <path_to_image> [path_to_model]")
        print("\nExamples:")
        print("  python test_image.py my_image.jpg")
        print("  python test_image.py my_image.jpg best_model.pth")
        print("  python test_image.py my_image.jpg checkpoints/best_model.pth")
        print("\nNote: If no model is specified, the script will look for 'best_model.pth'")
        print("      in the current directory.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Find model checkpoint
    model_path = None
    if len(sys.argv) >= 3:
        model_path = sys.argv[2]
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)
    else:
        # Try to find the best model automatically (prioritize current directory)
        if os.path.exists("best_model.pth"):
            model_path = "best_model.pth"
        elif os.path.exists("checkpoints/best_model.pth"):
            model_path = "checkpoints/best_model.pth"
        else:
            # Look for any checkpoint in checkpoints directory
            checkpoints_dir = Path("checkpoints")
            if checkpoints_dir.exists():
                checkpoints = list(checkpoints_dir.glob("*.pth"))
                if checkpoints:
                    # Get the most recent checkpoint
                    model_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
    
    if model_path is None or not os.path.exists(model_path):
        print("Error: No trained model found!")
        print("\nPlease provide the path to your trained model:")
        print("  python test_image.py <image_path> <model_path>")
        print("\nOr make sure 'best_model.pth' is in the current directory.")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model (2 classes: ai and real)
    num_classes = 2
    class_names = ["ai", "real"]  # Sorted alphabetically: ai=0, real=1
    
    try:
        model = create_alexnet(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure the model file is valid and matches the AlexNet architecture.")
        sys.exit(1)
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Make prediction
    print(f"\nAnalyzing: {image_path}")
    print("-" * 50)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    # Get probabilities for both classes
    ai_prob = probabilities[0][0].item()
    real_prob = probabilities[0][1].item()
    
    # Display results
    print(f"\n{'='*60}")
    print(f"  RESULT: {predicted_class.upper()}")
    print(f"{'='*60}")
    print(f"  Confidence: {confidence_score*100:.2f}%")
    print(f"\n  Detailed probabilities:")
    print(f"    AI-generated:  {ai_prob*100:.2f}%")
    print(f"    Real art:      {real_prob*100:.2f}%")
    print(f"{'='*60}\n")
    
    if predicted_class == "ai":
        print("  ⚠️  This image appears to be AI-GENERATED")
    else:
        print("  ✓ This image appears to be REAL ART")
    print()

if __name__ == "__main__":
    main()

