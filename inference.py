"""
Inference script for classifying galaxy images using a trained AlexNet model.
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
from pathlib import Path

from model import create_alexnet


def load_model(checkpoint_path, num_classes, device):
    """Load a trained model from checkpoint."""
    model = create_alexnet(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, img_size=224):
    """Preprocess image for inference."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict(model, image_tensor, class_names, device):
    """Make prediction on a single image."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, min(3, len(class_names)))
    
    results = {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'top3': [(class_names[idx], prob.item()) 
                for prob, idx in zip(top3_probs[0], top3_indices[0])]
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Classify galaxy images using trained AlexNet')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (e.g., checkpoints/best_model.pth)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file or directory of images')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of classes the model was trained on')
    parser.add_argument('--class-names', type=str, nargs='+',
                        help='List of class names in order (e.g., spiral elliptical irregular)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, args.num_classes, device)
    print("Model loaded successfully!")
    
    # Get class names
    if args.class_names:
        class_names = args.class_names
    else:
        # Default class names if not provided
        class_names = [f"class_{i}" for i in range(args.num_classes)]
        print(f"Warning: Class names not provided. Using default names: {class_names}")
    
    # Process image(s)
    image_path = Path(args.image)
    
    if image_path.is_file():
        # Single image
        print(f"\nClassifying: {image_path}")
        image_tensor = preprocess_image(str(image_path), args.img_size)
        results = predict(model, image_tensor, class_names, device)
        
        print(f"\nPrediction: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']*100:.2f}%")
        print("\nTop 3 predictions:")
        for class_name, prob in results['top3']:
            print(f"  {class_name}: {prob*100:.2f}%")
    
    elif image_path.is_dir():
        # Directory of images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            image_files.extend(image_path.glob(ext))
            image_files.extend(image_path.glob(ext.upper()))
        
        if not image_files:
            print(f"No images found in {image_path}")
            return
        
        print(f"\nFound {len(image_files)} images. Classifying...\n")
        
        for img_file in image_files:
            try:
                image_tensor = preprocess_image(str(img_file), args.img_size)
                results = predict(model, image_tensor, class_names, device)
                
                print(f"{img_file.name}:")
                print(f"  → {results['predicted_class']} ({results['confidence']*100:.2f}% confidence)")
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    else:
        print(f"Error: {args.image} is not a valid file or directory")


if __name__ == "__main__":
    main()

