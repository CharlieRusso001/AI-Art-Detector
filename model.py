"""
AlexNet model definition for galaxy classification.
"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet architecture adapted for custom number of classes.
    """
    
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_alexnet(num_classes, pretrained=False):
    """
    Create an AlexNet model.
    
    Args:
        num_classes: Number of output classes
        pretrained: If True, load ImageNet pretrained weights (only features)
    
    Returns:
        AlexNet model
    """
    model = AlexNet(num_classes=num_classes)
    
    if pretrained:
        # Try to load pretrained weights from torchvision
        try:
            import torchvision.models as models
            # Use new weights API to avoid deprecation warnings
            try:
                # Try new API first (torchvision >= 0.13)
                pretrained_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
            except AttributeError:
                # Fall back to old API for older torchvision versions
                pretrained_model = models.alexnet(pretrained=True)
            
            # Copy feature weights
            model.features.load_state_dict(pretrained_model.features.state_dict())
            
            # Optionally copy classifier weights (except last layer)
            # This helps with transfer learning
            pretrained_classifier = pretrained_model.classifier.state_dict()
            model_classifier = model.classifier.state_dict()
            
            # Copy all layers except the last one
            for name, param in pretrained_classifier.items():
                if name in model_classifier:
                    if model_classifier[name].shape == param.shape:
                        model_classifier[name] = param
            
            model.classifier.load_state_dict(model_classifier)
            print("Loaded pretrained ImageNet weights!")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Training from scratch...")
    
    return model

