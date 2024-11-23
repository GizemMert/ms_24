import torch.nn as nn
import timm
from torchvision.models import efficientnet_b0


class EfficientnetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientnetClassifier, self).__init__()

        # Load EfficientNet-B0 without pre-trained weights
        self.model = efficientnet_b0(weights=None)

        # Modify the classifier to match the number of classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)
