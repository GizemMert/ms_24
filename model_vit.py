import torch.nn as nn
import timm


class DeiTTinyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DeiTTinyClassifier, self).__init__()

        self.model = timm.create_model('deit_tiny_patch16_224', pretrained=False)


        self.model.head = nn.Linear(self.model.num_features, num_classes)

    def forward(self, x):
        return self.model(x)
