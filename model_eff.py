import torch.nn as nn
import timm


class Efficientnetv2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnetv2Classifier, self).__init__()

        model = timm.create_model('efficientnetv2_b0', pretrained=False)


        self.model.head = nn.Linear(self.model.num_features, num_classes)

    def forward(self, x):
        return self.model(x)
