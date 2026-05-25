import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BaselineModel(nn.Module):
    def __init__(self, num_classes=5):
        super(BaselineModel, self).__init__()
        pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(pretrained_model.children())[:-1])
        num_features = pretrained_model.fc.in_features
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        emb = torch.flatten(x, 1)
        out = self.classifier(emb)
        return emb, out


if __name__ == "__main__":
    model = BaselineModel()
    print(model)

    sample_input = torch.randn(2, 3, 224, 224)  # Example input tensor
    emb, out = model(sample_input)
    print("Embedding shape:", emb.shape) # Should be (2, 512)
    print("Output shape:", out.shape) # Should be (2, 5)
