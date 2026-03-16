import torch
import torch.nn as nn
import os

os.makedirs("models", exist_ok=True)

class BioViLClassificationHead(nn.Module):
    def __init__(self, in_features=2048, num_classes=14):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(in_features, num_classes)

    def forward(self, features):
        x = self.pool(features)
        x = x.flatten(1)
        return torch.sigmoid(self.fc(x))

class ReflexiveProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 2048)

    def forward(self, x):
        return self.proj(x)

classifier = BioViLClassificationHead()
torch.save(classifier.state_dict(), "models/biovil_classifier.pt")
print("Saved: models/biovil_classifier.pt")

projector = ReflexiveProjector()
torch.save(projector.state_dict(), "models/projector.pt")
print("Saved: models/projector.pt")

print("Done.")
