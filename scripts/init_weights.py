import os
import torch
from aegisrad.models import ClinicalHead, ReflexiveProjector

os.makedirs("models", exist_ok=True)

classifier = ClinicalHead()
torch.save(classifier.state_dict(), "models/biovil_classifier.pt")
print("Saved: models/biovil_classifier.pt")

projector = ReflexiveProjector()
torch.save(projector.state_dict(), "models/projector.pt")
print("Saved: models/projector.pt")

print("Done.")
