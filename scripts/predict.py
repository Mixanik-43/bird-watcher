import argparse
import torch
from birdlab.data import open_rgb, transforms
from birdlab.models import BirdClassifier

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("--checkpoint", default="runs/resnet/model.pt")
args = p.parse_args()
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
model = BirdClassifier(len(checkpoint["classes"]), pretrained=False)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
with torch.inference_mode():
    scores = model(transforms()(open_rgb(args.image))[None]).softmax(-1)[0]
for i in scores.argsort(descending=True)[:3]:
    print(checkpoint["classes"][i], round(scores[i].item(), 4))
