import json
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T


def open_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def transforms(train=False):
    spatial = [T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip()] if train else [T.Resize(256), T.CenterCrop(224)]
    return T.Compose(spatial + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


class Birds(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = Path(root)
        self.rows = [json.loads(line) for line in (self.root / "manifest.jsonl").read_text().splitlines()
                     if line.strip()]
        self.rows = [row for row in self.rows if row["split"] == split]
        self.transform = transform or transforms(False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(open_rgb(self.root / row["image"])), row["label"]


# Questions use binary CUB attributes. Missing/uncertain annotation is omitted,
# never converted into a negative answer or an invented not_visible label.
QUESTIONS = {
    "has_bill_shape::cone": ("Does the bird have a cone-shaped bill?", "Is the bill conical?"),
    "has_breast_color::white": ("Is the breast white?", "Does the bird have white on its breast?"),
    "has_wing_color::black": ("Are the wings black?", "Does the bird have black wing coloration?"),
    "has_wing_pattern::striped": ("Are the wings striped?", "Is there a striped pattern on the wings?"),
}


def make_qa(root, split, heldout_wording=False, balance=False, seed=42):
    import random
    root = Path(root)
    records = [json.loads(x) for x in (root / "manifest.jsonl").read_text().splitlines()]
    examples = []
    for row in records:
        if row["split"] != split:
            continue
        for attribute, answer in row["attributes"].items():
            if attribute in QUESTIONS:
                question = QUESTIONS[attribute][int(heldout_wording)] + " Answer only yes or no."
                examples.append(dict(image=str(root / row["image"]), image_id=row["image_id"],
                                     task=attribute, question=question, answer=answer))
    if balance:
        if split != "train":
            raise ValueError("Do not balance validation/test: report their actual distribution")
        rng, selected = random.Random(seed), []
        for task in QUESTIONS:
            groups = [[x for x in examples if x["task"] == task and x["answer"] == answer]
                      for answer in ("yes", "no")]
            n = min(map(len, groups))
            for group in groups:
                selected.extend(rng.sample(group, n))
        rng.shuffle(selected)
        examples = selected
    return examples
