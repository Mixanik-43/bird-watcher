import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from birdlab.data import Birds, transforms
from birdlab.models import BirdClassifier
from birdlab.training import run_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cub8")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--output", default="runs/resnet")
    args = p.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = {s: DataLoader(Birds(args.data, s, transforms(s == "train")), batch_size=32,
                             shuffle=s == "train", num_workers=0) for s in ("train", "val", "test")}
    classes = json.loads((Path(args.data) / "classes.json").read_text())
    model = BirdClassifier(len(classes)).to(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history, best, start = [], -1, time.perf_counter()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        if epoch == 3:
            model.set_stage("last_block")
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        row = dict(epoch=epoch, train=run_epoch(model, loaders["train"], device, optimizer),
                   val=run_epoch(model, loaders["val"], device))
        history.append(row)
        print(row, flush=True)
        if row["val"]["accuracy"] > best:
            best = row["val"]["accuracy"]
            torch.save(dict(state_dict=model.state_dict(), classes=classes), output / "model.pt")
    model.load_state_dict(torch.load(output / "model.pt", map_location=device, weights_only=True)["state_dict"])
    report = dict(device=device, torch=torch.__version__, seconds=time.perf_counter()-start,
                  peak_memory_gb=torch.cuda.max_memory_allocated()/1e9 if device == "cuda" else None,
                  history=history, test=run_epoch(model, loaders["test"], device))
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()
