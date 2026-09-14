import torch
from torch import nn


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(-1) == labels).sum().item()
    if not total:
        raise ValueError("Empty dataset")
    return {"loss": loss_sum / total, "accuracy": correct / total}
