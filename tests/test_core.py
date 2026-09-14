import torch
from torch import nn
from birdlab.models import BirdClassifier, LoRALinear
from birdlab.vlm import answer_labels

torch.set_num_threads(2)


def test_lora_initialization_gradient_and_merge():
    torch.manual_seed(7)
    base = nn.Linear(5, 3)
    layer = LoRALinear(base, rank=2)
    x = torch.randn(4, 5)
    torch.testing.assert_close(layer(x), base(x))
    layer(x).square().sum().backward()
    assert layer.B.grad.abs().sum() > 0
    assert layer.A.grad.abs().sum() == 0
    assert base.weight.grad is None
    with torch.no_grad():
        layer.B.add_(torch.randn_like(layer.B) * .1)
    torch.testing.assert_close(layer(x), layer.merged()(x))


def test_resnet_frozen_state_and_unfreezing():
    model = BirdClassifier(8, pretrained=False)
    model.train()
    before = model.backbone.bn1.running_mean.clone()
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 8)
    logits.square().sum().backward()
    assert model.head.weight.grad is not None
    assert model.backbone.conv1.weight.grad is None
    torch.testing.assert_close(before, model.backbone.bn1.running_mean)
    model.set_stage("last_block")
    assert model.backbone.layer4[0].conv1.weight.requires_grad
    assert not model.backbone.layer3[0].conv1.weight.requires_grad
    assert not model.backbone.bn1.training


def test_mask_preserves_eos_even_if_equal_to_pad_id():
    ids = torch.tensor([[1, 2, 7, 9, 9], [1, 2, 3, 8, 9]])
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    labels = answer_labels(ids, attention, [2, 3])
    assert labels.tolist() == [[-100, -100, 7, 9, -100], [-100, -100, -100, 8, 9]]
