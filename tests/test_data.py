import json
from pathlib import Path
from PIL import Image
from birdlab.data import Birds, make_qa


def test_missing_attributes_are_not_negative_and_splits_stay_separate(tmp_path):
    rows = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (40, 60)).save(path)
        attrs = {"has_bill_shape::cone": "yes" if i % 2 else "no"}
        rows.append(dict(image_id=i, image=path.name, split=split, label=0, attributes=attrs))
    (tmp_path / "manifest.jsonl").write_text("\n".join(map(json.dumps, rows)))
    train = make_qa(tmp_path, "train", balance=True)
    val = make_qa(tmp_path, "val", heldout_wording=True)
    assert len(train) == 2 and len(val) == 1
    assert {x["image_id"] for x in train}.isdisjoint(x["image_id"] for x in val)
    assert val[0]["question"] != train[0]["question"]
    image, label = Birds(tmp_path, "test")[0]
    assert image.shape == (3, 224, 224) and label == 0
