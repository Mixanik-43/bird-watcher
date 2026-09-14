"""Download official CUB archive and export a small, auditable folder dataset."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import tarfile
import urllib.request

URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
MD5 = "97eceeb196236b17998738112f37df78"
# Eurasian species present in CUB; not a complete Moscow park inventory.
CLASSES = {46: "Gadwall", 48: "European_Goldfinch", 87: "Mallard", 107: "Common_Raven",
           118: "House_Sparrow", 135: "Bank_Swallow", 136: "Barn_Swallow", 185: "Bohemian_Waxwing"}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table(path):
    return {int(line.split()[0]): line.strip().split(maxsplit=1)[1]
            for line in Path(path).read_text().splitlines() if line.strip()}


def prepare(source, destination, seed=42):
    from birdlab.data import QUESTIONS
    source, destination = Path(source), Path(destination)
    if (destination / "manifest.jsonl").exists():
        raise FileExistsError("Dataset already exists; choose a new output directory")
    classes = table(source / "classes.txt")
    for key, name in CLASSES.items():
        assert classes[key].split(".", 1)[1] == name, (key, classes[key])
    images = table(source / "images.txt")
    labels = {k: int(v) for k, v in table(source / "image_class_labels.txt").items()}
    official_train = {k: int(v) for k, v in table(source / "train_test_split.txt").items()}
    selected = {k for k, v in labels.items() if v in CLASSES}
    attr_path = source / "attributes" / "attributes.txt"
    if not attr_path.exists():
        attr_path = source.parent / "attributes.txt"
    names = table(attr_path)
    assert set(QUESTIONS) <= set(names.values()), "Attribute vocabulary mismatch"
    votes = defaultdict(list)
    with (source / "attributes" / "image_attribute_labels.txt").open() as f:
        for line in f:
            cols = line.split()
            image_id, attr_id, present, certainty = map(int, cols[:4])
            if image_id in selected and names[attr_id] in QUESTIONS and certainty >= 3:
                votes[image_id, names[attr_id]].append(present)
    attributes = defaultdict(dict)
    for (image_id, name), values in votes.items():
        # Conflicting annotations are omitted. Certainty 1/2 is not a negative.
        if len(set(values)) == 1:
            attributes[image_id][name] = "yes" if values[0] else "no"
    rng, splits = random.Random(seed), {}
    for class_id in CLASSES:
        ids = sorted(k for k in selected if labels[k] == class_id and official_train[k])
        rng.shuffle(ids)
        val_ids = set(ids[:max(1, round(len(ids) * .2))])
        for image_id in selected:
            if labels[image_id] == class_id:
                splits[image_id] = "test" if not official_train[image_id] else ("val" if image_id in val_ids else "train")
    rows, seen, skipped = [], set(), []
    # Test wins in an exact-duplicate collision: don't contaminate official test.
    for image_id in sorted(selected, key=lambda i: ({"test": 0, "val": 1, "train": 2}[splits[i]], i)):
        src = source / "images" / images[image_id]
        sha = digest(src)
        if sha in seen:
            skipped.append(image_id)
            continue
        seen.add(sha)
        label = list(CLASSES).index(labels[image_id])
        species = CLASSES[labels[image_id]].lower()
        relative = Path(splits[image_id]) / species / f"{image_id:05d}.jpg"
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        rows.append(dict(image_id=image_id, image=relative.as_posix(), label=label,
                         species=species, split=splits[image_id], sha256=sha,
                         attributes=attributes[image_id]))
    rows.sort(key=lambda r: r["image_id"])
    (destination / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (destination / "classes.json").write_text(json.dumps(list(CLASSES.values()), indent=2))
    stats = {"source": URL, "archive_md5": MD5, "seed": seed, "images": len(rows),
             "splits": dict(Counter(r["split"] for r in rows)), "duplicate_ids_removed": skipped,
             "attributes": {split: {task: dict(Counter(r["attributes"][task] for r in rows
                 if r["split"] == split and task in r["attributes"])) for task in QUESTIONS}
                 for split in ("train", "val", "test")}}
    (destination / "statistics.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--archive", type=Path, default=Path("data/CUB_200_2011.tgz"))
    parser.add_argument("--output", type=Path, default=Path("data/cub8"))
    args = parser.parse_args()
    if args.source is None:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        if not args.archive.exists():
            temporary = args.archive.with_suffix(".download")
            urllib.request.urlretrieve(URL, temporary)
            temporary.replace(args.archive)
        if digest(args.archive, "md5") != MD5:
            raise ValueError("Archive checksum mismatch; remove it and download again")
        with tarfile.open(args.archive) as archive:
            archive.extractall(args.archive.parent, filter="data")
        args.source = args.archive.parent / "CUB_200_2011"
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
