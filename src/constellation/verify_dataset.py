"""Verify a Roboflow YOLO export and draw its boxes for a visual check.

Extracts the zip safely, then writes to --out:
  dataset_report.json        counts, classes, label problems, leakage checks
  figures/label_check_*.jpg  sample images with their ground-truth boxes
  figures/duplicate_*.jpg    the closest cross-split image pairs, if any

Run from the repository root:
  python -I src/constellation/verify_dataset.py \
      --zip data/raw/constellations/dataset.zip
"""
import argparse
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract, refusing any member that would land outside dest."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"unsafe path in zip: {member.filename}")
        z.extractall(dest)


def find_root(extracted: Path) -> Path:
    hits = sorted(extracted.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"no data.yaml under {extracted}")
    return hits[0].parent


def class_names(root: Path) -> list[str]:
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def source_stem(path: Path) -> str:
    """Original file stem: Roboflow names copies '<stem>.rf.<hash>.jpg'."""
    return re.split(r"\.rf\.", path.name)[0]


def parse_label(path: Path, n_classes: int):
    """Return (boxes, problems, kinds); boxes are (cls, cx, cy, w, h)."""
    boxes, problems, kinds = [], [], Counter()
    for i, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        where = f"{path.name}:{i}"
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            problems.append(f"{where} not numeric")
            continue
        if len(vals) == 5:
            kinds["box"] += 1
        elif len(vals) > 5 and (len(vals) - 5) % 3 == 0:
            kinds["box_plus_triplets_maybe_keypoints"] += 1
        elif len(vals) > 5 and (len(vals) - 1) % 2 == 0:
            kinds["polygon"] += 1
        else:
            problems.append(f"{where} has {len(vals)} columns")
            continue
        cls = int(vals[0])
        if vals[0] != cls or not 0 <= cls < n_classes:
            problems.append(f"{where} bad class {parts[0]}")
            continue
        if len(vals) >= 5:
            cx, cy, w, h = vals[1:5]
            if w <= 0 or h <= 0 or min(cx - w / 2, cy - h / 2) < -1e-3 \
                    or max(cx + w / 2, cy + h / 2) > 1 + 1e-3:
                problems.append(f"{where} box out of range")
                continue
            boxes.append((cls, cx, cy, w, h))
    return boxes, problems, kinds


def dhash(gray: np.ndarray) -> np.uint64:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def draw(img: np.ndarray, boxes, names) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    thick = max(1, round(max(h, w) / 400))
    for cls, cx, cy, bw, bh in boxes:
        p1 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
        p2 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
        cv2.rectangle(out, p1, p2, (0, 255, 255), thick)
        cv2.putText(out, names[cls], (p1[0], max(14, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick + 0.1, (0, 255, 255),
                    thick, cv2.LINE_AA)
    return out


def tile(images, cols: int, cell: int = 320) -> np.ndarray:
    cells = []
    for im in images:
        s = cell / max(im.shape[:2])
        r = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        canvas = np.zeros((cell, cell, 3), np.uint8)
        canvas[:r.shape[0], :r.shape[1]] = r
        cells.append(canvas)
    while len(cells) % cols:
        cells.append(np.zeros((cell, cell, 3), np.uint8))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def scan_split(root: Path, split: str, names):
    img_dir, lab_dir = root / split / "images", root / split / "labels"
    rec = {"images": 0, "labels_missing": [], "labels_orphan": [], "unreadable": [],
           "empty_label_images": 0, "problems": [], "instances": Counter(),
           "annotation_kinds": Counter(), "sizes": Counter(), "boxes_per_image": Counter(),
           "box_area_fraction": []}
    items = []
    if not img_dir.is_dir():
        rec["split_missing"] = True
        return rec, items
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    stems = {p.stem for p in imgs}
    if lab_dir.is_dir():
        rec["labels_orphan"] = sorted(p.name for p in lab_dir.glob("*.txt")
                                      if p.stem not in stems)
    for p in imgs:
        rec["images"] += 1
        im = cv2.imread(str(p))
        if im is None:
            rec["unreadable"].append(p.name)
            continue
        rec["sizes"][f"{im.shape[1]}x{im.shape[0]}"] += 1
        lab = lab_dir / f"{p.stem}.txt"
        boxes = []
        if lab.exists():
            boxes, problems, kinds = parse_label(lab, len(names))
            rec["problems"] += problems
            rec["annotation_kinds"].update(kinds)
        else:
            rec["labels_missing"].append(p.name)
        if not boxes:
            rec["empty_label_images"] += 1
        rec["boxes_per_image"][len(boxes)] += 1
        for b in boxes:
            rec["instances"][names[b[0]]] += 1
            rec["box_area_fraction"].append(b[3] * b[4])
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        items.append({"path": p, "boxes": boxes, "hash": dhash(gray),
                      "stem": source_stem(p), "mean": float(gray.mean())})
    return rec, items


def leakage(by_split, threshold: int):
    out, pairs = {}, []
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        ia, ib = by_split[a], by_split[b]
        shared = {i["stem"] for i in ia} & {i["stem"] for i in ib}
        n_near = 0
        if ia and ib:
            ha = np.array([i["hash"] for i in ia], dtype=np.uint64)
            for item in ib:
                dist = np.bitwise_count(ha ^ item["hash"])
                j = int(dist.argmin())
                if dist[j] <= threshold:
                    n_near += 1
                    pairs.append((int(dist[j]), ia[j], item))
        out[f"{a}_vs_{b}"] = {"shared_source_stems": len(shared),
                              f"{b}_images_within_hamming_{threshold}": n_near}
    return out, sorted(pairs, key=lambda t: t[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="Roboflow YOLO export (omit if already extracted)")
    ap.add_argument("--extract-to", type=Path, default=Path("data/raw/constellations/extracted"))
    ap.add_argument("--out", type=Path, default=Path("results/constellation/phase1"))
    ap.add_argument("--samples", type=int, default=12, help="images drawn per split")
    ap.add_argument("--hamming", type=int, default=4, help="near-duplicate dHash threshold")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.zip:
        args.extract_to.mkdir(parents=True, exist_ok=True)
        safe_extract(args.zip, args.extract_to)
    root = find_root(args.extract_to)
    names = class_names(root)
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    report = {"root": str(root), "n_classes": len(names), "classes": names, "splits": {}}
    by_split = {}
    for split in SPLITS:
        rec, items = scan_split(root, split, names)
        by_split[split] = items
        areas = rec.pop("box_area_fraction")
        stems = Counter(i["stem"] for i in items)
        rec["unique_source_stems"] = len(stems)
        rec["images_that_are_extra_copies"] = sum(stems.values()) - len(stems)
        if areas:
            rec["box_area_fraction_percentiles"] = {
                str(q): float(np.percentile(areas, q)) for q in (5, 50, 95)}
        if items:
            rec["mean_gray_percentiles"] = {
                str(q): float(np.percentile([i["mean"] for i in items], q)) for q in (5, 50, 95)}
        rec["n_problems"] = len(rec["problems"])
        rec["problems"] = rec["problems"][:50]
        for k in ("instances", "annotation_kinds", "sizes", "boxes_per_image"):
            rec[k] = {str(a): b for a, b in sorted(rec[k].items(), key=lambda t: str(t[0]))}
        report["splits"][split] = rec

    report["total_images"] = sum(r["images"] for r in report["splits"].values())
    totals = defaultdict(int)
    for r in report["splits"].values():
        for c, n in r["instances"].items():
            totals[c] += n
    report["instances_total"] = {c: totals[c] for c in names}
    report["classes_without_instances"] = [c for c in names if not totals[c]]
    report["leakage"], pairs = leakage(by_split, args.hamming)
    for readme in sorted(root.glob("README*")):
        report.setdefault("readme_excerpts", {})[readme.name] = readme.read_text(errors="replace")[:3000]

    rng = random.Random(args.seed)
    for split, items in by_split.items():
        if items:
            chosen = rng.sample(items, min(args.samples, len(items)))
            grid = tile([draw(cv2.imread(str(i["path"])), i["boxes"], names) for i in chosen], cols=4)
            cv2.imwrite(str(fig_dir / f"label_check_{split}.jpg"), grid)
    if pairs:
        shown = []
        for _, a, b in pairs[:6]:
            shown += [cv2.imread(str(a["path"])), cv2.imread(str(b["path"]))]
        cv2.imwrite(str(fig_dir / "duplicate_pairs.jpg"), tile(shown, cols=2))

    (args.out / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("total_images", "n_classes", "classes",
                                             "instances_total", "leakage")}, indent=2))
    for split, r in report["splits"].items():
        print(f"{split}: images={r['images']} unique_sources={r['unique_source_stems']} "
              f"missing_labels={len(r['labels_missing'])} empty={r['empty_label_images']} "
              f"problems={r['n_problems']} kinds={r['annotation_kinds']}")
    print(f"Saved report and figures to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
