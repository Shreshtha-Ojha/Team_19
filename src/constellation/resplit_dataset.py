"""Rebuild train/valid/test so no source image or neighbouring frame crosses splits.

The Roboflow export scattered augmented copies of one source image across its
splits. The sources are also ordered sequences (daily sky screenshots, bursts
of one scene), so neighbours look alike even when their names differ. This
script therefore:

  1. groups all copies by source image;
  2. cuts each sequence into contiguous blocks and assigns whole blocks to a
     split, cycling through the year so seasonal constellations reach each split;
  3. drops frames close to a boundary between two different splits;
  4. keeps every copy for train, but one copy per source for valid and test
     (the copy with the fewest pure-black pixels, i.e. the least warped).

The split depends only on file names and --seed, so it can be rebuilt on Colab.

Run from the repository root:
  python -I src/constellation/resplit_dataset.py
"""
import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import yaml

SPLITS = ("train", "valid", "test")
DAILY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-\d{2}-\d{2}-\d{2}-[ns]_png_jpg$")
NUMBERED = re.compile(r"^(.+)_(\d+)_png_jpg$")


def family_position(stem: str) -> tuple[str, int]:
    """Return the sequence a source belongs to and its position in it."""
    m = DAILY.match(stem)
    if m:
        return "daily", date(*map(int, m.groups())).toordinal()
    m = NUMBERED.match(stem)
    if m:
        return m.group(1), int(m.group(2))
    raise ValueError(f"unrecognised source name: {stem}")


def assign_positions(positions, block: int, gap: int, cycle: list[str], rng: random.Random):
    """Map each position to a split, or to None if it falls in a boundary gap."""
    start = min(positions)
    block_ids = sorted({(p - start) // block for p in positions})
    block_split = {}
    for i in range(0, len(block_ids), len(cycle)):
        chunk = block_ids[i:i + len(cycle)]
        if len(chunk) < len(cycle):       # an incomplete cycle stays in train
            block_split.update({b: "train" for b in chunk})
            continue
        order = cycle[:]
        rng.shuffle(order)
        block_split.update(zip(chunk, order))

    def split_at(p):
        return block_split.get((p - start) // block) if p >= start else None

    # Check neighbouring slots, not neighbouring files, so that a missing frame
    # number can't shrink the gap.
    result = {}
    for p in positions:
        s = split_at(p)
        near = (split_at(p + d) for d in range(-gap, gap + 1))
        result[p] = None if any(n is not None and n != s for n in near) else s
    return result


def black_fraction(path: Path) -> float:
    return float((cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) == 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/raw/constellations/extracted"))
    ap.add_argument("--dst", type=Path, default=Path("data/raw/constellations/resplit"))
    ap.add_argument("--out", type=Path, default=Path("results/constellation/phase1"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--daily-block", type=int, default=10, help="days per block")
    ap.add_argument("--daily-gap", type=int, default=2, help="days dropped each side of a boundary")
    ap.add_argument("--burst-block", type=int, default=7, help="frames per block")
    ap.add_argument("--burst-gap", type=int, default=1, help="frames dropped each side of a boundary")
    args = ap.parse_args()

    yaml_paths = sorted(args.src.rglob("data.yaml"), key=lambda p: len(p.parts))
    src = yaml_paths[0].parent
    meta = yaml.safe_load((src / "data.yaml").read_text())

    copies = defaultdict(list)
    for split in SPLITS:
        for img in sorted((src / split / "images").iterdir()):
            copies[re.split(r"\.rf\.", img.name)[0]].append(img)

    families = defaultdict(set)
    where = {}
    for stem in copies:
        fam, pos = family_position(stem)
        families[fam].add(pos)
        where[stem] = (fam, pos)

    rng = random.Random(args.seed)
    cycle = ["train"] * 4 + ["valid", "test"]
    assigned = {}
    for fam in sorted(families):
        block, gap = ((args.daily_block, args.daily_gap) if fam == "daily"
                      else (args.burst_block, args.burst_gap))
        assigned[fam] = assign_positions(sorted(families[fam]), block, gap, cycle, rng)

    if args.dst.exists():
        shutil.rmtree(args.dst)
    for split in SPLITS:
        for sub in ("images", "labels"):
            (args.dst / split / sub).mkdir(parents=True)

    rows = []
    for stem in sorted(copies):
        fam, pos = where[stem]
        split = assigned[fam][pos]
        imgs = sorted(copies[stem], key=lambda p: p.name)
        keep = imgs
        if split in ("valid", "test"):
            keep = [min(imgs, key=lambda p: (black_fraction(p), p.name))]
        for img in imgs:
            status = "dropped_gap" if split is None else (
                "kept" if img in keep else "dropped_extra_copy")
            rows.append({"source": stem, "family": fam, "position": pos,
                         "split": split or "", "file": img.name, "status": status})
            if status == "kept":
                shutil.copy2(img, args.dst / split / "images" / img.name)
                shutil.copy2(img.parent.parent / "labels" / f"{img.stem}.txt",
                             args.dst / split / "labels" / f"{img.stem}.txt")

    (args.dst / "data.yaml").write_text(yaml.safe_dump({
        "path": str(args.dst.resolve()), "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": meta["nc"], "names": meta["names"]}, sort_keys=False))
    for readme in src.glob("README*"):
        shutil.copy2(readme, args.dst / readme.name)

    # Check the result from the files actually written, not from the plan above.
    on_disk = {s: {re.split(r"\.rf\.", p.name)[0] for p in (args.dst / s / "images").iterdir()}
               for s in SPLITS}
    report = {"seed": args.seed, "params": {k: v for k, v in vars(args).items()
                                            if k in ("daily_block", "daily_gap", "burst_block", "burst_gap")},
              "splits": {}, "overlap": {}, "min_sequence_distance": {}}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = on_disk[a] & on_disk[b]
        report["overlap"][f"{a}_vs_{b}"] = len(shared)
        best = defaultdict(lambda: None)
        for sa in on_disk[a]:
            fa, pa = where[sa]
            for sb in on_disk[b]:
                fb, pb = where[sb]
                if fa == fb and (best[fa] is None or abs(pa - pb) < best[fa]):
                    best[fa] = abs(pa - pb)
        report["min_sequence_distance"][f"{a}_vs_{b}"] = dict(best)
    names = meta["names"]
    for s in SPLITS:
        labels = sorted((args.dst / s / "labels").glob("*.txt"))
        inst = Counter(names[int(l.split()[0])] for f in labels for l in f.read_text().splitlines())
        fams = Counter(where[stem][0] for stem in on_disk[s])
        report["splits"][s] = {"images": len(labels), "sources": len(on_disk[s]),
                               "sources_per_family": dict(sorted(fams.items())),
                               "instances": {c: inst[c] for c in names}}
    report["status_counts"] = dict(Counter(r["status"] for r in rows))
    report["sources_dropped_in_gaps"] = len({r["source"] for r in rows if r["status"] == "dropped_gap"})
    report["classes_missing"] = {s: [c for c, n in report["splits"][s]["instances"].items() if not n]
                                 for s in SPLITS}

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "split_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "resplit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if any(report["overlap"].values()):
        print("ERROR: a source image appears in two splits", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
