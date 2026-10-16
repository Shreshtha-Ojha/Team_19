"""Discover and verify the LOL dataset layout under data/raw/LOL/.

Does NOT assume any particular internal folder naming. Walks the tree,
reports image counts per directory, and checks filename pairing for any
pair of sibling directories that look like a low/high split (by name
containing 'low' or 'high', case-insensitive) or train/test split.

Run: python src/data/verify_lol.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "LOL"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def count_images(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def walk_report(root: Path):
    print(f"Walking {root}\n")
    found_dirs = []
    for d in sorted(root.rglob("*")):
        if d.is_dir():
            n = count_images(d)
            if n > 0:
                rel = d.relative_to(root)
                print(f"  {rel}/  ->  {n} images")
                found_dirs.append(d)
    return found_dirs


def try_pair(dir_a: Path, dir_b: Path):
    exts_a = {p.stem for p in dir_a.iterdir() if p.suffix.lower() in IMG_EXTS}
    exts_b = {p.stem for p in dir_b.iterdir() if p.suffix.lower() in IMG_EXTS}
    common = exts_a & exts_b
    only_a = exts_a - exts_b
    only_b = exts_b - exts_a
    print(f"\nChecking pairing: {dir_a.name}/ ({len(exts_a)}) vs {dir_b.name}/ ({len(exts_b)})")
    print(f"  common filenames: {len(common)}")
    if only_a:
        print(f"  only in {dir_a.name}: {len(only_a)} e.g. {sorted(only_a)[:5]}")
    if only_b:
        print(f"  only in {dir_b.name}: {len(only_b)} e.g. {sorted(only_b)[:5]}")
    if not only_a and not only_b and len(common) > 0:
        print("  PAIRED OK: every filename matches 1:1")
        # sample dimension check
        import cv2
        for stem in list(common)[:3]:
            pa = next(p for p in dir_a.iterdir() if p.stem == stem)
            pb = next(p for p in dir_b.iterdir() if p.stem == stem)
            ia = cv2.imread(str(pa))
            ib = cv2.imread(str(pb))
            print(f"    {stem}: {dir_a.name} shape={ia.shape if ia is not None else 'READ FAILED'} "
                  f"{dir_b.name} shape={ib.shape if ib is not None else 'READ FAILED'}")
        return True
    return False


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist.")
        print("Download the LOL dataset per data/README.md and extract it there first.")
        sys.exit(1)

    dirs = walk_report(DATA_DIR)
    if not dirs:
        print("No directories with images found under data/raw/LOL/. "
              "Check that the archive was extracted (not left as a .zip) and that "
              "data/README.md's placement instructions were followed.")
        sys.exit(1)

    # Try to auto-detect low/high sibling pairs among leaf dirs with images.
    print("\n" + "=" * 60)
    print("Attempting to auto-detect low/high pairing among image folders...")
    candidates = [d for d in dirs]
    paired_any = False
    for i, da in enumerate(candidates):
        for db in candidates[i + 1:]:
            if da.parent != db.parent:
                continue
            name_a, name_b = da.name.lower(), db.name.lower()
            looks_like_pair = (
                ("low" in name_a and "high" in name_b) or
                ("high" in name_a and "low" in name_b) or
                ("low" in name_a and "normal" in name_b) or
                ("normal" in name_a and "low" in name_b)
            )
            if looks_like_pair:
                if try_pair(da, db):
                    paired_any = True

    if not paired_any:
        print("\nNo automatically-recognized low/high pair found by name heuristics.")
        print("Look at the directory listing above and tell me the actual paths for")
        print("the low-light and normal-light folders (train and test splits) so I")
        print("can wire them into src/data/lol_dataset.py explicitly.")
    else:
        print("\nAt least one low/high pair verified above.")


if __name__ == "__main__":
    main()
