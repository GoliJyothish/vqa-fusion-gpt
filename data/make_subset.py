"""
Extracts a small subset of CLEVR into data/clevr_subset/, matching the
folder structure clevr_dataset.py expects.

Run this from inside the vqa-fusion-gpt/ repo root:
    python data/make_subset.py

Adjust N_TRAIN / N_VAL below if you want a bigger or smaller subset.
Bigger subset = better training signal but slower to train/upload;
a few thousand is plenty for a portfolio/course project.
"""

import json
import shutil
from pathlib import Path

# ---- CONFIG: adjust these paths/sizes if needed ----
SOURCE_DIR = Path("data/clevr_download")   # where you extracted the 18GB zip
DEST_DIR = Path("data/clevr_subset")       # where the trimmed subset goes
N_TRAIN = 20000
N_VAL = 2000
# ------------------------------------------------------


def make_split_subset(split: str, n: int):
    src_questions_path = SOURCE_DIR / "questions" / f"CLEVR_{split}_questions.json"
    src_images_dir = SOURCE_DIR / "images" / split

    dest_images_dir = DEST_DIR / "images" / split
    dest_questions_dir = DEST_DIR / "questions"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_questions_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{split}] loading {src_questions_path} ...")
    with open(src_questions_path) as f:
        data = json.load(f)

    all_questions = data["questions"]
    subset_questions = all_questions[:n]

    copied_images = set()
    for q in subset_questions:
        img_name = q["image_filename"]
        if img_name not in copied_images:
            src_img = src_images_dir / img_name
            dest_img = dest_images_dir / img_name
            if src_img.exists():
                shutil.copy(src_img, dest_img)
                copied_images.add(img_name)
            else:
                print(f"  WARNING: missing image {src_img}")

    out_path = dest_questions_dir / f"{split}_questions.json"
    with open(out_path, "w") as f:
        json.dump({"questions": subset_questions}, f)

    print(f"[{split}] wrote {len(subset_questions)} questions, "
          f"{len(copied_images)} images -> {DEST_DIR}")


if __name__ == "__main__":
    make_split_subset("train", N_TRAIN)
    make_split_subset("val", N_VAL)
    print("\nDone. You can now delete data/clevr_download/ to free up space:")
    print(f"  rm -rf {SOURCE_DIR}")
