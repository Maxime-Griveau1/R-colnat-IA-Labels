"""
Prépare le dataset pour l'entraînement YOLO classification.

Structure attendue en entrée :
  APP/Datas/in/00_source_images/entomology/  -> classe "entomology"
  APP/Datas/in/00_source_images/herbarium/   -> classe "herbarium"

Structure générée en sortie (format YOLOv8/v11 classify) :
  APP/Datas/out/step01_classification/dataset/
    train/
      entomology/  herbarium/
    val/
      entomology/  herbarium/
    test/
      entomology/  herbarium/
"""

import os
import shutil
import random
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO = 0.15

BASE_DIR    = Path(__file__).resolve().parents[2]
INPUT_DIR   = BASE_DIR / "Datas" / "in" / "00_source_images"
OUTPUT_DIR  = BASE_DIR / "Datas" / "out" / "step01_classification" / "dataset"
EXTENSIONS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect_images(class_dir: Path) -> list[Path]:
    images = [
        p for p in class_dir.rglob("*")
        if p.suffix.lower() in EXTENSIONS
    ]
    return sorted(images)


def split_and_copy(images: list[Path], class_name: str) -> dict:
    random.seed(SEED)
    shuffled = images.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    counts = {}
    for split_name, split_images in splits.items():
        dest_dir = OUTPUT_DIR / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img_path in split_images:
            shutil.copy2(img_path, dest_dir / img_path.name)
        counts[split_name] = len(split_images)

    return counts


def main(max_images: int | None = None):
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    classes = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    if not classes:
        raise FileNotFoundError(f"Aucune classe trouvée dans {INPUT_DIR}")

    print(f"Classes détectées : {[c.name for c in classes]}")
    if max_images:
        print(f"Limite : {max_images} images par classe")
    print(f"Destination : {OUTPUT_DIR}\n")

    for class_dir in sorted(classes):
        images = collect_images(class_dir)
        if not images:
            print(f"  [{class_dir.name}] Aucune image trouvée, ignoré.")
            continue
        if max_images:
            images = images[:max_images]
        counts = split_and_copy(images, class_dir.name)
        print(
            f"  [{class_dir.name}] {len(images)} images -> "
            f"train={counts['train']}, val={counts['val']}, test={counts['test']}"
        )

    print("\nDataset prêt.")


if __name__ == "__main__":
    main()
