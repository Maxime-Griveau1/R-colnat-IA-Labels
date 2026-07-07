"""
Convertit les annotations Label Studio (champs de métadonnées) en dataset YOLO.

Deux exports distincts par type de zone :
  - collecte      → step03_collecte/dataset/
  - determination → step03_determination/dataset/

Les classes sont lues dynamiquement depuis les annotations (pas codées en dur).
"""

from __future__ import annotations
import random
import re
import shutil
from pathlib import Path

from source.paths import DET_PARTS_DIR, FIELDS_DATASET_DIR, BASE_DIR

SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

FLASK_BASE_URL = "http://localhost:5000/serve-field-image"


def _image_path_from_task(task: dict, zone_type: str) -> Path | None:
    """
    Résout le chemin local depuis l'URL Label Studio.
    Pattern : /serve-field-image/<zone>/<filename>
    """
    url: str = task.get("data", {}).get("image", "")
    m = re.search(r"/serve-field-image/([^/]+)/(.+)$", url)
    if not m:
        return None
    zone, filename = m.group(1), m.group(2)
    if zone != zone_type:
        return None
    p = DET_PARTS_DIR / zone / filename
    return p if p.exists() else None


def _extract_class_names(tasks: list[dict]) -> list[str]:
    seen: list[str] = []
    for task in tasks:
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                for label in result["value"].get("rectanglelabels", []):
                    if label not in seen:
                        seen.append(label)
    return sorted(seen)


def _parse_annotation(task: dict, class_names: list[str]) -> list[dict] | None:
    annotations = task.get("annotations", [])
    if not annotations:
        return None
    ann = next(
        (a for a in annotations if not a.get("was_cancelled") and not a.get("skipped")),
        None,
    )
    if not ann:
        return None

    boxes = []
    for result in ann.get("result", []):
        if result.get("type") != "rectanglelabels":
            continue
        v = result["value"]
        labels = v.get("rectanglelabels", [])
        if not labels:
            continue
        label = labels[0]
        if label not in class_names:
            continue
        class_id = class_names.index(label)

        x_pct = v["x"] / 100
        y_pct = v["y"] / 100
        w_pct = v["width"] / 100
        h_pct = v["height"] / 100

        boxes.append({
            "class_id": class_id,
            "x_center": round(x_pct + w_pct / 2, 6),
            "y_center": round(y_pct + h_pct / 2, 6),
            "width":    round(w_pct, 6),
            "height":   round(h_pct, 6),
        })
    return boxes if boxes else None


def convert(tasks: list[dict], zone_type: str, flask_base_url: str = FLASK_BASE_URL) -> dict:
    """
    Convertit les tâches LS annotées en dataset YOLO pour un type de zone.
    zone_type : 'collecte' ou 'determination'
    """
    dataset_dir = FIELDS_DATASET_DIR[zone_type]
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    class_names = _extract_class_names(tasks)
    print(f"  [{zone_type}] Classes depuis les annotations : {class_names}")

    valid      = []
    n_no_annot = 0
    n_no_image = 0

    for task in tasks:
        boxes = _parse_annotation(task, class_names)
        if boxes is None:
            n_no_annot += 1
            continue
        img_path = _image_path_from_task(task, zone_type)
        if img_path is None:
            n_no_image += 1
            img_url = task.get("data", {}).get("image", "?")
            print(f"  [AVERT] Image introuvable : {img_url.split('/')[-1]}")
            continue
        valid.append({"img_path": img_path, "boxes": boxes})

    print(f"  [{zone_type}] {len(tasks)} tâches | {len(valid)} OK"
          f" | {n_no_annot} sans annotation | {n_no_image} image manquante")

    if not valid:
        return {"error": f"[{zone_type}] Aucune annotation valide.", "count": 0}

    random.seed(SEED)
    random.shuffle(valid)
    n       = len(valid)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    splits  = {
        "train": valid[:n_train],
        "val":   valid[n_train: n_train + n_val],
        "test":  valid[n_train + n_val:],
    }

    for split_name, items in splits.items():
        img_dir = dataset_dir / split_name / "images"
        lbl_dir = dataset_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            src = item["img_path"]
            shutil.copy2(src, img_dir / src.name)
            lbl_file = lbl_dir / (src.stem + ".txt")
            with open(lbl_file, "w") as f:
                for b in item["boxes"]:
                    f.write(
                        f"{b['class_id']} {b['x_center']} {b['y_center']} "
                        f"{b['width']} {b['height']}\n"
                    )

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: train/images\n")
        f.write("val:   val/images\n")
        f.write("test:  test/images\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"  [{zone_type}] Dataset → {dataset_dir}")
    for split_name, items in splits.items():
        print(f"    {split_name:5s} : {len(items)} images")

    return {
        "zone":   zone_type,
        "count":  n,
        "splits": {k: len(v) for k, v in splits.items()},
        "output": str(dataset_dir),
    }
