"""
Convertit les annotations Label Studio (format JSON) en dataset YOLO détection.

Structure de sortie :
  APP/Datas/out/step02_zone_detection/dataset/
    train/images/  train/labels/
    val/images/    val/labels/
    test/images/   test/labels/
    data.yaml

Format YOLO détection (par fichier .txt) :
  <class_id> <x_center> <y_center> <width> <height>
  valeurs normalisées [0, 1] par rapport à la taille de l'image.
"""

from __future__ import annotations
import random
import re
import shutil
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[2]
OUTPUT_DIR  = BASE_DIR / "Datas" / "out" / "step02_zone_detection" / "dataset"
SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15


def _extract_class_names(tasks: list[dict]) -> list[str]:
    """
    Déduit la liste ordonnée des classes depuis les annotations elles-mêmes.
    Ordre : apparition dans les tâches (stable entre exports successifs via tri).
    """
    seen: list[str] = []
    for task in tasks:
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                for label in result["value"].get("rectanglelabels", []):
                    if label not in seen:
                        seen.append(label)
    return sorted(seen)  # tri alphabétique = ordre stable et reproductible


def _parse_annotation(task: dict, class_names: list[str]) -> list[dict] | None:
    """
    Extrait les bounding boxes d'une tâche LS annotée.
    Retourne None si aucune annotation valide.
    """
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
        v      = result["value"]
        labels = v.get("rectanglelabels", [])
        if not labels:
            continue
        label = labels[0]
        if label not in class_names:
            continue
        class_id = class_names.index(label)

        # LS exprime x, y, width, height en % de l'image (0–100)
        x_pct = v["x"] / 100
        y_pct = v["y"] / 100
        w_pct = v["width"] / 100
        h_pct = v["height"] / 100

        x_center = round(x_pct + w_pct / 2, 6)
        y_center = round(y_pct + h_pct / 2, 6)

        boxes.append({
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width":    round(w_pct, 6),
            "height":   round(h_pct, 6),
        })
    return boxes if boxes else None


def _image_path_from_task(task: dict, flask_base_url: str = "") -> Path | None:
    """
    Résout le chemin local depuis l'URL Label Studio.
    Cherche le pattern /serve-image/<collection>/<filename> dans le chemin,
    indépendamment du host (localhost vs 172.x.x.x).
    """
    url: str = task.get("data", {}).get("image", "")
    m = re.search(r"/serve-image/([^/]+)/(.+)$", url)
    if not m:
        return None
    collection = m.group(1)
    filename   = m.group(2).split("/")[-1]

    candidates = [
        BASE_DIR / "Datas" / "out" / "step00_preprocessing" / collection / filename,
        BASE_DIR / "Datas" / "in"  / "00_source_images" / collection / filename,
        BASE_DIR / "Datas" / "in"  / "00_source_images" / collection / "jpeg" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def convert(
    tasks: list[dict],
    flask_base_url: str = "http://localhost:5000/serve-image",
) -> dict:
    """
    Convertit les tâches LS annotées en dataset YOLO.
    Retourne un rapport de conversion.
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    class_names = _extract_class_names(tasks)
    print(f"  Classes détectées depuis les annotations : {class_names}")

    # Filtrer les tâches avec annotations valides
    valid      = []
    n_no_annot = 0
    n_no_image = 0
    for task in tasks:
        boxes = _parse_annotation(task, class_names)
        if boxes is None:
            n_no_annot += 1
            continue
        img_path = _image_path_from_task(task, flask_base_url)
        if img_path is None:
            n_no_image += 1
            img_url = task.get("data", {}).get("image", "?")
            print(f"  [AVERT] Image introuvable localement : {img_url.split('/')[-1]}")
            continue
        valid.append({"img_path": img_path, "boxes": boxes})

    print(f"  Tâches : {len(tasks)}  |  annotées OK : {len(valid)}"
          f"  |  sans annotation : {n_no_annot}  |  image manquante : {n_no_image}")

    if not valid:
        if n_no_image > 0 and len(valid) == 0:
            return {"error": (
                f"{len(tasks) - n_no_annot} annotation(s) trouvée(s) mais aucune image locale résolue. "
                "Lancez d'abord le Prétraitement (Étape 0) pour générer les images dans "
                "Datas/out/step00_preprocessing/."
            ), "count": 0}
        return {"error": "Aucune annotation valide à convertir.", "count": 0}

    # Split aléatoire
    random.seed(SEED)
    random.shuffle(valid)
    n       = len(valid)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    splits  = {
        "train": valid[:n_train],
        "val":   valid[n_train : n_train + n_val],
        "test":  valid[n_train + n_val :],
    }

    for split_name, items in splits.items():
        img_dir = OUTPUT_DIR / split_name / "images"
        lbl_dir = OUTPUT_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            src = item["img_path"]
            dst = img_dir / src.name
            shutil.copy2(src, dst)

            lbl_file = lbl_dir / (src.stem + ".txt")
            with open(lbl_file, "w") as f:
                for b in item["boxes"]:
                    f.write(
                        f"{b['class_id']} {b['x_center']} {b['y_center']} "
                        f"{b['width']} {b['height']}\n"
                    )

    # data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write(f"train: train/images\n")
        f.write(f"val:   val/images\n")
        f.write(f"test:  test/images\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"Dataset YOLO généré dans : {OUTPUT_DIR}")
    for split_name, items in splits.items():
        print(f"  {split_name:5s} : {len(items)} images")
    print(f"Classes : {class_names}")

    return {
        "count":  n,
        "splits": {k: len(v) for k, v in splits.items()},
        "output": str(OUTPUT_DIR),
    }
