"""
Détection des champs de métadonnées sur les crops de zone step02.

Deux modèles YOLO distincts, sélectionnés automatiquement selon le dossier
parent du crop :
  - DET_PARTS_DIR/collecte/      → modèle collecte  (collecteur, date_collecte, localite)
  - DET_PARTS_DIR/determination/ → modèle détermination (determination, date_determination,
                                    determinateur, statut_nomenclatural)

Les crops de sortie des deux modèles alimentent ensuite le même pipeline HTR (step04).

Usage :
  python -m source.step03_fields_detection.predict [--conf 0.25] [--save-crops]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image as PILImage
from ultralytics import YOLO

from source.paths import (
    DET_PARTS_DIR,
    FIELDS_BEST_PT,
    FIELDS_PREDICT_CSV,
    FIELDS_PARTS_DIR,
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ZONE_TYPES = ("collecte", "determination")


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS)


def run_predict(
    images: list[Path],
    models: dict[str, Path],
    conf: float = 0.25,
    save_crops: bool = True,
) -> list[dict]:
    """
    Lance l'inférence YOLO sur les crops de zone step02.

    models : {"collecte": Path, "determination": Path}
    La zone est détectée depuis img.parent.name → choisit le bon modèle.
    Les résultats de tous les crops sont consolidés dans un seul CSV.
    """
    # Chargement des modèles à la demande (évite de charger les deux si une zone est absente)
    _loaded: dict[str, YOLO] = {}

    def _model(zone: str) -> YOLO | None:
        if zone not in models:
            return None
        if zone not in _loaded:
            if not models[zone].exists():
                print(f"  [AVERT] Modèle {zone} introuvable : {models[zone]}")
                return None
            _loaded[zone] = YOLO(str(models[zone]))
        return _loaded[zone]

    all_rows: list[dict] = []
    total = len(images)

    for i, img_path in enumerate(images, 1):
        zone  = img_path.parent.name
        model = _model(zone)
        if model is None:
            if i % 50 == 0 or i == total:
                print(f"  {i}/{total}")
            continue

        results = model(str(img_path), conf=conf, verbose=False)
        boxes   = results[0].boxes

        if boxes is None or len(boxes) == 0:
            all_rows.append({
                "filename": img_path.name,
                "zone":     zone,
                "class":    "",
                "confidence": 0.0,
                "x_center": "", "y_center": "", "width": "", "height": "",
            })
            if i % 50 == 0 or i == total:
                print(f"  {i}/{total}")
            continue

        xywhn   = boxes.xywhn.cpu().numpy()
        confs   = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().int().numpy()

        if save_crops:
            img_pil = PILImage.open(img_path)
            img_w, img_h = img_pil.size
            xyxy = boxes.xyxy.cpu().numpy()

        class_counts: dict[str, int] = {}

        for j, (box_n, conf_val, cls_id) in enumerate(zip(xywhn, confs, cls_ids)):
            cls_name = model.names[int(cls_id)]
            all_rows.append({
                "filename":   img_path.name,
                "zone":       zone,
                "class":      cls_name,
                "confidence": round(float(conf_val), 4),
                "x_center":   round(float(box_n[0]), 6),
                "y_center":   round(float(box_n[1]), 6),
                "width":      round(float(box_n[2]), 6),
                "height":     round(float(box_n[3]), 6),
            })

            if save_crops:
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                parts_dir = FIELDS_PARTS_DIR[zone]
                crop_dir  = parts_dir / cls_name
                crop_dir.mkdir(parents=True, exist_ok=True)
                x1, y1, x2, y2 = [int(v) for v in xyxy[j]]
                x1, y1 = max(0, x1 - 4), max(0, y1 - 4)
                x2, y2 = min(img_w, x2 + 4), min(img_h, y2 + 4)
                crop = img_pil.crop((x1, y1, x2, y2))
                idx  = class_counts[cls_name]
                crop.save(crop_dir / f"{img_path.stem}_{cls_name}_{idx:02d}.jpg", "JPEG", quality=95)

        if i % 50 == 0 or i == total:
            print(f"  {i}/{total}")

    # CSV consolidé (toutes zones)
    consolidated_csv = FIELDS_PREDICT_CSV.get("collecte", list(FIELDS_PREDICT_CSV.values())[0]).parent.parent / "predictions.csv"
    consolidated_csv.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with open(consolidated_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--save-crops", action="store_true")
    args = parser.parse_args()

    images = []
    for zone in ZONE_TYPES:
        zone_dir = DET_PARTS_DIR / zone
        if zone_dir.exists():
            images.extend(collect_images(zone_dir))

    if not images:
        sys.exit(f"Aucune image trouvée dans {DET_PARTS_DIR}")

    models = {z: FIELDS_BEST_PT[z] for z in ZONE_TYPES}
    for z, pt in models.items():
        print(f"Modèle {z:15s} : {pt}")
    print(f"Images totales : {len(images)}")

    rows = run_predict(images, models, conf=args.conf, save_crops=args.save_crops)

    from collections import Counter
    counts = Counter(r["class"] for r in rows if r.get("class"))
    print(f"\n{sum(counts.values())} champs détectés :")
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n}")


if __name__ == "__main__":
    main()
