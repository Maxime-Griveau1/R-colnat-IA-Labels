"""
Détection des zones sur un dossier d'images (modèle YOLO détection).

Stratégie résolution :
  - Inférence YOLO  : image prétraitée 1024px (cohérent avec l'entraînement)
  - Crop de sortie  : image HD source originale (meilleure qualité pour l'HTR)
  Les coordonnées YOLO sont normalisées [0-1] → applicables à toute résolution.

Usage :
  python -m source.step02_zone_annotation.predict
         --input <dossier_preprocessed> --collection herbarium|entomology
         [--conf 0.25] [--save-crops]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image as PILImage
from ultralytics import YOLO

from source.paths import (
    DET_BEST_PT, DET_PREDICT_CSV, DET_PARTS_DIR,
    SOURCE_HD, PREPROCESS_DIRS,
)

DEFAULT_MODEL    = DET_BEST_PT
DEFAULT_OUT      = DET_PREDICT_CSV
PARTS_DIR        = DET_PARTS_DIR
EXTENSIONS       = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CROP_MARGIN      = 4   # pixels de marge sur le crop HD
LETTERBOX_SIZE   = 1024


def _letterbox_to_hd(xywhn: tuple, hd_w: int, hd_h: int) -> tuple[int, int, int, int]:
    """
    Convertit des coordonnées normalisées YOLO (espace letterbox 1024×1024)
    en pixels dans l'image HD originale.

    Le letterbox a scalé l'image pour que max(w,h)==1024, puis centré dans
    un canvas 1024×1024 avec du padding gris. Cette fonction inverse ce transform.
    """
    scale = LETTERBOX_SIZE / max(hd_w, hd_h)
    new_w = int(hd_w * scale)
    new_h = int(hd_h * scale)
    pad_x = (LETTERBOX_SIZE - new_w) // 2
    pad_y = (LETTERBOX_SIZE - new_h) // 2

    cx_lb = xywhn[0] * LETTERBOX_SIZE
    cy_lb = xywhn[1] * LETTERBOX_SIZE
    w_lb  = xywhn[2] * LETTERBOX_SIZE
    h_lb  = xywhn[3] * LETTERBOX_SIZE

    x1 = int((cx_lb - w_lb / 2 - pad_x) / scale) - CROP_MARGIN
    y1 = int((cy_lb - h_lb / 2 - pad_y) / scale) - CROP_MARGIN
    x2 = int((cx_lb + w_lb / 2 - pad_x) / scale) + CROP_MARGIN
    y2 = int((cy_lb + h_lb / 2 - pad_y) / scale) + CROP_MARGIN

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(hd_w, x2), min(hd_h, y2)
    return x1, y1, x2, y2


def _find_hd_source(stem: str, collection: str) -> Path | None:
    """Cherche l'image HD source par nom de fichier (sans extension)."""
    for hd_dir in SOURCE_HD.get(collection, []):
        for ext in EXTENSIONS:
            candidate = hd_dir / (stem + ext)
            if candidate.exists():
                return candidate
    return None


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS)


def run_predict(
    images: list[Path],
    model_path: Path,
    collection: str = "",
    conf: float = 0.25,
    output_csv: Path = DEFAULT_OUT,
    save_crops: bool = True,
) -> list[dict]:
    """
    Lance l'inférence YOLO sur images prétraitées.
    Les crops sont extraits de l'image HD source si disponible.
    """
    model = YOLO(str(model_path))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if save_crops:
        PARTS_DIR.mkdir(parents=True, exist_ok=True)

    rows  = []
    total = len(images)
    n_hd  = 0
    n_lr  = 0

    for i, img_path in enumerate(images, 1):
        # Inférence sur l'image prétraitée (1024px)
        results  = model(str(img_path), conf=conf, verbose=False)
        boxes    = results[0].boxes

        if boxes is None or len(boxes) == 0:
            rows.append({
                "filename":   img_path.name,
                "filepath":   str(img_path),
                "collection": collection,
                "class":      "",
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
            # Préférer l'image HD source pour le crop
            hd_path = _find_hd_source(img_path.stem, collection)
            if hd_path:
                crop_img = PILImage.open(hd_path)
                n_hd += 1
            else:
                crop_img = PILImage.open(img_path)
                n_lr += 1
            crop_w, crop_h = crop_img.size

        class_counts: dict[str, int] = {}

        for j, (box_n, conf_val, cls_id) in enumerate(zip(xywhn, confs, cls_ids)):
            cls_name = model.names[int(cls_id)]
            rows.append({
                "filename":   img_path.name,
                "filepath":   str(img_path),
                "collection": collection,
                "class":      cls_name,
                "confidence": round(float(conf_val), 4),
                "x_center":   round(float(box_n[0]), 6),
                "y_center":   round(float(box_n[1]), 6),
                "width":      round(float(box_n[2]), 6),
                "height":     round(float(box_n[3]), 6),
            })

            if save_crops:
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                crop_dir = PARTS_DIR / cls_name
                crop_dir.mkdir(parents=True, exist_ok=True)

                # Inversion du transform letterbox → coords pixels dans l'image HD
                x1, y1, x2, y2 = _letterbox_to_hd(box_n, crop_w, crop_h)

                idx  = class_counts[cls_name]
                crop = crop_img.crop((x1, y1, x2, y2))
                crop_name = f"{img_path.stem}_{cls_name}_{idx:02d}.jpg"
                crop.save(crop_dir / crop_name, "JPEG", quality=95)

        if i % 50 == 0 or i == total:
            print(f"  {i}/{total}")

    if save_crops and (n_hd + n_lr) > 0:
        print(f"  Crops HD : {n_hd}  |  Crops 1024px (HD non trouvé) : {n_lr}")

    if rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--collection", default="", help="herbarium|entomology")
    parser.add_argument("--model",      default=str(DEFAULT_MODEL))
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--output-csv", default=str(DEFAULT_OUT))
    parser.add_argument("--save-crops", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Modèle introuvable : {model_path}")

    images = collect_images(Path(args.input))
    if not images:
        sys.exit(f"Aucune image trouvée dans : {args.input}")

    print(f"Modèle     : {model_path}")
    print(f"Collection : {args.collection or '(non précisée)'}")
    print(f"Images     : {len(images)}")
    print(f"Seuil conf : {args.conf}")

    rows = run_predict(
        images, model_path,
        collection=args.collection,
        conf=args.conf,
        output_csv=Path(args.output_csv),
        save_crops=args.save_crops,
    )

    from collections import Counter
    counts = Counter(r["class"] for r in rows if r["class"])
    print(f"\n{sum(counts.values())} zones détectées sur {len(images)} images :")
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n}")


if __name__ == "__main__":
    main()
