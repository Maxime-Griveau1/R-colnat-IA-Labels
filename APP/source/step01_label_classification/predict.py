"""
Inférence du type d'étiquette sur un dossier d'images.

Usage :
  python predict.py --input <dossier_ou_image> [--model <chemin_best.pt>] [--output <csv>]

Sortie CSV :
  filename, predicted_class, confidence, top2_class, top2_conf
"""

import argparse
import csv
import sys
from pathlib import Path

from ultralytics import YOLO

BASE_DIR   = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = BASE_DIR / "Datas" / "out" / "step01_classification" / "models" / "run" / "weights" / "best.pt"
DEFAULT_OUT   = BASE_DIR / "Datas" / "out" / "step01_classification" / "predictions.csv"
EXTENSIONS    = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Image ou dossier d'images")
    parser.add_argument("--model",  default=str(DEFAULT_MODEL), help="Chemin vers best.pt")
    parser.add_argument("--output", default=str(DEFAULT_OUT),   help="Fichier CSV de sortie")
    parser.add_argument("--conf",   type=float, default=0.0,    help="Seuil de confiance minimum")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Modèle introuvable : {model_path}\nLancez d'abord train.py")

    images = collect_images(Path(args.input))
    if not images:
        sys.exit(f"Aucune image trouvée dans : {args.input}")

    print(f"Modèle    : {model_path}")
    print(f"Images    : {len(images)}")

    model = YOLO(str(model_path))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in images:
        results = model(str(img_path), verbose=False)
        probs = results[0].probs

        top1_idx  = int(probs.top1)
        top1_conf = float(probs.top1conf)
        top1_name = model.names[top1_idx]

        top5_idxs  = probs.top5
        top2_idx   = int(top5_idxs[1]) if len(top5_idxs) > 1 else top1_idx
        top2_conf  = float(probs.top5conf[1]) if len(top5_idxs) > 1 else 0.0
        top2_name  = model.names[top2_idx]

        if top1_conf < args.conf:
            top1_name = "uncertain"

        rows.append({
            "filename":      img_path.name,
            "filepath":      str(img_path),
            "predicted_class": top1_name,
            "confidence":    round(top1_conf, 4),
            "top2_class":    top2_name,
            "top2_conf":     round(top2_conf, 4),
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRésultats écrits dans : {output_path}")

    # Résumé
    from collections import Counter
    counts = Counter(r["predicted_class"] for r in rows)
    print("\nRépartition des prédictions :")
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n} images")


if __name__ == "__main__":
    main()
