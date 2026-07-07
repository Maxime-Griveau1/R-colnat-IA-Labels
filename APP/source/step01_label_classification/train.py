"""
Entraîne un modèle YOLOv11n-cls pour la classification d'étiquettes.

Prérequis : avoir lancé prepare_dataset.py au préalable.

Usage :
  python train.py [--epochs 50] [--imgsz 224] [--model yolo11n-cls.pt]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

BASE_DIR    = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Datas" / "out" / "step01_classification" / "dataset"
MODELS_DIR  = BASE_DIR / "Datas" / "out" / "step01_classification" / "models"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=75)
    parser.add_argument("--imgsz",    type=int, default=384)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--model",    type=str, default="yolo11s-cls.pt",
                        help="Modèle de base (yolo11n-cls.pt, yolo11s-cls.pt, ...)")
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--device", type=str, default="",
                        help="'cpu', '0', '0,1' ... (vide = auto)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_DIR}\n"
            "Lancez d'abord prepare_dataset.py"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    results = model.train(
        data=str(DATASET_DIR),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device if args.device else None,
        project=str(MODELS_DIR),
        name="run",
        exist_ok=True,
        # Augmentations légères adaptées aux images de spécimens
        degrees=5,
        translate=0.05,
        scale=0.1,
        fliplr=0.0,   # pas de flip horizontal (orientation des étiquettes compte)
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
    )

    best_model = MODELS_DIR / "run" / "weights" / "best.pt"
    print(f"\nModèle sauvegardé : {best_model}")
    print(f"Top-1 accuracy (val) : {results.results_dict.get('metrics/accuracy_top1', 'n/a'):.4f}")


if __name__ == "__main__":
    main()
