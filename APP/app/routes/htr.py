"""
Routes Flask pour l'étape 4 : extraction HTR (TrOCR).

Points d'entrée :
  GET  /step04/status   → état des fichiers de sortie (JSON)
  POST /step04/run      → lance l'HTR en arrière-plan (SSE)
"""

from __future__ import annotations

import base64
import csv
import io
from datetime import datetime
from pathlib import Path

from flask import jsonify, request, send_file

from ..app import app
from ..routes.generales import _start_job, BASE_DIR
from source.paths import (
    FIELDS_PARTS_DIR,
    HTR_OUT_DIR,
    HTR_RAW_CSV,
    HTR_GROUPED_CSV,
)

FIELD_CLASSES = [
    "collecteur",
    "date_collecte",
    "date_determination",
    "determinateur",
    "determination",
    "localite",
    "numero_inventaire",
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _fmt(p: Path) -> str | None:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%d/%m/%Y à %H:%M")
    except OSError:
        return None


def _parts_stats() -> dict:
    stats: dict[str, int] = {}
    for parts_dir in FIELDS_PARTS_DIR.values():
        if not parts_dir.exists():
            continue
        for cls_dir in parts_dir.iterdir():
            if cls_dir.is_dir():
                n = len([p for p in cls_dir.iterdir() if p.suffix.lower() in EXTENSIONS])
                stats[cls_dir.name] = stats.get(cls_dir.name, 0) + n
    return stats


def _all_crop_paths() -> list[Path]:
    crops = []
    for parts_dir in FIELDS_PARTS_DIR.values():
        if not parts_dir.exists():
            continue
        for cls_dir in parts_dir.iterdir():
            if cls_dir.is_dir():
                crops.extend(p for p in cls_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    return crops


def _htr_stats() -> dict:
    if not HTR_RAW_CSV.exists():
        return {}
    with open(HTR_RAW_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    counts: dict[str, int] = {}
    conf_sum: dict[str, float] = {}
    for r in rows:
        cls = r.get("field_class", "")
        if cls:
            counts[cls] = counts.get(cls, 0) + 1
            try:
                conf_sum[cls] = conf_sum.get(cls, 0.0) + float(r.get("confidence", 0))
            except ValueError:
                pass
    n_specimens = len({r["specimen_id"] for r in rows if r.get("specimen_id")})
    return {
        "total_crops":    sum(counts.values()),
        "total_specimens": n_specimens,
        "by_class": [
            {
                "class":    cls,
                "count":    counts[cls],
                "avg_conf": round(conf_sum.get(cls, 0) / counts[cls], 3) if counts[cls] else 0,
            }
            for cls in sorted(counts)
        ],
        "raw_csv_mtime":     _fmt(HTR_RAW_CSV),
        "grouped_csv_mtime": _fmt(HTR_GROUPED_CSV) if HTR_GROUPED_CSV.exists() else None,
    }


@app.route("/step04/status")
def step04_status():
    return jsonify({
        "parts_stats":       _parts_stats(),
        "htr_stats":         _htr_stats(),
        "raw_csv_exists":    HTR_RAW_CSV.exists(),
        "grouped_csv_exists": HTR_GROUPED_CSV.exists(),
    })


@app.route("/step04/preview-preprocess", methods=["POST"])
def step04_preview_preprocess():
    """
    Applique le prétraitement à un crop aléatoire (ou au crop demandé) et retourne
    l'image originale + l'image prétraitée en base64 pour affichage côté client.
    """
    import random
    from source.step04_extract_text.preprocess import preprocess_crop

    field_class  = request.form.get("field_class", "")
    binarize     = request.form.get("binarize", "true") == "true"
    deskew       = request.form.get("deskew", "true") == "true"
    target_h_str = request.form.get("target_height", "128").strip()
    target_h     = int(target_h_str) if target_h_str.isdigit() else 128

    # Sélectionner un crop
    crops = _all_crop_paths()
    if field_class:
        filtered = [p for p in crops if p.parent.name == field_class]
        if filtered:
            crops = filtered
    if not crops:
        return jsonify({"error": "Aucun crop disponible."}), 404

    crop_path = random.choice(crops)

    def _pil_to_b64(img) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    from PIL import Image as PILImage
    original = PILImage.open(crop_path).convert("RGB")
    processed = preprocess_crop(
        crop_path,
        target_height=target_h,
        binarize=binarize,
        deskew=deskew,
    )

    return jsonify({
        "filename":  crop_path.name,
        "field":     crop_path.parent.name,
        "original":  _pil_to_b64(original),
        "processed": _pil_to_b64(processed.convert("RGB")),
        "orig_size": list(original.size),
        "proc_size": list(processed.size),
    })


@app.route("/step04/run", methods=["POST"])
def step04_run():
    model_id      = request.form.get("model_id", "zai-org/GLM-OCR")
    device        = request.form.get("device", "auto")
    max_crops_str = request.form.get("max_crops", "").strip()
    max_crops     = int(max_crops_str) if max_crops_str.isdigit() else None
    binarize      = request.form.get("binarize", "true") == "true"
    deskew        = request.form.get("deskew", "true") == "true"
    target_h_str  = request.form.get("target_height", "128").strip()
    target_h      = int(target_h_str) if target_h_str.isdigit() else 128

    total_crops = sum(_parts_stats().values())
    if total_crops == 0:
        return jsonify({"error": "Aucun crop disponible — lancez d'abord la prédiction step03."}), 400

    def _job():
        from source.step04_extract_text.extract_text import run_htr
        print(f"Modèle HTR : {model_id}  |  Device : {device}")
        print(f"Prétraitement : binarize={binarize}  deskew={deskew}  target_height={target_h}px")
        if max_crops:
            print(f"Limite : {max_crops} crops")
        rows = run_htr(
            parts_dir=FIELDS_PARTS_DIR,
            model_id=model_id,
            device=device,
            max_crops=max_crops,
            raw_csv=HTR_RAW_CSV,
            grouped_csv=HTR_GROUPED_CSV,
            preprocess_kwargs={"binarize": binarize, "deskew": deskew, "target_height": target_h},
        )
        n_specimens = len({r["specimen_id"] for r in rows if r.get("specimen_id")})
        print(f"\n{len(rows)} transcriptions — {n_specimens} spécimens")

    return jsonify({"job_id": _start_job(_job)})
