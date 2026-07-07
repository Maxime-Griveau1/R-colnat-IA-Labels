"""
Mode démo : pipeline complet sur une image unique.

  GET  /demo                          → page démo
  POST /demo/upload                   → upload image, retourne session_id
  POST /demo/run/<session_id>         → lance le pipeline, retourne job_id SSE
  GET  /demo/image/<session_id>/<fn>  → sert une image de session
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from flask import jsonify, render_template, request, send_file

from ..app import app
from ..routes.generales import _start_job, BASE_DIR
from source.paths import DET_BEST_PT, FIELDS_BEST_PT

DEMO_DIR    = BASE_DIR / "Datas" / "out" / "demo"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Couleurs des zones (cohérentes avec label_studio_manager.py)
ZONE_COLORS: dict[str, tuple[int, int, int]] = {
    "collecte":          (33,  150, 243),
    "determination":     (233, 30,  99),
    "tampon":            (156, 39,  176),
    "numero_inventaire": (255, 152, 0),
    "graines":           (139, 195, 74),
    "notes":             (76,  175, 80),
    "dessin":            (121, 85,  72),
    "specimen":          (244, 67,  54),
    "logo":              (96,  125, 143),
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/demo")
def demo():
    return render_template("pages/demo.html")


@app.route("/demo/upload", methods=["POST"])
def demo_upload():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "Aucun fichier reçu"}), 400
    if Path(f.filename).suffix.lower() not in ALLOWED_EXT:
        return jsonify({"error": "Format non supporté (JPG, PNG, TIF)"}), 400

    session_id  = str(uuid.uuid4())[:8]
    session_dir = DEMO_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    ext       = Path(f.filename).suffix.lower()
    orig_path = session_dir / f"original{ext}"
    f.save(str(orig_path))

    return jsonify({"session_id": session_id, "original": orig_path.name})


@app.route("/demo/run/<session_id>", methods=["POST"])
def demo_run(session_id: str):
    session_dir = DEMO_DIR / session_id
    if not session_dir.exists():
        return jsonify({"error": "session inconnue"}), 404

    orig_files = [p for p in session_dir.iterdir()
                  if p.stem == "original" and p.suffix.lower() in ALLOWED_EXT]
    if not orig_files:
        return jsonify({"error": "image originale introuvable"}), 404

    orig_path = orig_files[0]

    def pipeline():
        _run_demo_pipeline(orig_path, session_dir)

    job_id = _start_job(pipeline)
    return jsonify({"job_id": job_id})


@app.route("/demo/image/<session_id>/<path:filename>")
def demo_image(session_id: str, filename: str):
    path = DEMO_DIR / session_id / filename
    if not path.exists():
        return "image introuvable", 404
    return send_file(str(path))


# ── Pipeline démo ─────────────────────────────────────────────────────────────

def _emit(msg: str):
    print(msg, flush=True)


def _emit_event(payload: dict):
    """Émet un événement structuré — préfixe JSON: détecté par le JS côté client."""
    print(f"JSON:{json.dumps(payload, ensure_ascii=False)}", flush=True)


def _annotate_zones(img_path: Path, detections: list[dict], out_path: Path):
    """Dessine les bounding boxes de zones sur l'image prétraitée."""
    from PIL import Image, ImageDraw

    img  = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    for det in detections:
        cls = det["class"]
        r, g, b = ZONE_COLORS.get(cls, (153, 153, 153))
        cx, cy = det["x_center"] * w, det["y_center"] * h
        bw, bh = det["width"]    * w, det["height"]   * h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=3)
        draw.rectangle([x1, y1, x2, min(y1 + 22, y2)], fill=(r, g, b, 200))
        conf_pct = int(det.get("confidence", 0) * 100)
        draw.text((x1 + 4, y1 + 3), f"{cls} {conf_pct}%", fill=(255, 255, 255))

    img.save(str(out_path), "JPEG", quality=92)


def _annotate_fields(img_path: Path, detections: list[dict], out_path: Path):
    """Dessine les bounding boxes de champs sur un crop de zone."""
    from PIL import Image, ImageDraw

    img  = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    for det in detections:
        cls = det["field_class"]
        cx, cy = det["x_center"] * w, det["y_center"] * h
        bw, bh = det["width"]    * w, det["height"]   * h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 200, 0, 255), width=2)
        draw.rectangle([x1, y1, x2, min(y1 + 18, y2)], fill=(255, 200, 0, 160))
        draw.text((x1 + 3, y1 + 1), cls, fill=(0, 0, 0))

    img.save(str(out_path), "JPEG", quality=92)


def _run_demo_pipeline(orig_path: Path, out_dir: Path):
    from source.preprocessImages import preprocess_single

    # ── Étape 0 : Prétraitement ───────────────────────────────────────────────
    _emit("=== Étape 0 : Prétraitement ===")
    pre_path = out_dir / "step0_preprocessed.jpg"
    preprocess_single(orig_path, pre_path, size=1024, clahe=True)
    _emit(f"Image redimensionnée 1024×1024 — CLAHE appliqué")
    _emit_event({"step": 0, "done": True,
                 "original": orig_path.name, "preprocessed": pre_path.name})

    # ── Étape 2 : Détection des zones ─────────────────────────────────────────
    _emit("=== Étape 2 : Détection des zones ===")
    if not DET_BEST_PT.exists():
        _emit("[AVERT] Modèle step02 absent — étape ignorée")
        _emit_event({"step": 2, "done": True, "skipped": True})
    else:
        from ultralytics import YOLO
        from source.step02_zone_annotation.predict import _letterbox_to_hd
        from PIL import Image

        model2   = YOLO(str(DET_BEST_PT))
        results2 = model2(str(pre_path), conf=0.25, verbose=False)
        boxes2   = results2[0].boxes

        detections_z2: list[dict] = []
        zone_crops_rel: dict[str, list[str]] = {}

        if boxes2 is not None and len(boxes2) > 0:
            xywhn2   = boxes2.xywhn.cpu().numpy()
            confs2   = boxes2.conf.cpu().numpy()
            cls_ids2 = boxes2.cls.cpu().int().numpy()

            orig_img = Image.open(orig_path)
            orig_w, orig_h = orig_img.size
            class_counts2: dict[str, int] = {}

            for box_n, conf_val, cls_id in zip(xywhn2, confs2, cls_ids2):
                cls_name = model2.names[int(cls_id)]
                det = {
                    "class":      cls_name,
                    "confidence": round(float(conf_val), 3),
                    "x_center":   round(float(box_n[0]), 4),
                    "y_center":   round(float(box_n[1]), 4),
                    "width":      round(float(box_n[2]), 4),
                    "height":     round(float(box_n[3]), 4),
                }
                detections_z2.append(det)

                class_counts2[cls_name] = class_counts2.get(cls_name, 0) + 1
                idx = class_counts2[cls_name]
                x1, y1, x2, y2 = _letterbox_to_hd(box_n, orig_w, orig_h)
                crop      = orig_img.crop((x1, y1, x2, y2))
                crop_dir  = out_dir / "step2_crops" / cls_name
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_name = f"zone_{cls_name}_{idx:02d}.jpg"
                crop.save(str(crop_dir / crop_name), "JPEG", quality=95)
                zone_crops_rel.setdefault(cls_name, []).append(
                    f"step2_crops/{cls_name}/{crop_name}"
                )

        _emit(f"{len(detections_z2)} zones détectées")
        ann2_path = out_dir / "step2_annotated.jpg"
        _annotate_zones(pre_path, detections_z2, ann2_path)
        _emit_event({"step": 2, "done": True,
                     "annotated": ann2_path.name,
                     "detections": detections_z2,
                     "zone_crops": zone_crops_rel})

    # ── Étape 3 : Détection des champs ────────────────────────────────────────
    _emit("=== Étape 3 : Détection des champs de métadonnées ===")
    field_crops_all: list[dict] = []
    field_annotated: dict[str, str] = {}   # zone_crop_stem → annotated filename

    for zone in ("collecte", "determination"):
        model3_path = FIELDS_BEST_PT.get(zone)
        if not model3_path or not model3_path.exists():
            _emit(f"[AVERT] Modèle step03/{zone} absent — ignoré")
            continue
        zone_dir = out_dir / "step2_crops" / zone
        if not zone_dir.exists():
            continue
        zone_imgs = sorted(p for p in zone_dir.iterdir()
                           if p.suffix.lower() in ALLOWED_EXT)
        if not zone_imgs:
            continue

        from ultralytics import YOLO
        from PIL import Image
        model3 = YOLO(str(model3_path))

        for zone_img in zone_imgs:
            r3    = model3(str(zone_img), conf=0.25, verbose=False)
            b3    = r3[0].boxes
            if b3 is None or len(b3) == 0:
                continue

            xywhn3   = b3.xywhn.cpu().numpy()
            confs3   = b3.conf.cpu().numpy()
            cls_ids3 = b3.cls.cpu().int().numpy()

            zi = Image.open(zone_img)
            zw, zh = zi.size
            cc3: dict[str, int] = {}
            dets_for_ann: list[dict] = []

            for box3, cv3, ci3 in zip(xywhn3, confs3, cls_ids3):
                fn3 = model3.names[int(ci3)]
                cc3[fn3] = cc3.get(fn3, 0) + 1
                idx3 = cc3[fn3]

                cx3, cy3 = float(box3[0]) * zw, float(box3[1]) * zh
                bw3, bh3 = float(box3[2]) * zw, float(box3[3]) * zh
                x1f = max(0, int(cx3 - bw3/2) - 2)
                y1f = max(0, int(cy3 - bh3/2) - 2)
                x2f = min(zw, int(cx3 + bw3/2) + 2)
                y2f = min(zh, int(cy3 + bh3/2) + 2)

                field_crop = zi.crop((x1f, y1f, x2f, y2f))
                fd = out_dir / "step3_fields" / fn3
                fd.mkdir(parents=True, exist_ok=True)
                fname = f"{zone_img.stem}_{fn3}_{idx3:02d}.jpg"
                field_crop.save(str(fd / fname), "JPEG", quality=95)

                det3 = {
                    "zone": zone, "field_class": fn3, "filename": fname,
                    "confidence": round(float(cv3), 3),
                    "x_center": round(float(box3[0]), 4),
                    "y_center": round(float(box3[1]), 4),
                    "width":    round(float(box3[2]), 4),
                    "height":   round(float(box3[3]), 4),
                    "path_rel": f"step3_fields/{fn3}/{fname}",
                }
                field_crops_all.append(det3)
                dets_for_ann.append(det3)

            # Image annotée du crop de zone avec les champs en surbrillance
            ann3_name = f"step3_ann_{zone_img.stem}.jpg"
            _annotate_fields(zone_img, dets_for_ann, out_dir / ann3_name)
            field_annotated[zone_img.stem] = ann3_name

    _emit(f"{len(field_crops_all)} champs détectés")
    _emit_event({"step": 3, "done": True,
                 "fields": field_crops_all,
                 "annotated": field_annotated})

    # ── Étape 4 : HTR ─────────────────────────────────────────────────────────
    _emit("=== Étape 4 : Transcription HTR ===")
    _emit("Chargement du modèle GLM-OCR…")
    transcriptions: list[dict] = []

    fields_dir = out_dir / "step3_fields"
    if fields_dir.exists() and field_crops_all:
        from source.step04_extract_text.extract_text import run_htr
        raw_csv     = out_dir / "step4_raw.csv"
        grouped_csv = out_dir / "step4_grouped.csv"
        run_htr(parts_dir=fields_dir, raw_csv=raw_csv, grouped_csv=grouped_csv)

        if raw_csv.exists():
            with open(raw_csv, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("text"):
                        transcriptions.append({
                            "field_class": row["field_class"],
                            "filename":    row["filename"],
                            "text":        row["text"],
                        })
        _emit(f"{len(transcriptions)} champs transcrits")
    else:
        grouped_csv = None
        _emit("[AVERT] Aucun crop de champ — HTR ignoré")

    _emit_event({"step": 4, "done": True, "transcriptions": transcriptions})

    # ── Étape 5 : Correction / normalisation ──────────────────────────────────
    _emit("=== Étape 5 : Normalisation et enrichissement ===")
    result_data: dict = {}

    if grouped_csv and Path(grouped_csv).exists():
        from source.step05_correct.correct import run_correction
        corrected_csv = out_dir / "step5_corrected.csv"
        run_correction(input_csv=Path(grouped_csv), output_csv=corrected_csv)
        if corrected_csv.exists():
            with open(corrected_csv, newline="", encoding="utf-8") as fh:
                rows_c = list(csv.DictReader(fh))
                if rows_c:
                    result_data = {k: v for k, v in rows_c[0].items() if v}
    else:
        _emit("[AVERT] Pas de CSV groupé — correction ignorée")

    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(result_data, fh, ensure_ascii=False, indent=2)

    _emit(f"{len(result_data)} champs Darwin Core produits")
    _emit_event({"step": 5, "done": True, "result": result_data})
    _emit("=== Pipeline terminé ===")
