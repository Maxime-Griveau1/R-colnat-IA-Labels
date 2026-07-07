"""
Routes Flask pour l'étape 2 : Annotation des zones avec Label Studio.

Points d'entrée :
  GET  /step02/status          → état LS + stats projet (JSON)
  POST /step02/save-config     → sauvegarde URL + API key dans .env
  POST /step02/create-project  → crée/retrouve le projet LS (SSE)
  POST /step02/import-images   → importe les images prétraitées (SSE)
  POST /step02/export-yolo     → exporte les annotations → YOLO (SSE)
  POST /step02/train           → entraîne YOLO détection (SSE)
  POST /step02/predict         → applique le modèle entraîné sur le dataset (SSE)
  GET  /step02/predict-status  → état du modèle + stats des prédictions
  GET  /serve-image/<path>     → sert les images locales vers Label Studio
"""

from __future__ import annotations
import csv
import os
import re
from pathlib import Path

import dotenv
from flask import Response, jsonify, request, send_file

from ..app import app
from ..routes.generales import _start_job, BASE_DIR
from source.paths import DET_BEST_PT as DETECTION_MODEL, DET_PREDICT_CSV as DETECTION_CSV, DET_PARTS_DIR as DETECTION_PARTS
from source.step02_zone_annotation.label_studio_manager import (
    CLASS_NAMES,
    LSManager,
    PROJECT_TITLE,
)
from source.step02_zone_annotation.export_to_yolo import convert as export_to_yolo

ENV_PATH = BASE_DIR / ".env"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ls_config() -> tuple[str, str]:
    """Retourne (url_interne, api_key) depuis les variables d'environnement.
    L'URL interne est celle utilisée par Flask pour les appels API (peut être
    un hostname Docker comme http://label-studio:8080).
    """
    return (
        os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
        os.environ.get("LABEL_STUDIO_API_KEY", ""),
    )


def _ls_public_url() -> str:
    """URL publique pour le navigateur — distincte de l'URL interne Docker."""
    return os.environ.get(
        "LABEL_STUDIO_PUBLIC_URL",
        os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
    )


def _ls() -> LSManager:
    url, key = _ls_config()
    return LSManager(url=url, api_key=key)


def _saved_project_id() -> int | None:
    pid = os.environ.get("LABEL_STUDIO_PROJECT_ID", "")
    return int(pid) if pid.isdigit() else None


def _save_env_key(key: str, value: str) -> None:
    """Met à jour (ou ajoute) une clé dans le fichier .env."""
    content = ENV_PATH.read_text() if ENV_PATH.exists() else ""
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    new_line = f"{key}={value}"
    if pattern.search(content):
        content = pattern.sub(new_line, content)
    else:
        content = content.rstrip("\n") + f"\n{new_line}\n"
    ENV_PATH.write_text(content)
    os.environ[key] = value


# ── Images prétraitées disponibles ────────────────────────────────────────────

PREPROCESSED_DIRS = {
    "entomology": BASE_DIR / "Datas" / "out" / "step00_preprocessing" / "entomology",
    "herbarium":  BASE_DIR / "Datas" / "out" / "step00_preprocessing" / "herbarium",
}
# Dossiers sources (fallback si l'image n'a pas encore été prétraitée)
SOURCE_DIRS = {
    "entomology": BASE_DIR / "Datas" / "in" / "00_source_images" / "entomology",
    "herbarium":  BASE_DIR / "Datas" / "in" / "00_source_images" / "herbarium" / "jpeg",
}
# Tous les dossiers consultés par serve-image (prétraités en priorité)
ALL_SERVE_DIRS: dict[str, list[Path]] = {
    col: [PREPROCESSED_DIRS[col], SOURCE_DIRS[col], SOURCE_DIRS[col] / "jpeg"]
    for col in ("entomology", "herbarium")
}
EXTENSIONS      = {".jpg", ".jpeg", ".png"}
PREDICTIONS_CSV = BASE_DIR / "Datas" / "out" / "step01_classification" / "predictions.csv"


def _zone_images_by_collection() -> tuple[dict[str, list[Path]], str]:
    """
    Retourne ({collection: [Path]}, source) pour herbier et entomologie.
    Utilisé par l'import pour construire les URLs avec la bonne collection.
    """
    if PREDICTIONS_CSV.exists():
        by_col: dict[str, list[Path]] = {"herbarium": [], "entomology": []}
        with open(PREDICTIONS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                col = row.get("predicted_class", "")
                if col not in PREPROCESSED_DIRS:
                    continue
                src_path = Path(row["filepath"])
                preprocessed = PREPROCESSED_DIRS[col] / src_path.name
                path = preprocessed if preprocessed.exists() else src_path if src_path.exists() else None
                if path:
                    by_col[col].append(path)
        for col in by_col:
            by_col[col] = sorted(by_col[col])
        return by_col, "classification step01 (herbier + entomologie)"

    by_col = {}
    for col, d in PREPROCESSED_DIRS.items():
        if d.exists():
            by_col[col] = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTENSIONS)
    return by_col, "dossiers prétraités (classification step01 non effectuée)"


def _zone_images() -> tuple[list[Path], str]:
    """Retourne (toutes les images à annoter, source) — herbier + entomologie."""
    by_col, source = _zone_images_by_collection()
    all_images = sorted(p for imgs in by_col.values() for p in imgs)
    return all_images, source


# ── Route : servir les images vers Label Studio ───────────────────────────────

@app.route("/serve-image/<collection>/<path:filename>")
def serve_image(collection: str, filename: str):
    """
    Sert une image depuis les dossiers connus (prétraité en priorité, source en fallback).
    CORS ouvert pour Label Studio.
    """
    dirs = ALL_SERVE_DIRS.get(collection)
    if dirs is None:
        return "collection inconnue", 404
    for base in dirs:
        img_path = base / filename
        if img_path.exists() and img_path.suffix.lower() in EXTENSIONS:
            response = send_file(str(img_path), mimetype="image/jpeg")
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            return response
    return "image introuvable", 404


# ── Route : état Label Studio ─────────────────────────────────────────────────

@app.route("/step02/status")
def step02_status():
    url, api_key = _ls_config()
    manager = LSManager(url=url, api_key=api_key)

    is_up   = manager.ping()
    authed  = manager.check_auth() if is_up and api_key else False
    project_id = _saved_project_id()
    stats   = {}
    if authed and project_id:
        try:
            stats = manager.project_stats(project_id)
        except Exception:
            pass

    all_imgs, img_source = _zone_images()
    by_col, _ = _zone_images_by_collection()
    diagnosis = manager.token_diagnosis()

    # Nombre d'images déjà traitées par le modèle de détection (lignes dans le CSV)
    treated_images = 0
    if DETECTION_CSV.exists():
        with open(DETECTION_CSV, newline="", encoding="utf-8") as _f:
            treated_images = len({r["filename"] for r in csv.DictReader(_f) if r.get("filename")})

    return jsonify({
        "ls_url":                _ls_public_url(),
        "is_up":                 is_up,
        "authed":                authed,
        "api_key_set":           bool(api_key),
        "token_diag":            diagnosis,
        "project_id":            project_id,
        "stats":                 stats,
        "available_images":      len(all_imgs),
        "available_herbarium":   len(by_col.get("herbarium", [])),
        "available_entomology":  len(by_col.get("entomology", [])),
        "image_source":          img_source,
        "treated_images":        treated_images,
    })


# ── Route : sauvegarder la config LS ─────────────────────────────────────────

@app.route("/step02/save-config", methods=["POST"])
def step02_save_config():
    url     = request.form.get("ls_url", "").strip()
    api_key = request.form.get("api_key", "").strip()
    if url:
        _save_env_key("LABEL_STUDIO_URL", url)
    if api_key:
        _save_env_key("LABEL_STUDIO_API_KEY", api_key)
    return jsonify({"ok": True})


# ── Route : créer / retrouver le projet ──────────────────────────────────────

@app.route("/step02/create-project", methods=["POST"])
def step02_create_project():
    def _job():
        url, api_key = _ls_config()
        if not api_key:
            print("[ERREUR] Clé API Label Studio non configurée.")
            return
        manager = LSManager(url=url, api_key=api_key)
        if not manager.ping():
            print(f"[ERREUR] Label Studio n'est pas accessible à {url}")
            return
        print(f"Connexion à Label Studio : {url}")
        pid = manager.get_or_create_project()
        _save_env_key("LABEL_STUDIO_PROJECT_ID", str(pid))
        print(f"Projet '{PROJECT_TITLE}' — id={pid}")
        print(f"Classes annotées : {', '.join(CLASS_NAMES)}")
        print(f"Ouvrez le projet : {url}/projects/{pid}/")

    return jsonify({"job_id": _start_job(_job)})


# ── Route : importer les images ───────────────────────────────────────────────

@app.route("/step02/import-images", methods=["POST"])
def step02_import_images():
    max_images  = request.form.get("max_images", "")
    max_images  = int(max_images) if max_images.isdigit() else None
    clear_first = request.form.get("clear_first") == "true"
    flask_host  = request.host_url.rstrip("/")   # ex: http://localhost:5000

    def _job():
        url, api_key = _ls_config()
        pid = _saved_project_id()
        if not api_key or pid is None:
            print("[ERREUR] Configurez d'abord la clé API et créez le projet.")
            return
        manager = LSManager(url=url, api_key=api_key)

        images_by_col, source = _zone_images_by_collection()
        print(f"Source des images : {source}")
        n_herb = len(images_by_col.get("herbarium", []))
        n_ento = len(images_by_col.get("entomology", []))
        print(f"  herbier : {n_herb}  — entomologie : {n_ento}")

        if clear_first:
            print("Suppression des tâches existantes…")
            manager.delete_all_tasks(pid)

        if not any(images_by_col.values()):
            print("[ERREUR] Aucune image trouvée. Vérifiez que le prétraitement et/ou la classification ont été effectués.")
            return

        # Sélection 50/50 herbier/entomologie si max_images demandé
        if max_images:
            half = max_images // 2
            herb  = images_by_col.get("herbarium",  [])[:half]
            ento  = images_by_col.get("entomology", [])[:max_images - len(herb)]
            # Si une collection a moins de half, compléter avec l'autre
            if len(herb) < half:
                ento = images_by_col.get("entomology", [])[:max_images - len(herb)]
            selected = herb + ento
            print(f"Sélection équilibrée : {len(herb)} herbier + {len(ento)} entomologie = {len(selected)}")
        else:
            selected = (
                images_by_col.get("herbarium",  []) +
                images_by_col.get("entomology", [])
            )

        # Construire les URLs Flask avec la bonne collection dans le chemin
        urls = []
        for col, imgs in [
            ("herbarium",  images_by_col.get("herbarium",  []) if not max_images else herb),
            ("entomology", images_by_col.get("entomology", []) if not max_images else ento),
        ]:
            for img in imgs:
                urls.append(f"{flask_host}/serve-image/{col}/{img.name}")

        if not urls:
            print("[ERREUR] Aucune URL d'image construite.")
            return

        imported = manager.import_images(pid, urls)
        print(f"{imported} images importées dans le projet {pid}.")
        print(f"Ouvrez Label Studio pour annoter : {url}/projects/{pid}/")

    return jsonify({"job_id": _start_job(_job)})


# ── Route : exporter les annotations → YOLO ──────────────────────────────────

@app.route("/step02/export-yolo", methods=["POST"])
def step02_export_yolo():
    flask_base = request.host_url.rstrip("/") + "/serve-image"

    def _job():
        url, api_key = _ls_config()
        pid = _saved_project_id()
        if not api_key or pid is None:
            print("[ERREUR] Configurez d'abord la clé API et créez le projet.")
            return
        manager = LSManager(url=url, api_key=api_key)
        print("Export des annotations depuis Label Studio…")
        tasks = manager.export_tasks(pid)
        print(f"{len(tasks)} tâches récupérées.")
        result = export_to_yolo(tasks, flask_base_url=flask_base)
        if "error" in result:
            print(f"[ERREUR] {result['error']}")
        else:
            print(f"Dataset YOLO créé : {result['count']} images annotées")
            for split, n in result["splits"].items():
                print(f"  {split:5s} → {n} images")

    return jsonify({"job_id": _start_job(_job)})


# ── Route : entraîner YOLO détection ─────────────────────────────────────────

@app.route("/step02/train-detection", methods=["POST"])
def step02_train_detection():
    epochs     = int(request.form.get("epochs", 150))
    imgsz      = int(request.form.get("imgsz", 640))
    patience   = int(request.form.get("patience", 30))
    batch      = int(request.form.get("batch", 8))
    model_name = request.form.get("model_name", "yolo11s.pt")

    dataset_yaml = BASE_DIR / "Datas" / "out" / "step02_zone_detection" / "dataset" / "data.yaml"

    if not dataset_yaml.exists():
        return jsonify({"error": "Dataset introuvable — exportez d'abord les annotations."}), 400

    def _job():
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ERREUR] ultralytics non installé.")
            return
        # Corrige le path: dans data.yaml pour qu'il corresponde à l'environnement actuel
        import re as _re
        yaml_text = dataset_yaml.read_text()
        yaml_text = _re.sub(r"^path:.*$", f"path: {dataset_yaml.parent}", yaml_text, flags=_re.MULTILINE)
        dataset_yaml.write_text(yaml_text)

        models_dir = BASE_DIR / "Datas" / "out" / "step02_zone_detection" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"Entraînement YOLO détection — {epochs} epochs, img={imgsz}")
        print(f"Modèle de base : {model_name}")
        model = YOLO(model_name)
        model.train(
            data=str(dataset_yaml),
            epochs=epochs, imgsz=imgsz, batch=batch, patience=patience,
            project=str(models_dir), name="run", exist_ok=True,
            max_det=50,
        )
        best = models_dir / "run" / "weights" / "best.pt"
        print(f"Modèle sauvegardé : {best}")

    return jsonify({"job_id": _start_job(_job)})


# DETECTION_MODEL, DETECTION_CSV, DETECTION_PARTS importés depuis source.paths


# ── Route : état du modèle + stats des prédictions ───────────────────────────

@app.route("/step02/predict-status")
def step02_predict_status():
    from datetime import datetime

    def _fmt(p):
        try:
            return datetime.fromtimestamp(p.stat().st_mtime).strftime("%d/%m/%Y à %H:%M")
        except OSError:
            return None

    model_ok = DETECTION_MODEL.exists()

    parts_stats = {}
    if DETECTION_PARTS.exists():
        for cls_dir in sorted(DETECTION_PARTS.iterdir()):
            if cls_dir.is_dir():
                parts_stats[cls_dir.name] = len(list(cls_dir.iterdir()))

    csv_stats = {}
    if DETECTION_CSV.exists():
        import csv as _csv
        with open(DETECTION_CSV, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        from collections import Counter
        counts = Counter(r["class"] for r in rows if r.get("class"))
        csv_stats = {
            "total_images": len({r["filename"] for r in rows}),
            "total_boxes":  sum(counts.values()),
            "by_class":     [{"class": c, "count": n} for c, n in sorted(counts.items())],
            "csv_mtime":    _fmt(DETECTION_CSV),
        }

    return jsonify({
        "model_exists":  model_ok,
        "model_trained_at": _fmt(DETECTION_MODEL) if model_ok else None,
        "parts_stats":   parts_stats,
        "csv_stats":     csv_stats,
    })


# ── Route : appliquer le modèle sur le dataset ────────────────────────────────

@app.route("/step02/predict", methods=["POST"])
def step02_predict():
    conf       = float(request.form.get("conf", 0.25))
    save_crops = request.form.get("save_crops", "false") == "true"

    if not DETECTION_MODEL.exists():
        return jsonify({"error": "Modèle introuvable — entraînez d'abord le modèle de détection."}), 400

    def _job():
        from source.step02_zone_annotation.predict import collect_images, run_predict
        from collections import Counter

        all_rows: list[dict] = []

        for collection, preprocess_dir in PREPROCESSED_DIRS.items():
            if not preprocess_dir.exists():
                print(f"[INFO] Dossier prétraité absent pour {collection}, ignoré.")
                continue
            images = collect_images(preprocess_dir)
            if not images:
                print(f"[INFO] Aucune image dans {preprocess_dir.relative_to(BASE_DIR)}")
                continue
            print(f"\n── {collection.upper()} : {len(images)} images ──")
            rows = run_predict(
                images,
                model_path=DETECTION_MODEL,
                collection=collection,
                conf=conf,
                output_csv=DETECTION_CSV,
                save_crops=save_crops,
            )
            all_rows.extend(rows)

        if not all_rows:
            print("[ERREUR] Aucune image trouvée dans les dossiers prétraités.")
            return

        # Réécrit le CSV consolidé (run_predict écrit au fur et à mesure, ici on consolide)
        import csv as _csv
        with open(DETECTION_CSV, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

        counts = Counter(r["class"] for r in all_rows if r.get("class"))
        n_images = len({r["filename"] for r in all_rows})
        print(f"\n{sum(counts.values())} zones détectées — images traitées : {n_images} :")
        for cls, n in sorted(counts.items()):
            print(f"  {cls} : {n}")
        if save_crops:
            print(f"Crops sauvegardés dans : {DETECTION_PARTS.relative_to(BASE_DIR)}")
        print(f"CSV → {DETECTION_CSV.relative_to(BASE_DIR)}")

    return jsonify({"job_id": _start_job(_job)})
