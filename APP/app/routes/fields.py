"""
Routes Flask pour l'étape 3 : détection des champs de métadonnées.

Architecture à deux projets Label Studio / deux modèles YOLO :
  - zone "collecte"      → classes : collecteur, date_collecte, localite
  - zone "determination" → classes : determination, date_determination,
                                     determinateur, statut_nomenclatural

Les crops produits par les deux modèles alimentent ensuite le même pipeline HTR (step04).

Points d'entrée :
  GET  /step03/status               → état LS (2 projets) + stats + modèles (JSON)
  POST /step03/create-project       → crée/retrouve les 2 projets LS (SSE)
  POST /step03/import-images        → importe les crops step02 dans les 2 projets (SSE)
  POST /step03/export-yolo          → exporte annotations → 2 datasets YOLO (SSE)
  POST /step03/train                → entraîne les 2 modèles séquentiellement (SSE)
  POST /step03/predict              → détecte les champs (modèle auto-sélectionné par zone) (SSE)
  GET  /serve-field-image/<zone>/<filename>  → sert les crops step02 vers LS
"""

from __future__ import annotations
import csv
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from flask import jsonify, request, send_file

from ..app import app
from ..routes.generales import _start_job, BASE_DIR
from source.paths import (
    DET_PARTS_DIR,
    FIELDS_BEST_PT,
    FIELDS_DATASET_DIR,
    FIELDS_PREDICT_CSV,
    FIELDS_PARTS_DIR,
)
from source.step03_fields_detection.label_studio_manager import (
    ZONE_CONFIGS,
    CLASS_NAMES,
    LSManager,
)
from source.step03_fields_detection.export_to_yolo import convert as export_to_yolo

ENV_PATH   = BASE_DIR / ".env"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Zones step02 annotées en step03
PARTS_TEXT_ZONES = ["collecte", "determination"]
PARTS_DIRS       = [DET_PARTS_DIR / z for z in PARTS_TEXT_ZONES]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ls_config() -> tuple[str, str]:
    return (
        os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
        os.environ.get("LABEL_STUDIO_API_KEY", ""),
    )


def _ls_public_url() -> str:
    return os.environ.get(
        "LABEL_STUDIO_PUBLIC_URL",
        os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
    )


def _ls(zone_type: str) -> LSManager:
    url, key = _ls_config()
    return LSManager(url=url, api_key=key, zone_type=zone_type)


def _saved_project_id(zone_type: str) -> int | None:
    key = ZONE_CONFIGS[zone_type]["env_key"]
    pid = os.environ.get(key, "")
    return int(pid) if pid.isdigit() else None


def _save_env_key(key: str, value: str) -> None:
    content = ENV_PATH.read_text() if ENV_PATH.exists() else ""
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    new_line = f"{key}={value}"
    if pattern.search(content):
        content = pattern.sub(new_line, content)
    else:
        content = content.rstrip("\n") + f"\n{new_line}\n"
    ENV_PATH.write_text(content)
    os.environ[key] = value


def _fmt(p: Path) -> str | None:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%d/%m/%Y à %H:%M")
    except OSError:
        return None


def _field_images() -> list[Path]:
    """Retourne tous les crops step02 des zones texte (herbier + entomologie)."""
    images = []
    for d in PARTS_DIRS:
        if d.exists():
            images.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in EXTENSIONS))
    return images


def _collection_of(stem: str) -> str:
    """Détecte la collection depuis le nom du crop : EC+chiffres → entomologie, P+chiffres → herbier."""
    if re.search(r'EC\d+', stem, re.IGNORECASE):
        return "entomology"
    if re.search(r'P\d+', stem, re.IGNORECASE):
        return "herbarium"
    return "unknown"


def _balanced_selection(images: list[Path], max_total: int | None) -> list[Path]:
    """
    Sélection équilibrée sur 4 axes : herbier×collecte, herbier×détermination,
    entomologie×collecte, entomologie×détermination.

    ⚠ Détection collection par regex : EC\\d+ → entomologie, P\\d+ → herbier.
    À adapter si de nouvelles collections sont ajoutées (voir _collection_of()).
    """
    buckets: dict[tuple[str, str], list[Path]] = {}
    for img in images:
        zone = img.parent.name
        col  = _collection_of(img.stem)
        buckets.setdefault((col, zone), []).append(img)

    if not buckets:
        return images

    per_bucket = min(len(v) for v in buckets.values())
    if max_total:
        per_bucket = min(per_bucket, max_total // len(buckets))

    selected = []
    for key in sorted(buckets):
        selected.extend(buckets[key][:per_bucket])
    return selected


# ── Route : servir les images vers Label Studio ───────────────────────────────

@app.route("/serve-field-image/<zone>/<path:filename>")
def serve_field_image(zone: str, filename: str):
    """Sert un crop step02 par zone : collecte ou determination."""
    if zone not in PARTS_TEXT_ZONES:
        return "zone inconnue", 404
    img_path = DET_PARTS_DIR / zone / filename
    if img_path.exists() and img_path.suffix.lower() in EXTENSIONS:
        response = send_file(str(img_path), mimetype="image/jpeg")
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        return response
    return "image introuvable", 404


# ── Route : état global ───────────────────────────────────────────────────────

@app.route("/step03/status")
def step03_status():
    url, api_key = _ls_config()

    is_up  = False
    authed = False
    try:
        mgr    = LSManager(url=url, api_key=api_key, zone_type="collecte")
        is_up  = mgr.ping()
        authed = mgr.check_auth() if is_up and api_key else False
    except Exception:
        pass

    projects = {}
    for z in PARTS_TEXT_ZONES:
        pid   = _saved_project_id(z)
        stats = {}
        if authed and pid:
            try:
                stats = LSManager(url=url, api_key=api_key, zone_type=z).project_stats(pid)
            except Exception:
                pass
        projects[z] = {
            "project_id": pid,
            "stats":      stats,
            "model_exists": FIELDS_BEST_PT[z].exists(),
            "model_trained_at": _fmt(FIELDS_BEST_PT[z]) if FIELDS_BEST_PT[z].exists() else None,
        }

    field_imgs = _field_images()

    # Nombre de crops déjà détectés (présents dans FIELDS_PARTS_DIR)
    EXTENSIONS_SET = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    treated_crops = 0
    for parts_dir in FIELDS_PARTS_DIR.values():
        if parts_dir.exists():
            for cls_dir in parts_dir.iterdir():
                if cls_dir.is_dir():
                    treated_crops += sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in EXTENSIONS_SET)

    return jsonify({
        "ls_url":           _ls_public_url(),
        "is_up":            is_up,
        "authed":           authed,
        "api_key_set":      bool(api_key),
        "projects":         projects,
        "available_images": len(field_imgs),
        "zones":            {z: len([p for p in field_imgs if p.parent.name == z]) for z in PARTS_TEXT_ZONES},
        "treated_crops":    treated_crops,
    })


# ── Route : créer / retrouver les projets ────────────────────────────────────

@app.route("/step03/create-project", methods=["POST"])
def step03_create_project():
    def _job():
        url, api_key = _ls_config()
        if not api_key:
            print("[ERREUR] Clé API Label Studio non configurée.")
            return
        for zone in PARTS_TEXT_ZONES:
            mgr = LSManager(url=url, api_key=api_key, zone_type=zone)
            if not mgr.ping():
                print(f"[ERREUR] Label Studio inaccessible à {url}")
                return
            pid = mgr.get_or_create_project()
            _save_env_key(ZONE_CONFIGS[zone]["env_key"], str(pid))
            print(f"[{zone}] Projet '{mgr.project_title}' — id={pid}")
            print(f"[{zone}] Classes : {', '.join(CLASS_NAMES[zone])}")
        print(f"Ouvrez Label Studio : {url}/projects/")

    return jsonify({"job_id": _start_job(_job)})


# ── Route : importer les images dans les 2 projets ───────────────────────────

@app.route("/step03/import-images", methods=["POST"])
def step03_import_images():
    max_images  = request.form.get("max_images", "")
    max_images  = int(max_images) if max_images.isdigit() else None
    clear_first = request.form.get("clear_first") == "true"
    flask_host  = request.host_url.rstrip("/")

    def _job():
        url, api_key = _ls_config()
        for zone in PARTS_TEXT_ZONES:
            pid = _saved_project_id(zone)
            if not api_key or pid is None:
                print(f"[ERREUR] Projet {zone} non créé — lancez d'abord 'Créer les projets'.")
                return

        images = _field_images()
        if not images:
            print("[ERREUR] Aucune image disponible. Vérifiez que l'étape 2 a été exécutée avec save_crops=True.")
            return

        for zone in PARTS_TEXT_ZONES:
            herb = sum(1 for p in images if p.parent.name == zone and _collection_of(p.stem) == "herbarium")
            ento = sum(1 for p in images if p.parent.name == zone and _collection_of(p.stem) == "entomology")
            print(f"  {zone}: {herb} herbier + {ento} entomologie")

        images = _balanced_selection(images, max_images)
        dist   = Counter(f"{_collection_of(p.stem)}/{p.parent.name}" for p in images)
        print(f"Sélection équilibrée ({len(images)} images) :")
        for k, n in sorted(dist.items()):
            print(f"  {k}: {n}")

        # Importer dans le bon projet selon la zone
        images_by_zone = {z: [p for p in images if p.parent.name == z] for z in PARTS_TEXT_ZONES}
        for zone, zone_imgs in images_by_zone.items():
            pid = _saved_project_id(zone)
            mgr = LSManager(url=url, api_key=api_key, zone_type=zone)
            if clear_first:
                print(f"[{zone}] Suppression des tâches existantes…")
                mgr.delete_all_tasks(pid)
                existing_urls: set[str] = set()
            else:
                print(f"[{zone}] Récupération des images déjà importées…")
                existing_urls = mgr.get_existing_urls(pid)
                print(f"[{zone}] {len(existing_urls)} images déjà présentes.")
            urls = [
                f"{flask_host}/serve-field-image/{zone}/{img.name}"
                for img in zone_imgs
                if f"{flask_host}/serve-field-image/{zone}/{img.name}" not in existing_urls
            ]
            if not urls:
                print(f"[{zone}] Aucune nouvelle image à importer.")
                continue
            imported = mgr.import_images(pid, urls)
            print(f"[{zone}] {imported} nouvelles images importées.")

        ls_url = url
        print(f"Ouvrez Label Studio pour annoter : {ls_url}/projects/")

    return jsonify({"job_id": _start_job(_job)})


# ── Route : exporter les annotations → 2 datasets YOLO ───────────────────────

@app.route("/step03/export-yolo", methods=["POST"])
def step03_export_yolo():
    flask_base = request.host_url.rstrip("/") + "/serve-field-image"

    def _job():
        url, api_key = _ls_config()
        for zone in PARTS_TEXT_ZONES:
            pid = _saved_project_id(zone)
            if not api_key or pid is None:
                print(f"[ERREUR] Projet {zone} non configuré.")
                continue
            mgr   = LSManager(url=url, api_key=api_key, zone_type=zone)
            tasks = mgr.export_tasks(pid)
            print(f"[{zone}] {len(tasks)} tâches récupérées.")
            result = export_to_yolo(tasks, zone_type=zone, flask_base_url=flask_base)
            if "error" in result:
                print(f"[ERREUR] {result['error']}")
            else:
                for split, n in result["splits"].items():
                    print(f"  {split:5s} → {n} images")

    return jsonify({"job_id": _start_job(_job)})


# ── Route : entraîner les 2 modèles ──────────────────────────────────────────

@app.route("/step03/train", methods=["POST"])
def step03_train():
    epochs     = int(request.form.get("epochs", 150))
    imgsz      = int(request.form.get("imgsz", 640))
    patience   = int(request.form.get("patience", 30))
    batch      = int(request.form.get("batch", 4))
    model_name = request.form.get("model_name", "yolo11n.pt")
    zone_filter = request.form.get("zone", "both")  # "collecte" | "determination" | "both"

    def _job():
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ERREUR] ultralytics non installé.")
            return

        zones_to_train = PARTS_TEXT_ZONES if zone_filter == "both" else [zone_filter]

        for zone in zones_to_train:
            dataset_yaml = FIELDS_DATASET_DIR[zone] / "data.yaml"
            if not dataset_yaml.exists():
                print(f"[{zone}] Dataset introuvable — exportez d'abord les annotations.")
                continue

            import re as _re
            yaml_text = dataset_yaml.read_text()
            yaml_text = _re.sub(r"^path:.*$", f"path: {dataset_yaml.parent}", yaml_text, flags=_re.MULTILINE)
            dataset_yaml.write_text(yaml_text)

            models_dir = FIELDS_BEST_PT[zone].parent.parent.parent
            models_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n── [{zone}] Entraînement — {epochs} epochs, img={imgsz} ──")
            model = YOLO(model_name)
            model.train(
                data=str(dataset_yaml),
                epochs=epochs, imgsz=imgsz, batch=batch, patience=patience,
                project=str(models_dir), name="run", exist_ok=True,
                cache=False, workers=0, max_det=50,
            )
            print(f"[{zone}] Modèle → {FIELDS_BEST_PT[zone]}")

    return jsonify({"job_id": _start_job(_job)})


# ── Route : appliquer les modèles sur les crops step02 ───────────────────────

@app.route("/step03/predict", methods=["POST"])
def step03_predict():
    conf       = float(request.form.get("conf", 0.25))
    save_crops = request.form.get("save_crops", "false") == "true"

    missing = [z for z in PARTS_TEXT_ZONES if not FIELDS_BEST_PT[z].exists()]
    if missing:
        return jsonify({"error": f"Modèle(s) manquant(s) : {missing}. Entraînez d'abord."}), 400

    def _job():
        from source.step03_fields_detection.predict import collect_images, run_predict
        import csv as _csv

        images = []
        for d in PARTS_DIRS:
            if d.exists():
                images.extend(collect_images(d))

        if not images:
            print("[ERREUR] Aucun crop trouvé — vérifiez que l'étape 2 a été exécutée avec save_crops=True.")
            return

        for zone in PARTS_TEXT_ZONES:
            n = sum(1 for p in images if p.parent.name == zone)
            print(f"  {zone}: {n} crops")

        models = {z: FIELDS_BEST_PT[z] for z in PARTS_TEXT_ZONES}
        all_rows = run_predict(images, models, conf=conf, save_crops=save_crops)

        counts   = Counter(r["class"] for r in all_rows if r.get("class"))
        n_images = len({r["filename"] for r in all_rows})
        total_crops = sum(counts.values())
        print(f"\n{total_crops} champs détectés sur {n_images} images ({n_images}/{len(images)} images traitées) :")
        for cls, n in sorted(counts.items()):
            print(f"  {cls} : {n}")
        if save_crops:
            for zone in PARTS_TEXT_ZONES:
                n_crops = sum(1 for p in FIELDS_PARTS_DIR[zone].rglob("*") if p.is_file()) if FIELDS_PARTS_DIR[zone].exists() else 0
                print(f"  Crops [{zone}] → {FIELDS_PARTS_DIR[zone].relative_to(BASE_DIR)} ({n_crops} fichiers)")

    return jsonify({"job_id": _start_job(_job)})


# ── Route : état des modèles + stats ─────────────────────────────────────────

@app.route("/step03/serve-crop/<zone>/<cls>/<filename>")
def step03_serve_crop(zone, cls, filename):
    """Sert un crop de champ détecté (step03 predict)."""
    path = FIELDS_PARTS_DIR.get(zone, Path("/dev/null")) / cls / filename
    if not path.exists():
        return "", 404
    return send_file(path)


@app.route("/step03/sample-crops")
def step03_sample_crops():
    """
    Retourne un échantillon de crops par (zone, classe) pour visualisation.
    Query params : n=20 (max par classe), zone= (filtre optionnel), cls= (filtre optionnel).
    """
    import random
    n_max   = int(request.args.get("n", 4))
    z_filter = request.args.get("zone", "")
    c_filter = request.args.get("cls", "")

    samples = []
    for zone, parts_dir in FIELDS_PARTS_DIR.items():
        if z_filter and zone != z_filter:
            continue
        if not parts_dir.exists():
            continue
        for cls_dir in sorted(parts_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            if c_filter and cls != c_filter:
                continue
            imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            chosen = random.sample(imgs, min(n_max, len(imgs)))
            for p in chosen:
                samples.append({
                    "zone": zone,
                    "cls":  cls,
                    "filename": p.name,
                    "url": f"/step03/serve-crop/{zone}/{cls}/{p.name}",
                })

    random.shuffle(samples)
    return jsonify({"samples": samples, "total": len(samples)})


@app.route("/step03/predict-status")
def step03_predict_status():
    parts_stats = {}
    for zone in PARTS_TEXT_ZONES:
        parts_dir = FIELDS_PARTS_DIR[zone]
        if parts_dir.exists():
            for cls_dir in sorted(parts_dir.iterdir()):
                if cls_dir.is_dir():
                    parts_stats[f"{zone}/{cls_dir.name}"] = len(list(cls_dir.iterdir()))

    return jsonify({
        "models": {
            z: {
                "exists":     FIELDS_BEST_PT[z].exists(),
                "trained_at": _fmt(FIELDS_BEST_PT[z]) if FIELDS_BEST_PT[z].exists() else None,
            }
            for z in PARTS_TEXT_ZONES
        },
        "parts_stats": parts_stats,
    })
