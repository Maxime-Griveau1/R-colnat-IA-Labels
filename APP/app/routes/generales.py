import contextlib
import csv
import io
import logging
import os
import queue
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import Response, render_template, request, stream_with_context, url_for, redirect, flash

from ..app import app
from source.preprocessImages import preprocessImages
from source.step01_label_classification.prepare_dataset import main as prepare_dataset_main

# ── Streaming infrastructure (SSE) ───────────────────────────────────────────

_jobs: dict[str, queue.Queue] = {}
_jobs_lock = threading.Lock()


# Supprime tous les codes d'échappement ANSI (couleurs, déplacements curseur, etc.)
_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _clean(s: str) -> str:
    return _ANSI_RE.sub("", s)


class _QueueWriter(io.TextIOBase):
    """
    Redirige stdout/stderr vers une queue, proprement :
    - Supprime les séquences ANSI (couleurs, [K, déplacements curseur…)
    - Gère les \\r de tqdm : chaque \\r écrase le contenu de la ligne courante,
      seule la version finale (avant \\n) est émise → une ligne par epoch, pas 9.
    """
    def __init__(self, q: queue.Queue):
        self._q = q
        self._line = ""   # ligne en cours de construction (réinitialisée par \r)

    def write(self, s: str) -> int:
        s = _clean(s)
        for ch in s:
            if ch == "\r":
                self._line = ""          # retour chariot → on réécrit depuis le début
            elif ch == "\n":
                line = self._line.strip()
                if line:
                    self._q.put(line)
                self._line = ""
            else:
                self._line += ch
        return len(s)

    def flush(self):
        pass


class _QueueLogHandler(logging.Handler):
    """Capture les logs Python (ultralytics utilise logging) vers la queue SSE."""
    def __init__(self, q: queue.Queue):
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord):
        msg = _clean(self.format(record)).strip()
        if msg:
            self._q.put(msg)


def _start_job(fn) -> str:
    """Lance fn dans un thread, retourne le job_id pour le SSE."""
    job_id = str(uuid.uuid4())[:8]
    q: queue.Queue = queue.Queue()
    with _jobs_lock:
        _jobs[job_id] = q

    def _run():
        writer = _QueueWriter(q)
        log_handler = _QueueLogHandler(q)
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        # Branche sur le logger racine ET sur celui d'ultralytics
        root_logger = logging.getLogger()
        ultra_logger = logging.getLogger("ultralytics")
        root_logger.addHandler(log_handler)
        ultra_logger.addHandler(log_handler)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                fn()
        except Exception as exc:
            q.put(f"[ERREUR] {exc}")
        finally:
            root_logger.removeHandler(log_handler)
            ultra_logger.removeHandler(log_handler)
            q.put(None)  # sentinelle de fin

    threading.Thread(target=_run, daemon=True).start()
    return job_id


@app.route("/stream/<job_id>")
def stream(job_id: str):
    """Endpoint SSE : diffuse les lignes d'un job en cours."""
    with _jobs_lock:
        q = _jobs.get(job_id)
    if q is None:
        return "job introuvable", 404

    def generate():
        while True:
            try:
                line = q.get(timeout=30)
            except queue.Empty:
                # Heartbeat : garde la connexion vivante pendant les longues epochs
                yield "data: [en cours…]\n\n"
                continue
            if line is None:
                yield "data: __DONE__\n\n"
                with _jobs_lock:
                    _jobs.pop(job_id, None)
                break
            # Les sauts de ligne dans data SSE sont interdits → on les remplace
            safe = line.replace("\n", " ").replace("\r", "")
            yield f"data: {safe}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


BASE_DIR = Path(app.root_path).parent

# ── Chemins de données ────────────────────────────────────────────────────────

# Images prétraitées (étape 0 — resize 1024×1024)
PREPROCESS_OUT = {
    "herbarium":  BASE_DIR / "Datas" / "out" / "step00_preprocessing" / "herbarium",
    "entomology": BASE_DIR / "Datas" / "out" / "step00_preprocessing" / "entomology",
}
# Images sources d'origine
SOURCE_IN = {
    "herbarium":  BASE_DIR / "Datas" / "in" / "00_source_images" / "herbarium" / "jpeg",
    "entomology": BASE_DIR / "Datas" / "in" / "00_source_images" / "entomology",
}
from source.paths import (
    CLS_BEST_PT as BEST_PT,
    CLS_RESULTS_CSV as RESULTS_CSV,
    CLS_DATASET_DIR as DATASET_DIR,
    CLS_PREDICT_CSV as PREDICT_CSV,
    DET_BEST_PT,
    DET_RESULTS_CSV,
    DET_DATASET_DIR,
    DET_PREDICT_CSV,
    FIELDS_BEST_PT,
    FIELDS_DATASET_DIR,
    FIELDS_PREDICT_CSV,
    FIELDS_PARTS_DIR,
    HTR_RAW_CSV,
    HTR_GROUPED_CSV,
    CORRECTED_CSV,
)


def _fmt_mtime(path: Path) -> str | None:
    """Retourne la date de modification d'un fichier en format lisible, ou None."""
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%d/%m/%Y à %H:%M")
    except OSError:
        return None


# ── Helpers de statut ─────────────────────────────────────────────────────────

def _dataset_stats() -> dict:
    """Lit le nombre d'images par split/classe dans le dataset préparé."""
    stats = {}
    if not DATASET_DIR.exists():
        return stats
    for split in ("train", "val", "test"):
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
        stats[split] = {}
        _img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                stats[split][cls_dir.name] = len([
                    p for p in cls_dir.iterdir()
                    if p.suffix.lower() in _img_exts
                ])
    return stats


def _model_status() -> dict:
    """Retourne l'état du modèle entraîné, avec date d'entraînement."""
    status = {
        "exists":      BEST_PT.exists(),
        "path":        str(BEST_PT),
        "trained_at":  _fmt_mtime(BEST_PT) if BEST_PT.exists() else None,
        "accuracy":    None,
        "epochs_done": None,
    }
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            acc_key = next((k for k in last if "accuracy" in k.lower()), None)
            status["accuracy"]    = round(float(last[acc_key]), 4) if acc_key else None
            status["epochs_done"] = len(rows)
    return status


def _prediction_stats() -> dict:
    """Lit le CSV de prédictions et retourne un résumé."""
    if not PREDICT_CSV.exists():
        return {}
    with open(PREDICT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    counts: dict[str, int] = {}
    conf_sum: dict[str, float] = {}
    uncertain = 0
    for r in rows:
        cls = r["predicted_class"]
        conf = float(r["confidence"])
        if cls == "uncertain":
            uncertain += 1
            continue
        counts[cls]    = counts.get(cls, 0) + 1
        conf_sum[cls]  = conf_sum.get(cls, 0.0) + conf
    avg_conf = {cls: round(conf_sum[cls] / counts[cls], 3) for cls in counts}
    return {
        "total":     len(rows),
        "uncertain": uncertain,
        "csv_mtime": _fmt_mtime(PREDICT_CSV),
        "by_class":  [
            {"class": cls, "count": counts[cls], "avg_conf": avg_conf[cls]}
            for cls in sorted(counts)
        ],
    }


def _preprocess_stats() -> dict:
    """Compte les images prétraitées disponibles."""
    stats = {}
    _img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for col, d in PREPROCESS_OUT.items():
        if d.exists():
            stats[col] = len([p for p in d.iterdir() if p.suffix.lower() in _img_exts])
        else:
            stats[col] = 0
    return stats


def _det_dataset_stats() -> dict:
    """Compte les images par split dans le dataset de détection."""
    stats = {}
    if not DET_DATASET_DIR.exists():
        return stats
    for split in ("train", "val", "test"):
        img_dir = DET_DATASET_DIR / split / "images"
        _img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        if img_dir.exists():
            stats[split] = len([p for p in img_dir.iterdir() if p.suffix.lower() in _img_exts])
    return stats


def _det_model_status() -> dict:
    """Retourne l'état du modèle de détection avec ses métriques."""
    status = {
        "exists":     DET_BEST_PT.exists(),
        "trained_at": _fmt_mtime(DET_BEST_PT) if DET_BEST_PT.exists() else None,
        "map50":      None,
        "map50_95":   None,
        "epochs_done": None,
    }
    if DET_RESULTS_CSV.exists():
        with open(DET_RESULTS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            # Strip whitespace from keys (YOLO sometimes adds spaces)
            last = {k.strip(): v.strip() for k, v in last.items()}
            map50_key    = next((k for k in last if "map50(b)" in k.lower() and "95" not in k.lower()), None)
            map50_95_key = next((k for k in last if "map50-95" in k.lower()), None)
            status["map50"]       = round(float(last[map50_key]), 4)    if map50_key    else None
            status["map50_95"]    = round(float(last[map50_95_key]), 4) if map50_95_key else None
            status["epochs_done"] = len(rows)
    return status


def _det_prediction_stats() -> dict:
    """Résume le CSV de détection (prédictions zones)."""
    if not DET_PREDICT_CSV.exists():
        return {}
    with open(DET_PREDICT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    counts: dict[str, int] = {}
    for r in rows:
        cls = r.get("class", "")
        if cls:
            counts[cls] = counts.get(cls, 0) + 1
    total_images = len({r["filename"] for r in rows})
    total_boxes  = sum(counts.values())
    return {
        "total_images": total_images,
        "total_boxes":  total_boxes,
        "csv_mtime":    _fmt_mtime(DET_PREDICT_CSV),
        "by_class": [
            {"class": cls, "count": counts[cls]}
            for cls in sorted(counts)
        ],
    }


_ZONES = ("collecte", "determination")


def _fields_dataset_stats() -> dict:
    """Compte les images par split dans le dataset des champs (par zone)."""
    _img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    result = {}
    for z in _ZONES:
        stats = {}
        for split in ("train", "val", "test"):
            img_dir = FIELDS_DATASET_DIR[z] / split / "images"
            if img_dir.exists():
                stats[split] = len([p for p in img_dir.iterdir() if p.suffix.lower() in _img_exts])
        result[z] = stats
    return result


def _fields_model_status() -> dict:
    """Retourne l'état des modèles de détection des champs (par zone)."""
    result = {}
    for z in _ZONES:
        pt = FIELDS_BEST_PT[z]
        status = {
            "exists":      pt.exists(),
            "trained_at":  _fmt_mtime(pt) if pt.exists() else None,
            "map50":       None,
            "map50_95":    None,
            "epochs_done": None,
        }
        results_csv = pt.parent.parent.parent / "results.csv"
        if results_csv.exists():
            with open(results_csv, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = {k.strip(): v.strip() for k, v in rows[-1].items()}
                map50_key    = next((k for k in last if "map50(b)" in k.lower() and "95" not in k.lower()), None)
                map50_95_key = next((k for k in last if "map50-95" in k.lower()), None)
                status["map50"]       = round(float(last[map50_key]), 4)    if map50_key    else None
                status["map50_95"]    = round(float(last[map50_95_key]), 4) if map50_95_key else None
                status["epochs_done"] = len(rows)
        result[z] = status
    return result


_HTR_FIELD_CLASSES = [
    "collecteur", "date_collecte", "date_determination",
    "determinateur", "determination", "localite", "numero_inventaire",
]
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _htr_parts_stats() -> dict[str, int]:
    """Compte les crops disponibles par classe depuis step03 (toutes zones)."""
    stats: dict[str, int] = {}
    for parts_dir in FIELDS_PARTS_DIR.values():
        if not parts_dir.exists():
            continue
        for cls_dir in parts_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            count = len([p for p in cls_dir.iterdir() if p.suffix.lower() in _IMG_EXTS])
            stats[cls_dir.name] = stats.get(cls_dir.name, 0) + count
    return stats


def _htr_stats() -> dict:
    """Résume les transcriptions GLM-OCR (transcriptions_raw.csv)."""
    if not HTR_RAW_CSV.exists():
        return {}
    with open(HTR_RAW_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    counts: dict[str, int] = {}
    for r in rows:
        cls = r.get("field_class", "")
        if cls:
            counts[cls] = counts.get(cls, 0) + 1
    n_specimens = len({r["specimen_id"] for r in rows if r.get("specimen_id")})
    return {
        "total_crops":     sum(counts.values()),
        "total_specimens": n_specimens,
        "raw_csv_mtime":   _fmt_mtime(HTR_RAW_CSV),
        "grouped_csv_exists": HTR_GROUPED_CSV.exists(),
        "by_class": [
            {"class": cls, "count": counts[cls]}
            for cls in sorted(counts)
        ],
    }


_CORRECT_STATUT_COLS = [
    "date_collecte_statut", "date_determination_statut",
    "determination_statut", "localite_statut", "numero_inventaire_statut",
]


def _correct_stats() -> dict:
    """Résume le CSV corrigé par colonne de statut."""
    if not CORRECTED_CSV.exists():
        return {}
    with open(CORRECTED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    by_col: dict[str, dict[str, int]] = {}
    for col in _CORRECT_STATUT_COLS:
        counts: dict[str, int] = {}
        for r in rows:
            s = r.get(col, "")
            if s:
                counts[s] = counts.get(s, 0) + 1
        by_col[col] = counts
    return {
        "total_specimens": len(rows),
        "csv_mtime": _fmt_mtime(CORRECTED_CSV),
        "by_col": by_col,
    }


def _fields_prediction_stats() -> dict:
    """
    Résume les résultats de détection des champs (toutes zones).
    Lit d'abord les CSV de prédiction s'ils existent, sinon compte
    les crops directement dans FIELDS_PARTS_DIR (cas le plus courant).
    """
    EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    # ── Essai 1 : CSV de prédiction ───────────────────────────────────────────
    rows: list[dict] = []
    last_mtime: str | None = None
    for z, csv_path in FIELDS_PREDICT_CSV.items():
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            zone_rows = list(csv.DictReader(f))
        for r in zone_rows:
            r.setdefault("zone", z)
        rows.extend(zone_rows)
        mt = _fmt_mtime(csv_path)
        if mt:
            last_mtime = mt

    if rows:
        counts: dict[str, int] = {}
        by_zone: dict[str, int] = {}
        for r in rows:
            cls  = r.get("class", "")
            zone = r.get("zone", "")
            if cls:
                counts[cls] = counts.get(cls, 0) + 1
            if zone:
                by_zone[zone] = by_zone.get(zone, 0) + 1
        return {
            "total_images": len({r["filename"] for r in rows}),
            "total_boxes":  sum(counts.values()),
            "by_zone":      by_zone,
            "csv_mtime":    last_mtime,
            "by_class":     [{"class": cls, "count": counts[cls]} for cls in sorted(counts)],
        }

    # ── Essai 2 : compter les crops dans les dossiers (pas de CSV) ────────────
    counts = {}
    by_zone = {}
    for zone, parts_dir in FIELDS_PARTS_DIR.items():
        if not parts_dir.exists():
            continue
        for cls_dir in parts_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            n = sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in EXTS)
            if n:
                counts[cls_dir.name] = counts.get(cls_dir.name, 0) + n
                by_zone[zone] = by_zone.get(zone, 0) + n

    if not counts:
        return {}

    return {
        "total_images": None,  # nb images sources inconnu sans CSV — évite dénominateur = total_boxes
        "total_boxes":  sum(counts.values()),
        "by_zone":      by_zone,
        "csv_mtime":    None,
        "by_class":     [{"class": cls, "count": counts[cls]} for cls in sorted(counts)],
    }


# ── Routes principales ────────────────────────────────────────────────────────

@app.route("/accueil")
@app.route("/")
def accueil():
    return render_template(
        "partials/conteneur.html",
        dataset_stats      = _dataset_stats(),
        model_status       = _model_status(),
        prediction_stats   = _prediction_stats(),
        preprocess_stats   = _preprocess_stats(),
        det_dataset_stats    = _det_dataset_stats(),
        det_model_status     = _det_model_status(),
        det_prediction_stats = _det_prediction_stats(),
        fields_dataset_stats    = _fields_dataset_stats(),
        fields_model_status     = _fields_model_status(),
        fields_prediction_stats = _fields_prediction_stats(),
        htr_parts_stats         = _htr_parts_stats(),
        htr_stats               = _htr_stats(),
        correct_stats           = _correct_stats(),
    )


# ── Étape 0 : Prétraitement (resize 1024×1024) ────────────────────────────────

@app.route("/pretraitement", methods=["POST"])
def lancer_pretraitement():
    """Lance le prétraitement en arrière-plan (SSE) et retourne un job_id."""
    max_images_str = request.form.get("max_images", "").strip()
    max_images   = int(max_images_str) if max_images_str.isdigit() else None
    use_clahe    = request.form.get("clahe",    "1") != "0"
    use_binarize = request.form.get("binarize", "0") == "1"
    target_size  = int(request.form.get("size", 1024))

    sources   = []
    in_paths  = []
    out_paths = []

    if request.form.get("source_herbarium"):
        in_paths.append(str(SOURCE_IN["herbarium"]))
        out_paths.append(str(PREPROCESS_OUT["herbarium"]))
        sources.append("Herbier")

    if request.form.get("source_entomology"):
        in_paths.append(str(SOURCE_IN["entomology"]))
        out_paths.append(str(PREPROCESS_OUT["entomology"]))
        sources.append("Entomologie")

    if not in_paths:
        return {"error": "Aucune source sélectionnée."}, 400

    def _job():
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Démarrage du prétraitement — sources : {', '.join(sources)}")
        if max_images:
            print(f"  Limite : {max_images} images par dossier")
        for src, dst in zip(in_paths, out_paths):
            src_p = Path(src)
            dst_p = Path(dst)
            if not src_p.exists():
                print(f"  [AVERTISSEMENT] Dossier source introuvable : {src_p}")
                continue
            n_src = len([p for p in src_p.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            print(f"  {src_p.name} — {n_src} images sources → {dst_p.relative_to(BASE_DIR)}")
        preprocessImages(in_paths, out_paths, max_images,
                         size=target_size, clahe=use_clahe, binarize=use_binarize)
        # Bilan
        for dst in out_paths:
            dst_p = Path(dst)
            n = len([p for p in dst_p.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if dst_p.exists() else 0
            print(f"  ✓ {Path(dst).name} : {n} images prétraitées")
        print("Prétraitement terminé.")

    return {"job_id": _start_job(_job)}


# ── Étape 1 : Reconnaissance du type d'étiquette (YOLO classification) ────────

@app.route("/step01/preparer-dataset", methods=["POST"])
def step01_preparer_dataset():
    """Prépare le dataset YOLO classification (train/val/test)."""
    max_images_str = request.form.get("max_images", "").strip()
    max_images = int(max_images_str) if max_images_str else None

    def _job():
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Préparation du dataset de classification")
        print(f"  Source : {SOURCE_IN['herbarium'].parent.parent}")
        print(f"  Destination : {DATASET_DIR.relative_to(BASE_DIR)}")
        if max_images:
            print(f"  Limite : {max_images} images par classe")
        prepare_dataset_main(max_images=max_images)
        stats = _dataset_stats()
        total = sum(n for split in stats.values() for n in split.values())
        print(f"Dataset prêt — {total} images au total.")
        for split, classes in stats.items():
            detail = ", ".join(f"{c}={n}" for c, n in classes.items())
            print(f"  {split:5s} → {sum(classes.values())} images ({detail})")

    return {"job_id": _start_job(_job)}


@app.route("/step01/entrainer", methods=["POST"])
def step01_entrainer():
    """Entraîne le modèle YOLO classification."""
    epochs     = int(request.form.get("epochs", 75))
    imgsz      = int(request.form.get("imgsz", 384))
    patience   = int(request.form.get("patience", 20))
    batch      = int(request.form.get("batch", 16))
    model_name = request.form.get("model_name", "yolo11s-cls.pt")

    if not DATASET_DIR.exists():
        return {"error": "Dataset introuvable — préparez d'abord le dataset."}, 400

    def _job():
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Entraînement YOLO classification")
        print(f"  Modèle de base : {model_name}")
        print(f"  Dataset : {DATASET_DIR.relative_to(BASE_DIR)}")
        print(f"  Paramètres : {epochs} epochs, imgsz={imgsz}")
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ERREUR] ultralytics n'est pas installé. Lancez : pip install ultralytics")
            return
        models_dir = BEST_PT.parent.parent.parent  # .../models
        models_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(model_name)
        model.train(
            data=str(DATASET_DIR),
            epochs=epochs, imgsz=imgsz, batch=batch, patience=patience,
            project=str(models_dir), name="run", exist_ok=True,
            degrees=5, translate=0.05, scale=0.1,
            fliplr=0.0, flipud=0.0, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
        )
        status = _model_status()
        acc = f" — accuracy top-1 : {status['accuracy']:.4f}" if status.get("accuracy") else ""
        trained_at = status.get("trained_at") or "maintenant"
        print(f"Entraînement terminé ({epochs} epochs){acc}")
        print(f"  Modèle sauvegardé : {BEST_PT.relative_to(BASE_DIR)}  [{trained_at}]")

    return {"job_id": _start_job(_job)}


@app.route("/step01/predire", methods=["POST"])
def step01_predire():
    """Lance la classification de toutes les images sources."""
    source         = request.form.get("predict_source", "both")
    conf_threshold = float(request.form.get("conf_threshold", 0.0))

    dirs_to_process = (
        [SOURCE_IN["herbarium"]]  if source == "herbarium"  else
        [SOURCE_IN["entomology"]] if source == "entomology" else
        list(SOURCE_IN.values())
    )

    if not BEST_PT.exists():
        return {"error": "Modèle introuvable — entraînez d'abord le modèle."}, 400

    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    images = [
        p for d in dirs_to_process
        for p in sorted(d.rglob("*")) if p.suffix.lower() in extensions
    ]
    if not images:
        return {"error": "Aucune image source trouvée."}, 400

    def _job():
        trained_at = _fmt_mtime(BEST_PT) or "date inconnue"
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Classification des images (step01)")
        print(f"  Modèle : {BEST_PT.relative_to(BASE_DIR)}  [entraîné le {trained_at}]")
        print(f"  Source : {source}  ({len(images)} images)")
        if conf_threshold > 0:
            print(f"  Seuil de confiance : {conf_threshold:.2f}")
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ERREUR] ultralytics n'est pas installé.")
            return
        model  = YOLO(str(BEST_PT))
        PREDICT_CSV.parent.mkdir(parents=True, exist_ok=True)
        rows  = []
        total = len(images)
        for i, img_path in enumerate(images, 1):
            results   = model(str(img_path), verbose=False)
            probs     = results[0].probs
            top1_idx  = int(probs.top1)
            top1_conf = float(probs.top1conf)
            top1_name = model.names[top1_idx] if top1_conf >= conf_threshold else "uncertain"
            top5      = probs.top5
            top2_name = model.names[int(top5[1])] if len(top5) > 1 else top1_name
            top2_conf = float(probs.top5conf[1]) if len(top5) > 1 else 0.0
            rows.append({
                "filename":        img_path.name,
                "filepath":        str(img_path),
                "predicted_class": top1_name,
                "confidence":      round(top1_conf, 4),
                "top2_class":      top2_name,
                "top2_conf":       round(top2_conf, 4),
            })
            if i % 50 == 0 or i == total:
                print(f"  {i}/{total} images traitées")
        with open(PREDICT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        from collections import Counter
        counts = Counter(r["predicted_class"] for r in rows)
        print(f"Classification terminée — {len(rows)} images :")
        for cls, n in sorted(counts.items()):
            print(f"  {cls} : {n} images")
        print(f"  CSV → {PREDICT_CSV.relative_to(BASE_DIR)}")

    return {"job_id": _start_job(_job)}
