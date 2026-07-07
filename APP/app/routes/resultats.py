"""
Routes Flask pour la page de visualisation des résultats (toutes étapes).

Points d'entrée :
  GET  /resultats                        → page HTML
  GET  /api/resultats/step01/samples     → échantillon images classifiées
  GET  /api/resultats/step02/samples     → échantillon crops zones
  GET  /api/resultats/step04/rows        → lignes HTR paginées/filtrées
  GET  /api/resultats/step05/rows        → lignes corrigées paginées/filtrées
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from flask import jsonify, render_template, request

from ..app import app
from source.paths import (
    CLS_PREDICT_CSV,
    DET_PARTS_DIR,
    DET_PREDICT_CSV,
    FIELDS_PARTS_DIR,
    FIELDS_PREDICT_CSV,
    HTR_RAW_CSV,
    HTR_GROUPED_CSV,
    CORRECTED_CSV,
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Couleurs par classe (step02 zones)
ZONE_COLORS = {
    "collecte":        "#2196F3",
    "determination":   "#4CAF50",
    "tampon":          "#FF9800",
    "numero_inventaire": "#9C27B0",
    "notes":           "#607D8B",
    "code_barre":      "#F44336",
    "graines":         "#8BC34A",
    "dessin":          "#00BCD4",
    "specimen":        "#795548",
    "logo":            "#E91E63",
}

# Couleurs par classe (step03 champs)
FIELD_COLORS = {
    "collecteur":            "#F44336",
    "date_collecte":         "#607D8B",
    "localite":              "#00BCD4",
    "determination":         "#4CAF50",
    "date_determination":    "#FF9800",
    "determinateur":         "#9C27B0",
    "statut_nomenclatural":  "#E91E63",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _all_zone_crops() -> list[dict]:
    """Crops step02 : DET_PARTS_DIR/<zone>/<filename>"""
    crops = []
    if not DET_PARTS_DIR.exists():
        return crops
    for zone_dir in sorted(DET_PARTS_DIR.iterdir()):
        if not zone_dir.is_dir():
            continue
        for p in zone_dir.iterdir():
            if p.suffix.lower() in EXTENSIONS:
                crops.append({"zone": zone_dir.name, "filename": p.name,
                               "url": f"/serve-image-zone/{zone_dir.name}/{p.name}"})
    return crops


def _all_field_crops() -> list[dict]:
    """Crops step03 : FIELDS_PARTS_DIR[zone]/<cls>/<filename>"""
    crops = []
    for zone, parts_dir in FIELDS_PARTS_DIR.items():
        if not parts_dir.exists():
            continue
        for cls_dir in sorted(parts_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in EXTENSIONS:
                    crops.append({
                        "zone": zone, "cls": cls_dir.name, "filename": p.name,
                        "url": f"/step03/serve-crop/{zone}/{cls_dir.name}/{p.name}",
                    })
    return crops


# ── Routes API ────────────────────────────────────────────────────────────────

def _collection_from_row(r: dict) -> str:
    """Déduit la collection depuis le filepath du CSV step01."""
    fp = r.get("filepath", "")
    if "entomology" in fp.lower():
        return "entomology"
    if "herbarium" in fp.lower():
        return "herbarium"
    # Fallback : regarder la classe prédite
    cls = r.get("predicted_class", "")
    return cls if cls in ("herbarium", "entomology") else "herbarium"


@app.route("/api/resultats/step01/samples")
def api_step01_samples():
    n      = int(request.args.get("n", 12))
    cls_f  = request.args.get("cls", "")
    rows   = _read_csv(CLS_PREDICT_CSV)
    if cls_f:
        rows = [r for r in rows if r.get("predicted_class") == cls_f]
    chosen = random.sample(rows, min(n, len(rows))) if rows else []
    return jsonify([{
        "filename":   r.get("filename", ""),
        "collection": _collection_from_row(r),
        "cls":        r.get("predicted_class", ""),
        "conf":       r.get("confidence", ""),
        "url":        f"/serve-image/{_collection_from_row(r)}/{r.get('filename','')}",
    } for r in chosen])


@app.route("/api/resultats/step02/samples")
def api_step02_samples():
    n      = int(request.args.get("n", 12))
    zone_f = request.args.get("zone", "")
    crops  = _all_zone_crops()
    if zone_f:
        crops = [c for c in crops if c["zone"] == zone_f]
    return jsonify(random.sample(crops, min(n, len(crops))))


@app.route("/api/resultats/step03/samples")
def api_step03_samples():
    n      = int(request.args.get("n", 12))
    zone_f = request.args.get("zone", "")
    cls_f  = request.args.get("cls", "")
    crops  = _all_field_crops()
    if zone_f:
        crops = [c for c in crops if c["zone"] == zone_f]
    if cls_f:
        crops = [c for c in crops if c["cls"] == cls_f]
    return jsonify(random.sample(crops, min(n, len(crops))))


@app.route("/api/resultats/step04/rows")
def api_step04_rows():
    n      = int(request.args.get("n", 50))
    cls_f  = request.args.get("cls", "")
    q      = request.args.get("q", "").lower()
    rows   = _read_csv(HTR_RAW_CSV)
    if cls_f:
        rows = [r for r in rows if r.get("field_class") == cls_f]
    if q:
        rows = [r for r in rows if q in r.get("text", "").lower()
                or q in r.get("specimen_id", "").lower()]
    total = len(rows)
    chosen = random.sample(rows, min(n, total)) if rows else []
    return jsonify({"total": total, "rows": chosen})


@app.route("/api/resultats/step05/rows")
def api_step05_rows():
    n      = int(request.args.get("n", 50))
    status = request.args.get("status", "")
    q      = request.args.get("q", "").lower()
    rows   = _read_csv(CORRECTED_CSV)
    if status:
        rows = [r for r in rows if
                r.get("localite_statut") == status
                or r.get("date_collecte_statut") == status
                or r.get("determination_1_statut") == status]
    if q:
        rows = [r for r in rows if
                q in r.get("specimen_id", "").lower()
                or q in r.get("localite", "").lower()
                or q in r.get("collecteur", "").lower()]
    total = len(rows)
    chosen = random.sample(rows, min(n, total)) if rows else []
    return jsonify({"total": total, "rows": chosen})


@app.route("/serve-image-zone/<zone>/<filename>")
def serve_image_zone(zone, filename):
    """Sert un crop de zone step02."""
    from flask import send_file
    path = DET_PARTS_DIR / zone / filename
    if path.exists():
        return send_file(path)
    return "", 404


# ── Page principale ───────────────────────────────────────────────────────────

def _stats_step01():
    rows = _read_csv(CLS_PREDICT_CSV)
    if not rows:
        return None
    counts: dict[str, int] = {}
    for r in rows:
        cls = r.get("predicted_class", "")
        if cls:
            counts[cls] = counts.get(cls, 0) + 1
    classes = sorted(counts)
    return {
        "total": len(rows),
        "by_class": [{"cls": c, "count": counts[c]} for c in classes],
        "classes": classes,
    }


def _stats_step02():
    rows = _read_csv(DET_PREDICT_CSV)
    if not rows:
        return None
    counts: dict[str, int] = {}
    for r in rows:
        cls = r.get("class", "")
        if cls:
            counts[cls] = counts.get(cls, 0) + 1
    n_imgs = len({r["filename"] for r in rows if r.get("filename")})
    classes = sorted(counts)
    return {
        "total_images": n_imgs,
        "total_boxes": sum(counts.values()),
        "by_class": [{"cls": c, "count": counts[c]} for c in classes],
        "classes": classes,
    }


def _stats_step03():
    """Compte les crops step03 directement depuis les dossiers FIELDS_PARTS_DIR."""
    counts: dict[str, int] = {}
    by_zone: dict[str, int] = {}
    total_images = 0

    for zone, parts_dir in FIELDS_PARTS_DIR.items():
        if not parts_dir.exists():
            continue
        for cls_dir in parts_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            n = len([p for p in cls_dir.iterdir() if p.suffix.lower() in EXTENSIONS])
            if n:
                counts[cls_dir.name] = counts.get(cls_dir.name, 0) + n
                by_zone[zone] = by_zone.get(zone, 0) + n
                total_images += n

    if not counts:
        return None

    total_boxes = sum(counts.values())
    classes = sorted(counts)
    return {
        "total_images": total_images,
        "total_boxes":  total_boxes,
        "by_zone":      by_zone,
        "by_class":     [{"cls": c, "count": counts[c]} for c in classes],
        "classes":      classes,
    }


def _stats_step04():
    # Compter les crops disponibles (indépendamment des transcriptions HTR)
    crop_counts: dict[str, int] = {}
    for parts_dir in FIELDS_PARTS_DIR.values():
        if not parts_dir.exists():
            continue
        for cls_dir in parts_dir.iterdir():
            if cls_dir.is_dir():
                n = len([p for p in cls_dir.iterdir() if p.suffix.lower() in EXTENSIONS])
                if n:
                    crop_counts[cls_dir.name] = crop_counts.get(cls_dir.name, 0) + n

    # Transcriptions si disponibles
    rows = _read_csv(HTR_RAW_CSV)
    htr_counts: dict[str, int] = {}
    n_spec = 0
    if rows:
        for r in rows:
            cls = r.get("field_class", "")
            if cls:
                htr_counts[cls] = htr_counts.get(cls, 0) + 1
        n_spec = len({r["specimen_id"] for r in rows if r.get("specimen_id")})

    # Au moins les crops doivent exister
    if not crop_counts:
        return None

    classes = sorted(crop_counts)
    return {
        "total_crops":      sum(crop_counts.values()),
        "total_specimens":  n_spec,
        "htr_done":         bool(rows),
        "by_class":         [{"cls": c, "count": crop_counts[c],
                              "htr": htr_counts.get(c, 0)} for c in classes],
        "classes":          classes,
    }


def _stats_step05():
    rows = _read_csv(CORRECTED_CSV)
    if not rows:
        return None
    geo_ok  = sum(1 for r in rows if r.get("localite_statut") == "OK")
    geo_tot = sum(1 for r in rows if r.get("localite_brut"))
    gbif_ok = sum(1 for r in rows if r.get("determination_1_statut") == "OK")
    gbif_tot = sum(1 for r in rows if r.get("determination_1_brut"))
    date_ok = sum(1 for r in rows if r.get("date_collecte_statut") == "OK")
    date_tot = sum(1 for r in rows if r.get("date_collecte_brut"))
    geo_rows = [r for r in rows if r.get("latitude") and r.get("longitude")]
    return {
        "total":       len(rows),
        "geo_ok":      geo_ok,
        "geo_total":   geo_tot,
        "gbif_ok":     gbif_ok,
        "gbif_total":  gbif_tot,
        "date_ok":     date_ok,
        "date_total":  date_tot,
        "has_geo":     len(geo_rows),
        "geo_points":  [{"lat": r["latitude"], "lon": r["longitude"],
                         "label": r.get("localite", r.get("specimen_id", ""))}
                        for r in geo_rows[:500]],
    }


@app.route("/resultats")
def resultats():
    return render_template(
        "pages/resultats.html",
        s1=_stats_step01(),
        s2=_stats_step02(),
        s3=_stats_step03(),
        s4=_stats_step04(),
        s5=_stats_step05(),
        zone_colors=ZONE_COLORS,
        field_colors=FIELD_COLORS,
    )
