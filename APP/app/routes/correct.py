"""
Routes Flask pour l'étape 5 : post-traitement / correction.

GET  /step05/status  → état des fichiers (JSON)
POST /step05/run     → lance la correction en arrière-plan (SSE)
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from flask import jsonify, request

from ..app import app
from ..routes.generales import _start_job
from source.paths import HTR_GROUPED_CSV, CORRECTED_CSV, CORRECT_OUT_DIR

STATUT_COLS = [
    "date_collecte_statut",
    "date_determination_statut",
    "determination_statut",
    "localite_statut",
    "numero_inventaire_statut",
]


def _fmt(p: Path) -> str | None:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%d/%m/%Y à %H:%M")
    except OSError:
        return None


def _corrected_stats() -> dict:
    if not CORRECTED_CSV.exists():
        return {}
    with open(CORRECTED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    by_col: dict[str, dict[str, int]] = {}
    for col in STATUT_COLS:
        counts: dict[str, int] = {}
        for r in rows:
            s = r.get(col, "")
            if s:
                counts[s] = counts.get(s, 0) + 1
        by_col[col] = counts

    return {
        "total_specimens": len(rows),
        "csv_mtime": _fmt(CORRECTED_CSV),
        "by_col": by_col,
    }


@app.route("/step05/status")
def step05_status():
    return jsonify({
        "input_exists":  HTR_GROUPED_CSV.exists(),
        "output_exists": CORRECTED_CSV.exists(),
        "input_mtime":   _fmt(HTR_GROUPED_CSV),
        "output_mtime":  _fmt(CORRECTED_CSV),
        "stats":         _corrected_stats(),
    })


@app.route("/step05/run", methods=["POST"])
def step05_run():
    max_rows_str = request.form.get("max_rows", "").strip()
    max_rows = int(max_rows_str) if max_rows_str.isdigit() else None

    if not HTR_GROUPED_CSV.exists():
        return jsonify({"error": "Fichier extracted_text.csv introuvable — lancez d'abord l'HTR step04."}), 400

    def _job():
        from source.step05_correct.correct import run_correction
        if max_rows:
            print(f"Limite : {max_rows} spécimens")
        results = run_correction(
            input_csv=HTR_GROUPED_CSV,
            output_csv=CORRECTED_CSV,
            max_rows=max_rows,
        )
        n_err = sum(
            1 for r in results
            if any(r.get(c, "") == "ERREUR" for c in STATUT_COLS)
        )
        print(f"\nTerminé — {len(results)} spécimens, {n_err} avec au moins une erreur")

    return jsonify({"job_id": _start_job(_job)})
