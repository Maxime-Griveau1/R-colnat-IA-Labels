"""
Chemins partagés entre les routes Flask et les scripts source.
Toutes les constantes sont relatives à BASE_DIR (APP/).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # APP/

# ── Images sources HD ────────────────────────────────────────────────────────
SOURCE_HD: dict[str, list[Path]] = {
    "herbarium":  [
        BASE_DIR / "Datas" / "in" / "00_source_images" / "herbarium" / "jpeg",
        BASE_DIR / "Datas" / "in" / "00_source_images" / "herbarium",
    ],
    "entomology": [
        BASE_DIR / "Datas" / "in" / "00_source_images" / "entomology",
    ],
}
PREPROCESS_DIRS: dict[str, Path] = {
    "herbarium":  BASE_DIR / "Datas" / "out" / "step00_preprocessing" / "herbarium",
    "entomology": BASE_DIR / "Datas" / "out" / "step00_preprocessing" / "entomology",
}

# ── Étape 1 : Classification ──────────────────────────────────────────────────
CLS_BEST_PT     = BASE_DIR / "Datas" / "out" / "step01_classification" / "models" / "run" / "weights" / "best.pt"
CLS_RESULTS_CSV = BASE_DIR / "Datas" / "out" / "step01_classification" / "models" / "run" / "results.csv"
CLS_DATASET_DIR = BASE_DIR / "Datas" / "out" / "step01_classification" / "dataset"
CLS_PREDICT_CSV = BASE_DIR / "Datas" / "out" / "step01_classification" / "predictions.csv"

# ── Étape 2 : Détection des zones ─────────────────────────────────────────────
DET_BEST_PT     = BASE_DIR / "Datas" / "out" / "step02_zone_detection" / "models" / "run" / "weights" / "best.pt"
DET_RESULTS_CSV = BASE_DIR / "Datas" / "out" / "step02_zone_detection" / "models" / "run" / "results.csv"
DET_DATASET_DIR = BASE_DIR / "Datas" / "out" / "step02_zone_detection" / "dataset"
DET_PREDICT_CSV = BASE_DIR / "Datas" / "out" / "step02_zone_detection" / "predictions.csv"
DET_PARTS_DIR   = BASE_DIR / "Datas" / "out" / "step02_zone_detection" / "parts"

# ── Étape 3 : Détection des champs de métadonnées ────────────────────────────
# Deux modèles distincts selon le type de zone (collecte / determination).
# Les crops des deux modèles alimentent ensuite le même pipeline HTR (step04).
_STEP03_ZONES = ("collecte", "determination")
_STEP03_BASE  = {z: BASE_DIR / "Datas" / "out" / f"step03_{z}" for z in _STEP03_ZONES}

FIELDS_BEST_PT     = {z: _STEP03_BASE[z] / "models" / "run" / "weights" / "best.pt" for z in _STEP03_ZONES}
FIELDS_DATASET_DIR = {z: _STEP03_BASE[z] / "dataset"                                  for z in _STEP03_ZONES}
FIELDS_PREDICT_CSV = {z: _STEP03_BASE[z] / "predictions.csv"                          for z in _STEP03_ZONES}
FIELDS_PARTS_DIR   = {z: _STEP03_BASE[z] / "parts"                                    for z in _STEP03_ZONES}

# ── Étape 4 : Extraction HTR ──────────────────────────────────────────────────
HTR_OUT_DIR           = BASE_DIR / "Datas" / "out" / "step04_extract_text"
HTR_PREPROCESSED_DIR  = HTR_OUT_DIR / "preprocessed"   # crops après prétraitement image
HTR_RAW_CSV           = HTR_OUT_DIR / "transcriptions_raw.csv"
HTR_GROUPED_CSV       = HTR_OUT_DIR / "extracted_text.csv"

# ── Étape 5 : Post-traitement / correction ────────────────────────────────────
CORRECT_OUT_DIR    = BASE_DIR / "Datas" / "out" / "step05_correct"
CORRECTED_CSV      = CORRECT_OUT_DIR / "corrected.csv"
