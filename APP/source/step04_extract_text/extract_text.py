"""
Extraction HTR des champs de métadonnées via GLM-OCR (zai-org/GLM-OCR).

Usage :
  python -m source.step04_extract_text.extract_text
         [--parts-dir <dossier>] [--model <model_id>]
         [--batch-size 4] [--device cpu|cuda|auto]

Entrée : crops par classe depuis step03 (FIELDS_PARTS_DIR/<classe>/<stem>_<classe>_<idx>.jpg)
Sorties :
  - step04_extract_text/transcriptions_raw.csv  (une ligne par crop)
  - step04_extract_text/extracted_text.csv      (une ligne par spécimen, une colonne par champ)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from PIL import Image

from source.paths import FIELDS_PARTS_DIR, HTR_OUT_DIR, HTR_RAW_CSV, HTR_GROUPED_CSV

# Classes de champs simples (une valeur par spécimen)
FIELD_CLASSES_SINGLE = [
    "collecteur",
    "date_collecte",
    "localite",
    "numero_inventaire",
]

# Classes liées aux déterminations (peuvent apparaître N fois par spécimen)
FIELD_CLASSES_DET = ["determination", "determinateur", "date_determination"]

FIELD_CLASSES = FIELD_CLASSES_SINGLE + FIELD_CLASSES_DET

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

DEFAULT_MODEL = "zai-org/GLM-OCR"
DEFAULT_BATCH = 4

# Prompt adapté au contexte herbier/entomologie MNHN (français + latin scientifique)
HTR_PROMPT = (
    "Transcribe the handwritten or printed text exactly as it appears. "
    "Use only Latin characters (French or English). "
    "Do not translate. Do not add any explanation."
)

# Caractères autorisés : ASCII imprimable + diacritiques français + latin scientifique
_RE_ALLOWED = re.compile(
    r"[^\x20-\x7EàâäéèêëîïôùûüÿæœçÀÂÄÉÈÊËÎÏÔÙÛÜŸÆŒÇ°’‘]"
)
# Détection de boucles répétitives (même motif de ≤ 6 chars répété ≥ 8 fois)
_RE_LOOP = re.compile(r"(.{1,6})\1{7,}")


def _clean_htr(text: str) -> str:
    """
    Nettoie la sortie HTR :
    1. Supprime les caractères non-latins (CJK, hébreu, arabe, cyrillique…)
    2. Détecte et vide les hallucinations répétitives (ex: -1.1.1.1.1…)
    3. Normalise les espaces résiduels.
    """
    if not text:
        return text
    # Hallucination répétitive → texte vide (sera ignoré dans le CSV groupé)
    if _RE_LOOP.search(text):
        return ""
    # Suppression des caractères hors alphabet latin + français
    cleaned = _RE_ALLOWED.sub("", text)
    # Nettoyage des espaces multiples
    return re.sub(r" {2,}", " ", cleaned).strip()


def _specimen_id(stem: str, class_name: str) -> str:
    """Extrait l'identifiant spécimen depuis le stem du nom de fichier.

    Exemple : MNHN-EC-EC4906_localite_01 → MNHN-EC-EC4906
    """
    marker = f"_{class_name}_"
    idx = stem.rfind(marker)
    return stem[:idx] if idx != -1 else stem


def collect_crops(parts_dir: Path | dict) -> list[tuple[Path, str]]:
    """Retourne [(image_path, class_name)] pour tous les crops disponibles.

    parts_dir peut être un Path unique ou un dict {zone: Path} (architecture step03 multi-zones).
    """
    dirs: list[Path] = list(parts_dir.values()) if isinstance(parts_dir, dict) else [parts_dir]
    crops = []
    for base in dirs:
        if not base.exists():
            continue
        for cls_dir in sorted(base.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in EXTENSIONS:
                    crops.append((p, cls_name))
    return crops


def run_htr(
    parts_dir: Path | dict = FIELDS_PARTS_DIR,
    model_id: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
    device: str = "auto",
    max_crops: int | None = None,
    raw_csv: Path = HTR_RAW_CSV,
    grouped_csv: Path = HTR_GROUPED_CSV,
    preprocess_kwargs: dict | None = None,
) -> list[dict]:
    import os
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    # Token HuggingFace — évite les limitations de débit sur les téléchargements
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    else:
        print("[AVERT] HF_TOKEN non défini — téléchargements HuggingFace limités en débit.")

    preprocess_kwargs = preprocess_kwargs or {}
    use_preprocess = bool(preprocess_kwargs)
    if use_preprocess:
        from source.step04_extract_text.preprocess import preprocess_crop
        print(f"Prétraitement activé : {preprocess_kwargs}")

    print(f"Chargement du modèle HTR : {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    try:
        import accelerate  # noqa: F401
        device_map = "auto" if device == "auto" else {"": device}
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype="auto", device_map=device_map,
        )
    except ImportError:
        # accelerate absent → chargement simple puis déplacement manuel
        resolved = "cpu"
        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ("cpu", "cuda"):
            resolved = device
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype="auto",
        ).to(resolved)

    model.eval()

    crops = collect_crops(parts_dir)
    if not crops:
        print(f"[ERREUR] Aucun crop trouvé dans {parts_dir}")
        return []

    if max_crops is not None:
        crops = crops[:max_crops]

    FIELDNAMES = ["filename", "specimen_id", "field_class", "text", "confidence"]

    # Reprise : sauter les crops déjà traités si le CSV existe
    already_done: set[str] = set()
    if raw_csv.exists():
        with open(raw_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                already_done.add(r.get("filename", ""))
        if already_done:
            print(f"Reprise : {len(already_done)} crops déjà traités, ignorés.")

    crops = [c for c in crops if c[0].name not in already_done]
    total_remaining = len(crops)
    total_done = len(already_done)
    total_all = total_done + total_remaining
    print(f"{total_remaining} crops restants ({total_done} déjà faits / {total_all} total)")

    HTR_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ouvrir le CSV en append (création avec header si nouveau, sinon ajout)
    write_header = not raw_csv.exists() or total_done == 0
    csv_file = open(raw_csv, "a" if not write_header else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    raw_rows: list[dict] = []
    try:
        for i, (img_path, cls_name) in enumerate(crops):
            if use_preprocess:
                pil_img = preprocess_crop(img_path, **preprocess_kwargs).convert("RGB")
            else:
                pil_img = Image.open(img_path).convert("RGB")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": HTR_PROMPT},
                ],
            }]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)

            text = _clean_htr(processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip())

            specimen_id = _specimen_id(img_path.stem, cls_name)
            row = {
                "filename":    img_path.name,
                "specimen_id": specimen_id,
                "field_class": cls_name,
                "text":        text,
                "confidence":  1.0,
            }
            writer.writerow(row)
            csv_file.flush()  # écriture immédiate sur disque
            raw_rows.append(row)

            done = i + 1
            if done % batch_size == 0 or done == total_remaining:
                print(f"  {total_done + done}/{total_all}")
    finally:
        csv_file.close()

    # Relire tout le CSV (y compris les lignes des runs précédents) pour le groupé
    all_rows = _read_all_rows(raw_csv, FIELDNAMES)
    grouped = _group_by_specimen(all_rows)
    _write_grouped(grouped, grouped_csv)

    print(f"CSV brut    → {raw_csv}")
    print(f"CSV groupé  → {grouped_csv}")
    return all_rows


def _read_all_rows(raw_csv: Path, fieldnames: list[str]) -> list[dict]:
    if not raw_csv.exists():
        return []
    with open(raw_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_by_specimen(raw_rows: list[dict]) -> dict[str, dict]:
    """
    Regroupe les transcriptions par spécimen.

    - Champs simples (collecteur, date_collecte, localite, numero_inventaire) :
      première valeur retenue, les suivantes ignorées.
    - Champs de détermination (determination, determinateur, date_determination) :
      les crops sont issus de zones step02 `determination` distinctes.
      Chaque zone produit un triplet numéroté : determination_1, determinateur_1,
      date_determination_1, determination_2, …

    Le filename encode l'index de zone : <stem>_<classe>_<zone_idx>_<crop_idx>.jpg
    On groupe d'abord par zone_idx pour reconstituer les triplets.
    """
    # Accumulation brute par spécimen
    raw: dict[str, dict[str, list[tuple[str, str]]]] = {}
    # raw[sid][cls] = [(zone_idx, text), ...]
    for r in raw_rows:
        sid  = r["specimen_id"]
        cls  = r["field_class"]
        txt  = r["text"]
        name = r["filename"]  # <stem>_<cls>_<zone>_<crop>.jpg ou <stem>_<cls>_<crop>.jpg
        if not txt:
            continue
        # Extraire zone_idx depuis le nom de fichier (avant-dernier segment numérique)
        parts = Path(name).stem.split("_")
        nums = [p for p in parts if p.isdigit()]
        zone_idx = nums[-2] if len(nums) >= 2 else nums[-1] if nums else "0"
        raw.setdefault(sid, {}).setdefault(cls, []).append((zone_idx, txt))

    result: dict[str, dict] = {}
    for sid, cls_map in raw.items():
        row: dict[str, str] = {}

        # Champs simples — première occurrence
        for cls in FIELD_CLASSES_SINGLE:
            vals = cls_map.get(cls, [])
            if vals:
                row[cls] = vals[0][1]

        # Champs de détermination — groupés par zone_idx, numérotés
        # Collecter tous les zone_idx présents dans les classes det
        zone_texts: dict[str, dict[str, str]] = {}
        for cls in FIELD_CLASSES_DET:
            for zone_idx, txt in cls_map.get(cls, []):
                zone_texts.setdefault(zone_idx, {})[cls] = txt

        for n, zone_idx in enumerate(sorted(zone_texts.keys()), start=1):
            triplet = zone_texts[zone_idx]
            for cls in FIELD_CLASSES_DET:
                if cls in triplet:
                    row[f"{cls}_{n}"] = triplet[cls]

        row["_n_determinations"] = str(len(zone_texts)) if zone_texts else "0"
        result[sid] = row

    return result


def _det_fieldnames(grouped: dict[str, dict]) -> list[str]:
    """Construit la liste de colonnes det_N en fonction du max de déterminations."""
    max_n = max(
        (int(r.get("_n_determinations", "0")) for r in grouped.values()),
        default=0,
    )
    cols = []
    for n in range(1, max_n + 1):
        for cls in FIELD_CLASSES_DET:
            cols.append(f"{cls}_{n}")
    return cols


def _write_grouped(grouped: dict[str, dict], path: Path) -> None:
    det_cols  = _det_fieldnames(grouped)
    fieldnames = ["specimen_id"] + FIELD_CLASSES_SINGLE + det_cols + ["_n_determinations"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for sid, fields in sorted(grouped.items()):
            row = {"specimen_id": sid}
            row.update(fields)
            writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parts-dir",  default=None, help="Dossier parts (défaut: FIELDS_PARTS_DIR de paths.py)")
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--device",     default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-crops",  type=int, default=None, help="Limite le nombre de crops à traiter")
    return p.parse_args()


def main():
    args = parse_args()
    parts_dir = Path(args.parts_dir) if args.parts_dir else FIELDS_PARTS_DIR

    run_htr(
        parts_dir=parts_dir,
        model_id=args.model,
        batch_size=args.batch_size,
        device=args.device,
        max_crops=args.max_crops,
    )


if __name__ == "__main__":
    main()
