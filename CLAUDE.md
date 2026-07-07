# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Flask-based ML pipeline tool for the MNHN (Muséum national d'Histoire naturelle) to process and annotate specimen images (herbarium sheets, entomology).

The workflow is the following:
## Approach 1 — Structured pipeline (YOLO + Kraken + python/pandas)

### Step 1 — `process_image.py`
**Tool: Pillow**
Pre-process input images to improve model recognition: adjust colours, contrast, crop, remove noise tied to label background.
Input: raw scans
Output: `Data/in/01_preprocessed/`

### Step 2 — `image_type.py`
**Tool: YOLO**
Classify each image by collection type (herbarium, entomology, etc.).
Output: routes image to `Data/in/02_entomology/` or `Data/in/02_herbarium/`

### Step 3 — `parts.py` (shared, both collection types)
**Tool: YOLO**
Detect and segment the physical parts of the label image: stickers, logos, barcodes, stamps, determination labels, annotation zones, etc.
Applies to both herbarium and entomology — zone detection is now unified.
Output: `Data/out/03_parts/`

Key design: `etiquette_determination` is a dedicated zone class for determination labels.
Each determination label (specimen can have several) gets its own bounding box → step 4 extracts
(determination, determinateur, date_determination) within that zone, so multi-determination
specimens are handled without ambiguity.

### Step 4 — `fields.py` (×2, one per collection type)
**Tool: YOLO**
Within each detected part, identify individual data fields (date, collector, locality, determination, etc.).
Output: `Data/out/04_entomology_fields/` and `Data/out/04_herbarium_fields/`

Note : this is used for both the herbarium and the entomology (the workflow resumes here)

### Step 5 — `extract_text.py`
**Tool: Kraken or another HTR Tool**
OCR extraction using a model generalised over scripts from the 17th to 21st century (handles wide paleographic variation). lassifies text zones for chronological ordering.C
Input: field-segmented images
Output: `Data/out/05_extracted_text.csv`

### Step 6 — `correct.py`
**Tool: python/pandas**
Post-processing: rule-based script + dictionaries to correct OCR errors, normalise field values, standardise name formats, etc.
Input: `05_extracted_text.csv`
Output: `06_results.csv`

### Final output — `results.csv`
Structured CSV. The Darwin Core (`DwC`) field is **one column per DwC term**. Multiple determinations are supported. Column order is fixed.

## Running the App

```bash
# Activate the virtualenv first
source env/bin/activate

# Run the Flask app
cd APP
python run.py
```

The app runs at `http://localhost:5000`. Configure via `APP/.env`.

## Data Directory Layout

All data lives under `APP/Datas/` (not tracked in git):

```
APP/Datas/
  in/
    00_source_images/
      herbarium/jpeg/    # source herbarium JPEGs
      entomology/        # source entomology images
  out/
    step00_preprocessing/herbarium|entomology/  # resized 1024×1024 JPEGs
    step01_classification/
      dataset/train|val|test/<class>/           # YOLO classify dataset
      models/run/weights/best.pt                # trained classifier
      predictions.csv                           # classification results
    step02_zone_detection/
      dataset/train|val|test/images|labels/     # YOLO detect dataset
      dataset/data.yaml
      models/run/weights/best.pt                # trained detector
```

## Flask App Architecture (`APP/app/`)

- `app.py`: Flask app factory; imports routes after creating the app instance.
- `config.py`: Loads config from `APP/.env` via `dotenv`.
- `routes/generales.py`: Core infrastructure — SSE job streaming, step 0 preprocessing routes, step 1 classification routes. The `_start_job(fn)` pattern runs any callable in a daemon thread and streams its stdout/stderr to the client via Server-Sent Events at `GET /stream/<job_id>`.
- `routes/annotation.py`: Step 2 routes — Label Studio integration, image serving at `/serve-image/<collection>/<filename>`.
- `routes/erreurs.py`: Error page handlers.

## SSE Job Pattern

All long-running operations (training, preprocessing, export) use the same pattern:
1. POST endpoint returns `{"job_id": "<id>"}`.
2. Client polls `GET /stream/<job_id>` for SSE lines.
3. `__DONE__` sentinel signals completion.
4. `_QueueWriter` strips ANSI codes and handles `\r` (tqdm progress) before emitting lines.

## Environment Variables (APP/.env)

| Key | Purpose |
|-----|---------|
| `DEBUG` | Flask debug mode |
| `SECRET_KEY` | Flask session key |
| `SQLALCHEMY_DATABASE_URI` | DB URI (unused currently) |
| `LABEL_STUDIO_URL` | Label Studio server URL (default: `http://localhost:8080`) |
| `LABEL_STUDIO_API_KEY` | LS API token — use the "Access Token" from LS Settings, not a JWT refresh token |
| `LABEL_STUDIO_PROJECT_ID` | Auto-set when project is created |

## Zone Classes (step 2 annotation / step 3 detection)

Defined in `label_studio_manager.py` as `ZONE_CLASSES` (order = YOLO class index).
Applies to **both** herbarium and entomology collections.

| # | Class | Usage |
|---|-------|-------|
| 0 | `collecte` | Collection label (date, locality, collector) |
| 1 | `determination` | Determination label — **one box per determination**; step 4 extracts determination + determinateur + date_determination within that zone |
| 2 | `tampon` | Rubber stamp |
| 3 | `numero_inventaire` | Barcode, QR code or printed MNHN inventory label |
| 4 | `graines` | Seed packet (herbarium only) |
| 5 | `notes` | Free handwritten annotations |
| 6 | `dessin` | Botanical drawing / illustration |
| 7 | `specimen` | Specimen itself (insect, plant fragment) |
| 8 | `logo` | Institution logo |

**Multiple determinations**: each `determination` zone is numbered chronologically in step 5 by `date_determination`. If date is absent the status is `INCERTAIN`. Output columns: `determination_1`, `determinateur_1`, `date_determination_1`, `numero_determination_1`, …`_N`.

## Docker Compose

The app runs via `docker compose up` (port 5000). **Do not modify or remove the volume mounts** in `docker-compose.yml`:

```yaml
volumes:
  - ./APP/Datas:/app/Datas      # données persistantes (ne pas supprimer)
  - ./APP/app:/app/app          # hot-reload Flask en mode DEBUG (ne pas supprimer)
```

Le second mount (`./APP/app`) est indispensable pour que le rechargement automatique de Flask fonctionne en développement — sans lui, chaque modification de code nécessite un `docker compose build`.

## Running Scripts Standalone

```bash
# Prepare classification dataset
python -m source.step01_label_classification.prepare_dataset

# Train classifier
python -m source.step01_label_classification.train --epochs 50 --model yolo11s-cls.pt

# Run inference
python -m source.step01_label_classification.predict --input Datas/in/00_source_images
```
