# IA Labels — Pipeline d'annotation des spécimens MNHN

Outil Flask de traitement et d'annotation automatique des images de spécimens du Muséum national d'Histoire naturelle (MNHN). Il guide un opérateur à travers un pipeline ML complet : du scan brut jusqu'à la détection des champs de métadonnées sur les étiquettes.

Collections supportées : **herbier** et **entomologie**.

---

## Architecture générale

```
APP/
├── app/                    # Application Flask
│   ├── app.py              # Factory Flask
│   ├── config.py           # Chargement de APP/.env
│   └── routes/
│       ├── generales.py    # Étapes 0 et 1 + infrastructure SSE
│       ├── annotation.py   # Étape 2 — détection des zones
│       ├── fields.py       # Étape 3 — détection des champs
│       └── erreurs.py      # Gestionnaires d'erreurs
├── source/
│   ├── paths.py            # Chemins partagés (source → routes)
│   ├── preprocessImages.py # Redimensionnement 1024×1024
│   ├── tiffTojpeg.py       # Conversion TIFF → JPEG
│   ├── step01_label_classification/
│   │   ├── prepare_dataset.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── step02_zone_annotation/
│   │   ├── label_studio_manager.py
│   │   ├── export_to_yolo.py
│   │   └── predict.py
│   └── step03_fields_detection/
│       ├── label_studio_manager.py
│       ├── export_to_yolo.py
│       └── predict.py
├── Datas/                  # Données (non versionné, voir .gitignore)
├── run.py                  # Point d'entrée Flask
└── .env                    # Configuration locale (non versionné)
```

---

## Pipeline en 4 étapes

### Étape 0 — Prétraitement

Redimensionne les scans bruts en JPEG 1024×1024 pour normaliser les entrées YOLO.

- **Entrée :** `Datas/in/00_source_images/herbarium/jpeg/` et `Datas/in/00_source_images/entomology/`
- **Sortie :** `Datas/out/step00_preprocessing/herbarium/` et `.../entomology/`

### Étape 1 — Classification du type d'image (YOLO classify)

Identifie la collection de chaque image (`herbarium`, `entomology`, etc.).

- **Modèles disponibles :** `yolo11n-cls.pt`, `yolo11s-cls.pt`, `yolo11m-cls.pt`
- **Sortie :** `Datas/out/step01_classification/predictions.csv`

Sous-étapes gérées par l'interface :
1. Préparer le dataset (split train/val/test par classe)
2. Entraîner le modèle YOLO classification
3. Lancer les prédictions sur les images sources

### Étape 2 — Détection des zones (YOLO detect + Label Studio)

Détecte et segmente les parties physiques des étiquettes : étiquettes, codes-barres, tampons, annotations manuscrites, spécimens, logos.

Classes (`ZONE_CLASSES`, index = classe YOLO) :
`etiquette`, `code_barre`, `tampon`, `annotation_ms`, `specimen`, `logo`

Workflow d'annotation via Label Studio :
1. Configurer l'URL et la clé API Label Studio
2. Créer le projet d'annotation
3. Importer les images herbier (filtrées par la classification étape 1)
4. Annoter dans Label Studio
5. Exporter au format YOLO
6. Entraîner le modèle de détection
7. Lancer les prédictions (+ export des crops par classe)

- **Sortie modèle :** `Datas/out/step02_zone_detection/models/run/weights/best.pt`
- **Sortie prédictions :** `Datas/out/step02_zone_detection/predictions.csv`
- **Crops par classe :** `Datas/out/step02_zone_detection/parts/<classe>/`

### Étape 3 — Détection des champs de métadonnées (YOLO detect + Label Studio)

Identifie les champs individuels au sein de chaque zone (date, collecteur, localité, détermination, etc.).

- **Herbier :** travaille sur les crops `etiquette` et `code_barre` produits à l'étape 2
- **Entomologie :** travaille sur les images prétraitées (étape 0), filtrées par la classification (étape 1)

Workflow identique à l'étape 2 (annotation → export YOLO → entraînement → prédiction).

- **Sortie modèle :** `Datas/out/step03_fields_detection/models/run/weights/best.pt`
- **Sortie prédictions :** `Datas/out/step03_fields_detection/predictions.csv`

---

## Installation

**Prérequis :** Python 3.10+, [Label Studio](https://labelstud.io/) (instance locale ou distante)

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd IA_labels/APP

# Créer et activer le virtualenv
python -m venv env
source env/bin/activate      # Linux / macOS
# env\Scripts\activate       # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration

Créer `APP/.env` :

```env
DEBUG=true
SECRET_KEY=une-cle-secrete-aleatoire
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=<token-access-label-studio>
# LABEL_STUDIO_PROJECT_ID est écrit automatiquement par l'app
```

> La clé API Label Studio se trouve dans **LS Settings → Access Token**. Ne pas utiliser un token JWT.

---

## Lancement

```bash
source env/bin/activate
cd APP
python run.py
```

L'interface est accessible à `http://localhost:5000`.

---

## Utilisation en ligne de commande

Les scripts peuvent aussi être exécutés indépendamment :

```bash
# Préparer le dataset de classification
python -m source.step01_label_classification.prepare_dataset

# Entraîner le classifieur (50 epochs, modèle small)
python -m source.step01_label_classification.train --epochs 50 --model yolo11s-cls.pt

# Lancer les prédictions de classification
python -m source.step01_label_classification.predict --input Datas/in/00_source_images
```

---

## Flux de données résumé

```
Scans bruts
    │  Étape 0 — resize 1024×1024
    ▼
Images prétraitées
    │  Étape 1 — classification YOLO (herbarium / entomologie)
    ▼
predictions.csv
    │  Étape 2 — détection des zones (annotation Label Studio → YOLO detect)
    ▼
Crops par zone (étiquettes, codes-barres…)
    │  Étape 3 — détection des champs (annotation Label Studio → YOLO detect)
    ▼
Champs de métadonnées localisés
```

---

## Architecture SSE (jobs longs)

Toutes les opérations longues (entraînement, prétraitement, export) tournent dans des threads daemon. La progression est streamée en temps réel vers le navigateur via Server-Sent Events :

1. `POST /step0X/<action>` → retourne `{"job_id": "<id>"}`
2. Le client s'abonne à `GET /stream/<job_id>`
3. `__DONE__` signale la fin du job

Le `_QueueWriter` nettoie les codes ANSI et gère les `\r` de tqdm pour n'émettre qu'une ligne propre par epoch.

---

## Variables d'environnement

| Variable | Description |
|---|---|
| `DEBUG` | Mode debug Flask |
| `SECRET_KEY` | Clé de session Flask |
| `SQLALCHEMY_DATABASE_URI` | URI base de données (non utilisé actuellement) |
| `LABEL_STUDIO_URL` | URL de l'instance Label Studio (défaut : `http://localhost:8080`) |
| `LABEL_STUDIO_API_KEY` | Token d'accès Label Studio |
| `LABEL_STUDIO_PROJECT_ID` | ID du projet zones (écrit automatiquement) |
| `LABEL_STUDIO_FIELDS_PROJECT_ID` | ID du projet champs (écrit automatiquement) |
