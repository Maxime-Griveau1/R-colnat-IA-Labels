# IA Labels — Pipeline de transcription automatique d'étiquettes de specimens naturels

Outil de traitement d'images et de transcription automatique des étiquettes de specimens des collections naturalistes françaises ([https://récolnat.fr]). Le pipeline combine détection d'objets (YOLO), reconnaissance de texte manuscrit (HTR) et post-traitement pour produire des données structurées au standard [Darwin Core](https://dwc.tdwg.org/).

---

## Aperçu du pipeline

```
Images sources (scans haute résolution)
        │
        ▼
┌─────────────────────────────────┐
│  Étape 0 — Prétraitement        │  Pillow : CLAHE, resize, letterbox
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Herbier           Entomologie
        │                 │
        └────────┬─────────┘
                 ▼
┌─────────────────────────────────┐
│  Étape 1 — Classification       │  YOLO classify : herbier / entomologie
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  Étape 2 — Détection des zones  │  YOLO detect : collecte, détermination,
│            (Label Studio)       │  tampon, code-barres, notes…
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  Étape 3 — Détection des champs │  YOLO detect : date, localité,
│            (Label Studio)       │  collecteur, déterminateur…
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  Étape 4 — HTR                  │  Kraken / GLM-OCR : transcription
│                                 │  scripts 17e–21e siècle
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  Étape 5 — Post-traitement      │  pandas : correction OCR, normalisation
│                                 │  dates, GBIF, Nominatim, Darwin Core
└────────────────┬────────────────┘
                 ▼
         results.csv (Darwin Core)
```

---

## Fonctionnalités

- **Interface web** (Flask) pour piloter chaque étape sans ligne de commande
- **Intégration Label Studio** : création automatique de projets, import des images, export des annotations au format YOLO
- **Déterminations multiples** : chaque étiquette de détermination est une zone indépendante ; les déterminations sont numérotées chronologiquement en sortie
- **Démo interactive** : pipeline complet sur une image importée depuis l'ordinateur, avec visualisation étape par étape

---

## Stack technique

| Composant | Rôle |
|-----------|------|
| [Flask](https://flask.palletsprojects.com/) | Interface web et API |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | Détection et classification |
| [Label Studio](https://labelstud.io/) | Annotation des images |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) GLM-OCR | Reconnaissance de texte manuscrit (HTR) |
| [spaCy](https://spacy.io/) `fr_core_news_md` | NER pour normalisation des noms |
| [Pillow](https://python-pillow.org/) | Prétraitement des images |
| [pandas](https://pandas.pydata.org/) | Post-traitement et export CSV |
| Docker Compose | Orchestration Flask + Label Studio |

---

## Prérequis

- Docker et Docker Compose
- GPU recommandé pour l'entraînement YOLO et le fonctionne du modèle GLM-OCR (fonctionne aussi en CPU)
- ~20 Go d'espace disque pour les modèles et données

---

## Installation et démarrage

### 1. Cloner le dépôt

```bash
git clone
cd IA_labels
```

### 2. Configurer l'environnement

```bash
cp APP/.env.example APP/.env
# Éditer APP/.env : renseigner LABEL_STUDIO_API_KEY après le premier démarrage
```

### 3. Lancer avec Docker Compose

```bash
docker compose up -d
```
NB. Le premier docker compose up -d prend un certain temps (build des images) et demande une vingtaine de Go d'espace disque. 

- Application Flask : [http://localhost:5000](http://localhost:5000)
- Label Studio : [http://localhost:8080](http://localhost:8080)

### 4. Premier démarrage de Label Studio

1. Ouvrir [http://localhost:8080](http://localhost:8080)
2. Créer un compte (l'email et le mot de passe sont à définir dans `docker-compose.yml`)
3. Récupérer le token (legacy) d'API dans **Settings → Access Token** (s'il n'est pas activer, activer le legacy token dans la partie "organisation" de label-studio)
4. Le renseigner dans `APP/.env` : `LABEL_STUDIO_API_KEY=<token>`
5. Redémarrer Flask : `docker compose restart flask`

---

## Structure du projet

```
IA_labels/
├── APP/
│   ├── app/
│   │   ├── app.py                         # Flask
│   │   ├── routes/
│   │   │   ├── generales.py               # SSE, prétraitement, classification
│   │   │   ├── annotation.py              # Intégration Label Studio (étape 2)
│   │   │   ├── fields.py                  # Détection des champs (étape 3)
│   │   │   ├── htr.py                     # HTR — transcription (étape 4)
│   │   │   ├── correct.py                 # Post-traitement (étape 5)
│   │   │   ├── resultats.py               # Visualisation des résultats
│   │   │   └── demo.py                    # Démo interactive
│   │   └── templates/
│   ├── source/
│   │   ├── preprocessImages.py            # Prétraitement Pillow
│   │   ├── step01_label_classification/   # YOLO classify
│   │   ├── step02_zone_annotation/        # YOLO detect zones + Label Studio
│   │   ├── step03_fields_detection/       # YOLO detect champs + Label Studio
│   │   ├── step04_extract_text/           # HTR (Kraken)
│   │   └── step05_correct/               # Post-traitement pandas + spaCy
│   ├── Datas/                             # Données (non versionnées)
│   ├── Dockerfile
│   └── requirements.txt
└── docker-compose.yml
```

### Données (non versionnées)

```
APP/Datas/
  in/
    00_source_images/
      herbarium/jpeg/       # scans herbier
      entomology/           # scans entomologie
  out/
    step00_preprocessing/   # images redimensionnées
    step01_classification/  # dataset + modèle YOLO classify
    step02_zone_detection/  # dataset + modèle YOLO detect zones
    step03_fields_detection/# dataset + modèle YOLO detect champs
    step04_htr/             # textes extraits (CSV)
    step05_correct/         # résultats corrigés (CSV)
```

---

## Classes de zones détectées (étape 2)

| # | Classe | Description |
|---|--------|-------------|
| 0 | `collecte` | Étiquette de collecte (date, localité, collecteur) |
| 1 | `determination` | Étiquette de détermination — **une boîte par détermination** |
| 2 | `tampon` | Cachet / tampon |
| 3 | `numero_inventaire` | Code-barres, QR code, numéro MNHN |
| 4 | `graines` | Sachet de graines (herbier) |
| 5 | `notes` | Annotations manuscrites libres |
| 6 | `dessin` | Illustration botanique |
| 7 | `specimen` | Spécimen lui-même |
| 8 | `logo` | Logo d'institution |

---

## Format de sortie (Darwin Core)

Le fichier `results.csv` produit une colonne par terme DwC. Les déterminations multiples sont numérotées :

| Colonne | Exemple |
|---------|---------|
| `scientificName` | *Quercus robur* L. |
| `recordedBy` | Bonpland, A. |
| `eventDate` | 1802-07-15 |
| `decimalLatitude` | 48.8566 |
| `determination_1` | *Quercus robur* |
| `determinateur_1` | Bonpland |
| `date_determination_1` | 1803 |
| `determination_2` | *Quercus pedunculata* |
| … | … |

---

## Variables d'environnement (`APP/.env`)

| Variable | Description |
|----------|-------------|
| `DEBUG` | Mode debug Flask (`True` / `False`) |
| `SECRET_KEY` | Clé secrète Flask |
| `LABEL_STUDIO_URL` | URL de l'instance Label Studio |
| `LABEL_STUDIO_API_KEY` | Token d'accès LS (Settings → Access Token) |
| `LABEL_STUDIO_PROJECT_ID` | Rempli automatiquement à la création du projet |
| `LABEL_STUDIO_FIELDS_PROJECT_ID` | Idem pour le projet champs (étape 3) |
| `HF_TOKEN` | Token Hugging Face (modèles HTR) |

---

## Développement

Le volume `./APP/app:/app/app` dans `docker-compose.yml` permet le rechargement automatique de Flask en mode `DEBUG=True` sans rebuild :

```bash
# Modifier un fichier dans APP/app/ → Flask recharge automatiquement
docker compose up -d
```

Pour lancer les scripts de manière autonome (hors Docker) :

```bash
source env/bin/activate
cd APP

# Préparer le dataset de classification
python -m source.step01_label_classification.prepare_dataset

# Entraîner le classificateur
python -m source.step01_label_classification.train --epochs 50 --model yolo11s-cls.pt

# Lancer une inférence
python -m source.step01_label_classification.predict --input Datas/in/00_source_images
```

--

## Auteurs

Développé pour l'infrastructure de recherche Récolnat [https://récolnat.fr] par Maxime GRIVEAU.
