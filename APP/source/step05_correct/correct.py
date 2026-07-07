"""
Post-traitement des transcriptions HTR — étape 5.

Règles métier par colonne :
  collecteur        → copie brute, pas de transformation
  date_collecte     → normalisation ISO 8601 partiel (YYYY / YYYY-MM / YYYY-MM-DD)
  date_determination→ même normalisation + suppression préfixe DET
  determinateur     → copie brute
  determination     → validation GBIF Backbone (match exact uniquement)
  localite          → géocodage Nominatim (match exact → lat/lon)
  numero_inventaire → doit correspondre à specimen_id

Usage :
  python -m source.step05_correct.correct
         [--input <csv>] [--output <csv>] [--max-rows N]
"""

from __future__ import annotations

import argparse
import calendar
import csv
import re
import sys
import time
from pathlib import Path

import requests

from source.paths import HTR_GROUPED_CSV, CORRECTED_CSV, CORRECT_OUT_DIR

# ── spaCy NER (chargement paresseux) ──────────────────────────────────────────
# Modèle requis : fr_core_news_md  (pip install fr-core-news-md)
# Fallback accepté : fr_core_news_sm ou fr_core_news_lg si md absent.
# _nlp = None  → non chargé  |  False → échec permanent (évite O(N) retries)

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    if _nlp is False:
        raise RuntimeError(
            "Aucun modèle spaCy fr_core_news_* trouvé. "
            "Installez-en un : pip install fr-core-news-md"
        )
    import spacy as _spacy
    for model in ("fr_core_news_md", "fr_core_news_sm", "fr_core_news_lg"):
        try:
            _nlp = _spacy.load(model)
            print(f"[NER] Modèle spaCy chargé : {model}")
            return _nlp
        except OSError:
            continue
    _nlp = False  # type: ignore[assignment]  # sentinel : évite O(N) retries
    raise RuntimeError(
        "Aucun modèle spaCy fr_core_news_* trouvé. "
        "Installez-en un : pip install fr-core-news-md"
    )

# ── Constantes ────────────────────────────────────────────────────────────────

ANNEE_MIN, ANNEE_MAX = 1850, 2025

MOIS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
    "jan": 1, "fév": 2, "fev": 2, "mar": 3, "avr": 4,
    "jui": 6, "juil": 7, "aoû": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "déc": 12, "dec": 12,
}
MOIS_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# Chiffres romains pour les mois 
MOIS_ROMAINS = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
    "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12,
}
TOUS_MOIS = {**MOIS_FR, **MOIS_EN, **MOIS_ROMAINS}

# Regex de détection de plage d'années (ex: 1950-1951)
_RE_PLAGE_ANNEES = re.compile(r"\b(\d{4})\s*[-–]\s*(\d{4})\b")

FIELDNAMES_FIXED = [
    "specimen_id",
    "collecteur", "collecteur_brut",
    "date_collecte", "date_collecte_brut", "date_collecte_statut",
    "localite_brut",                       # texte brut HTR complet
    "localite",                            # lieux extraits par NER (envoyés à Nominatim)
    "localite_statut",
    "localite_remarques",                  # reste du texte (milieu, altitude, habitat…)
    "latitude", "longitude", "source_coordonnees",
    "numero_inventaire", "numero_inventaire_brut", "numero_inventaire_statut",
    "n_determinations",
]

def _det_fieldnames_for_n(n: int) -> list[str]:
    """Colonnes pour la détermination numéro n."""
    return [
        f"determination_{n}", f"determination_{n}_brut", f"determination_{n}_statut",
        f"determination_{n}_gbif_key", f"determination_{n}_gbif_rang",
        f"determinateur_{n}", f"determinateur_{n}_brut",
        f"date_determination_{n}", f"date_determination_{n}_brut", f"date_determination_{n}_statut",
        f"numero_determination_{n}", f"numero_determination_{n}_statut",
    ]

def _output_fieldnames(max_det: int) -> list[str]:
    cols = list(FIELDNAMES_FIXED)
    for n in range(1, max_det + 1):
        cols.extend(_det_fieldnames_for_n(n))
    return cols

GBIF_URL = "https://api.gbif.org/v1/species/match"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA = "MNHN-IA-Labels/1.0 (griveaumaxime@gmail.com)"


# ── Normalisation des dates ───────────────────────────────────────────────────

def _annee_valide(a: int) -> bool:
    return ANNEE_MIN <= a <= ANNEE_MAX


def _expand_annee(v: int) -> int | None:
    """
    Tente d'élargir un entier vers une année à 4 chiffres.
    - 4 chiffres déjà valides → retourne tel quel
    - 2 chiffres : >= 25 → 1900+v, < 25 → 2000+v  (pivot 2025)
    - Autre → None
    """
    if _annee_valide(v):
        return v
    if 0 <= v <= 99:
        candidate = 1900 + v if v >= 25 else 2000 + v
        if _annee_valide(candidate):
            return candidate
    return None


def _jours_du_mois(mois: int, annee: int) -> int:
    try:
        return calendar.monthrange(annee, mois)[1]
    except Exception:
        return 31


def _extraire_groupes(texte: str) -> list[str | int]:
    """
    Retourne les composantes de la date détectées : entiers numériques
    ou entiers déduits d'un nom de mois textuel (y compris romains).
    Exemple : "15 juin 1923" → [15, 6, 1923]
             "9-V-1974"     → [9, 5, 1974]
    """
    # Nettoyage des préfixes parasites (DET, Date, d.d, Leg, AN, : en tête)
    texte = re.sub(r"^\s*(det|date|d\.d\.?\.?|leg\.?|an\.?\b|:)\s*", "", texte, flags=re.IGNORECASE)
    # Nettoyage des suffixes parasites (M.D = déterminateur, Leg.)
    texte = re.sub(r"\bm\.?\s*d\.?\s*$", "", texte, flags=re.IGNORECASE)
    texte = re.sub(r"\bleg\.?\s*$", "",   texte, flags=re.IGNORECASE)

    # Cherche les tokens (mots ou chiffres)
    tokens = re.findall(r"[a-zA-Zàâäéèêëîïôùûüÿæœç]+|\d+", texte.lower())
    groupes: list[int] = []
    for tok in tokens:
        if tok.isdigit():
            groupes.append(int(tok))
        elif tok in TOUS_MOIS:
            # Inclut les noms français/anglais ET les chiffres romains (i…xii)
            groupes.append(TOUS_MOIS[tok])
    return groupes


def normaliser_date(texte: str) -> tuple[str, str]:
    """
    Normalise une date brute OCR.
    Retourne (date_iso_partiel, statut) où statut ∈ {OK, CORRIGÉ, AMBIGU, ERREUR, VIDE}.
    """
    if not texte or not texte.strip():
        return "", "VIDE"

    # Détection préalable des plages d'années (ex: 1950-1951, 1953–1954)
    m_plage = _RE_PLAGE_ANNEES.search(texte)
    if m_plage:
        a1, a2 = int(m_plage.group(1)), int(m_plage.group(2))
        if _annee_valide(a1):
            return str(a1), "AMBIGU"

    groupes = _extraire_groupes(texte.strip())

    if not groupes:
        return texte.strip(), "ERREUR"

    # ── 1 composante ──────────────────────────────────────────────────────────
    if len(groupes) == 1:
        v = groupes[0]
        # 4 chiffres : année
        if 1000 <= v <= 9999:
            if _annee_valide(v):
                return str(v), "OK"
            return str(v), "ERREUR"
        # 2 chiffres : ambiguïté mois / année abrégée
        if 1 <= v <= 99:
            if v <= 12:
                return texte.strip(), "AMBIGU"
            # Probable année abrégée (ex: 23 → 1923 ou 2023)
            candidate = 1900 + v if v >= 25 else 2000 + v
            if _annee_valide(candidate):
                return str(candidate), "AMBIGU"
        return texte.strip(), "ERREUR"

    # ── 2 composantes ─────────────────────────────────────────────────────────
    if len(groupes) == 2:
        a, b = groupes
        # Ambiguïté structurelle : les deux composantes ≤ 12 (ex: VIII.5 ou v.11)
        # On ne peut pas distinguer mois/mois, mois/année abrégée ou jour/mois sans année.
        if 1 <= a <= 12 and 1 <= b <= 12:
            return texte.strip(), "AMBIGU"
        ann_a = _expand_annee(a)
        ann_b = _expand_annee(b)
        # Second est une année, premier est un mois  (cas le plus fréquent : 7-76, 05 1923)
        if ann_b is not None and 1 <= a <= 12:
            statut = "OK" if _annee_valide(b) else "CORRIGÉ"
            return f"{ann_b:04d}-{a:02d}", statut
        # Premier est une année, second est un mois  (cas inversé)
        if ann_a is not None and 1 <= b <= 12:
            statut = "OK" if _annee_valide(a) else "CORRIGÉ"
            return f"{ann_a:04d}-{b:02d}", statut
        # Deux années ? aberrant
        if ann_a is not None and ann_b is not None:
            return texte.strip(), "ERREUR"
        # Ambiguïté résiduelle
        return texte.strip(), "AMBIGU"

    # ── 3 composantes (format français DD MM YYYY) ────────────────────────────
    if len(groupes) >= 3:
        # On prend les 3 premiers
        c1, c2, c3 = groupes[0], groupes[1], groupes[2]

        # Identifier l'année parmi les 3 — cherche d'abord en position 2 (la plus courante)
        # puis 0, puis 1 ; accepte aussi les années à 2 chiffres via _expand_annee
        def _candidats_annee():
            for pos, v in [(2, c3), (0, c1), (1, c2)]:
                expanded = _expand_annee(v)
                if expanded is not None:
                    return pos, expanded, v != expanded  # (position, année4, était_abrégée)
            return None, None, False

        pos_a, annee, etait_abregee = _candidats_annee()
        if annee is None:
            return texte.strip(), "ERREUR"

        reste = [v for i, v in enumerate([c1, c2, c3]) if i != pos_a]
        x, y = reste[0], reste[1]

        # Convention française DD/MM : quand l'année est en fin (pos 2) ou fin naturelle,
        # le premier résidu est le jour et le second le mois.
        # On tente d'abord cette interprétation, puis on essaie l'inverse si invalide.
        def _assigner_jour_mois(premier, second):
            """Retourne (jour, mois) ou None si les valeurs sont incohérentes."""
            if 1 <= premier <= 31 and 1 <= second <= 12:
                return premier, second
            return None

        assignation = _assigner_jour_mois(x, y)
        if assignation is None:
            # Tentative inversée
            assignation = _assigner_jour_mois(y, x)
        if assignation is None:
            return texte.strip(), "ERREUR"
        jour, mois = assignation

        if mois > 12 or jour > 31:
            return texte.strip(), "ERREUR"
        if jour > _jours_du_mois(mois, annee):
            return texte.strip(), "ERREUR"

        statut = "CORRIGÉ" if (pos_a != 2 or etait_abregee) else "OK"
        return f"{annee:04d}-{mois:02d}-{jour:02d}", statut

    return texte.strip(), "ERREUR"


# ── GBIF Backbone ─────────────────────────────────────────────────────────────

_gbif_cache: dict[str, dict] = {}


def valider_gbif(nom: str) -> dict:
    """
    Valide un nom taxonomique contre le GBIF Backbone.
    Retourne {nom, key, rang, statut}.
    """
    if not nom or not nom.strip():
        return {"nom": "", "key": "", "rang": "", "statut": "VIDE"}

    nom = nom.strip()
    if nom in _gbif_cache:
        return _gbif_cache[nom]

    try:
        time.sleep(0.2)
        r = requests.get(
            GBIF_URL,
            params={"name": nom, "strict": "true"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        result = {"nom": nom, "key": "", "rang": "", "statut": "ERREUR_API"}
        _gbif_cache[nom] = result
        return result

    match_type = data.get("matchType", "NONE")
    if match_type == "EXACT":
        result = {
            "nom": data.get("canonicalName", nom),
            "key": str(data.get("usageKey", "")),
            "rang": data.get("rank", ""),
            "statut": "OK",
        }
    else:
        result = {"nom": nom, "key": "", "rang": "", "statut": "NON_TROUVÉ_GBIF"}

    _gbif_cache[nom] = result
    return result


# ── Géocodage Nominatim ───────────────────────────────────────────────────────

# Référentiel des départements français (métropole + DOM)
DEPARTEMENTS_FR: dict[int, str] = {
    1: "Ain", 2: "Aisne", 3: "Allier", 4: "Alpes-de-Haute-Provence",
    5: "Hautes-Alpes", 6: "Alpes-Maritimes", 7: "Ardèche", 8: "Ardennes",
    9: "Ariège", 10: "Aube", 11: "Aude", 12: "Aveyron",
    13: "Bouches-du-Rhône", 14: "Calvados", 15: "Cantal", 16: "Charente",
    17: "Charente-Maritime", 18: "Cher", 19: "Corrèze", 20: "Corse",
    21: "Côte-d'Or", 22: "Côtes-d'Armor", 23: "Creuse", 24: "Dordogne",
    25: "Doubs", 26: "Drôme", 27: "Eure", 28: "Eure-et-Loir",
    29: "Finistère", 30: "Gard", 31: "Haute-Garonne", 32: "Gers",
    33: "Gironde", 34: "Hérault", 35: "Ille-et-Vilaine", 36: "Indre",
    37: "Indre-et-Loire", 38: "Isère", 39: "Jura", 40: "Landes",
    41: "Loir-et-Cher", 42: "Loire", 43: "Haute-Loire", 44: "Loire-Atlantique",
    45: "Loiret", 46: "Lot", 47: "Lot-et-Garonne", 48: "Lozère",
    49: "Maine-et-Loire", 50: "Manche", 51: "Marne", 52: "Haute-Marne",
    53: "Mayenne", 54: "Meurthe-et-Moselle", 55: "Meuse", 56: "Morbihan",
    57: "Moselle", 58: "Nièvre", 59: "Nord", 60: "Oise",
    61: "Orne", 62: "Pas-de-Calais", 63: "Puy-de-Dôme", 64: "Pyrénées-Atlantiques",
    65: "Hautes-Pyrénées", 66: "Pyrénées-Orientales", 67: "Bas-Rhin", 68: "Haut-Rhin",
    69: "Rhône", 70: "Haute-Saône", 71: "Saône-et-Loire", 72: "Sarthe",
    73: "Savoie", 74: "Haute-Savoie", 75: "Paris", 76: "Seine-Maritime",
    77: "Seine-et-Marne", 78: "Yvelines", 79: "Deux-Sèvres", 80: "Somme",
    81: "Tarn", 82: "Tarn-et-Garonne", 83: "Var", 84: "Vaucluse",
    85: "Vendée", 86: "Vienne", 87: "Haute-Vienne", 88: "Vosges",
    89: "Yonne", 90: "Territoire de Belfort", 91: "Essonne",
    92: "Hauts-de-Seine", 93: "Seine-Saint-Denis", 94: "Val-de-Marne",
    95: "Val-d'Oise",
    971: "Guadeloupe", 972: "Martinique", 973: "Guyane",
    974: "La Réunion", 976: "Mayotte",
}

# Pattern : nom de localité suivi d'un numéro 1-99 (ou 971-976)
_RE_DEPT = re.compile(
    r"^(.*?)\s+(97[1-6]|[1-9][0-9]?)\s*$",
    re.IGNORECASE,
)


def _parse_dept_fr(localite: str) -> tuple[str, int, str] | None:
    """
    Détecte le pattern '<ville> <numéro_dept>'.
    Retourne (ville, num_dept, nom_dept) ou None.
    """
    m = _RE_DEPT.match(localite.strip())
    if not m:
        return None
    ville = m.group(1).strip()
    num = int(m.group(2))
    if num not in DEPARTEMENTS_FR:
        return None
    return ville, num, DEPARTEMENTS_FR[num]


def _nominatim_request(query: str) -> list[dict]:
    """Envoie une requête Nominatim et retourne la liste de résultats (avec addressdetails)."""
    time.sleep(1.0)  # ToS Nominatim : max 1 req/s
    r = requests.get(
        NOMINATIM_URL,
        params={"q": query, "format": "json", "limit": 3, "addressdetails": 1},
        headers={"User-Agent": NOMINATIM_UA},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _est_en_france(res: dict) -> bool:
    """Vérifie que le résultat Nominatim est localisé en France."""
    address = res.get("address", {})
    return address.get("country_code", "").lower() == "fr"


def extraire_lieux_ner(texte: str) -> tuple[str, str]:
    """
    Extrait les entités géographiques (LOC, GPE) d'un texte brut via spaCy.

    Retourne (lieux, remarques) où :
      - lieux      : entités géo jointes par ", " → envoyées à Nominatim
      - remarques  : le reste du texte (habitat, altitude, description…)
    """
    if not texte or not texte.strip():
        return "", ""

    doc = _get_nlp()(texte)
    lieu_spans = [ent for ent in doc.ents if ent.label_ in ("LOC", "GPE")]

    if not lieu_spans:
        # Aucune entité géo détectée → tout en remarques
        return "", texte.strip()

    lieux = ", ".join(ent.text for ent in lieu_spans)

    # Remarques = texte original sans les spans reconnus comme lieux
    covered = set()
    for ent in lieu_spans:
        for i in range(ent.start_char, ent.end_char):
            covered.add(i)
    remarques = "".join(
        ch for i, ch in enumerate(texte) if i not in covered
    ).strip(" ,;-\n")

    return lieux, remarques


_nominatim_cache: dict[str, dict] = {}


def geocoder_nominatim(localite: str) -> dict:
    """
    Géocode une localité via Nominatim (OSM).

    Si la localité contient un numéro de département français (ex: 'St Vivien 24'),
    reformule la requête avec le nom du département ('St Vivien, Dordogne, France')
    et valide que le résultat est bien en France.

    Retourne {lat, lon, statut}.
    """
    if not localite or not localite.strip():
        return {"lat": "", "lon": "", "statut": "VIDE"}

    localite = localite.strip()
    if localite in _nominatim_cache:
        return _nominatim_cache[localite]

    def _make_result(res: dict) -> dict:
        return {
            "lat": res.get("lat", ""),
            "lon": res.get("lon", ""),
            "statut": "OK",
        }

    try:
        # ── Cas 1 : numéro de département français détecté ────────────────────
        dept_info = _parse_dept_fr(localite)
        if dept_info:
            ville, num_dept, nom_dept = dept_info
            query_fr = f"{ville}, {nom_dept}, France"
            results = _nominatim_request(query_fr)
            # On cherche un résultat en France
            for res in results:
                if _est_en_france(res):
                    result = _make_result(res)
                    _nominatim_cache[localite] = result
                    return result
            # Aucun résultat français avec le département → essai sans département
            results = _nominatim_request(f"{ville}, France")
            for res in results:
                if _est_en_france(res):
                    result = _make_result(res)
                    _nominatim_cache[localite] = result
                    return result

        # ── Cas 2 : requête directe ───────────────────────────────────────────
        results = _nominatim_request(localite)
        if results:
            # Si un numéro de département a été détecté, on exige la France
            if dept_info:
                fr_results = [r for r in results if _est_en_france(r)]
                if fr_results:
                    result = _make_result(fr_results[0])
                    _nominatim_cache[localite] = result
                    return result
                result = {"lat": "", "lon": "", "statut": "NON_GÉOCODÉ"}
            else:
                result = _make_result(results[0])
            _nominatim_cache[localite] = result
            return result

    except Exception:
        result = {"lat": "", "lon": "", "statut": "ERREUR_API"}
        _nominatim_cache[localite] = result
        return result

    result = {"lat": "", "lon": "", "statut": "NON_GÉOCODÉ"}
    _nominatim_cache[localite] = result
    return result


# ── Vérification numéro inventaire ────────────────────────────────────────────

def _cle_inventaire(s: str) -> str:
    """
    Extrait la clé de comparaison d'un identifiant spécimen.
    Stratégie : concaténation de toutes les séquences de chiffres.
    Ignore les séparateurs (tirets, espaces, points…) et les codes
    de collection qui peuvent être répétés ou absents sur l'étiquette.
    Ex: 'MNHN-EC-EC4849' -> '4849'
        'MNHN EC4849'    -> '4849'
        'MNHN.EC.4849'   -> '4849'
    """
    return "".join(re.findall(r"\d+", s))


def verifier_inventaire(ocr: str, specimen_id: str) -> str:
    if not ocr or not ocr.strip():
        return "VIDE"
    cle_ocr = _cle_inventaire(ocr)
    cle_sid = _cle_inventaire(specimen_id)
    if not cle_ocr:
        return "ERREUR"  # texte présent mais aucun chiffre → impossible à valider
    if cle_ocr == cle_sid:
        return "OK"
    return "ERREUR"


# ── Numérotation chronologique des déterminations ────────────────────────────

def _numeroter_determinations(dets: list[dict]) -> list[dict]:
    """
    Trie les déterminations par date_determination (ISO) et attribue
    numero_determination_N. Si la date est absente ou ambiguë,
    le statut est 'INCERTAIN'.
    """
    def _sort_key(d: dict) -> str:
        iso = d.get("date_determination", "") or ""
        # Les dates ISO partielles (YYYY, YYYY-MM, YYYY-MM-DD) se trient bien lexicalement
        return iso if iso and d.get("date_determination_statut") in ("OK", "AMBIGU") else "zzzz"

    has_date   = [d for d in dets if d.get("date_determination")]
    no_date    = [d for d in dets if not d.get("date_determination")]

    sorted_dets = sorted(has_date, key=_sort_key) + no_date

    for i, d in enumerate(sorted_dets, start=1):
        has_iso = bool(d.get("date_determination"))
        if has_iso:
            d["numero_determination"]        = str(i)
            d["numero_determination_statut"] = "OK"
        else:
            d["numero_determination"]        = str(i)
            d["numero_determination_statut"] = "INCERTAIN"

    return sorted_dets


# ── Traitement d'une ligne ────────────────────────────────────────────────────

def process_row(row: dict) -> dict:
    sid = row.get("specimen_id", "")
    out: dict = {"specimen_id": sid}

    # collecteur — pas de transformation
    brut = row.get("collecteur", "")
    out["collecteur"] = brut
    out["collecteur_brut"] = brut

    # date_collecte
    brut = row.get("date_collecte", "")
    iso, statut = normaliser_date(brut)
    out["date_collecte"] = iso
    out["date_collecte_brut"] = brut
    out["date_collecte_statut"] = statut

    # localite — NER spaCy → Nominatim
    brut = row.get("localite", "")
    lieux, remarques = extraire_lieux_ner(brut)
    geo = geocoder_nominatim(lieux or brut)   # fallback sur brut si NER vide
    out["localite_brut"]      = brut
    out["localite"]           = lieux or brut
    out["localite_statut"]    = geo["statut"]
    out["localite_remarques"] = remarques
    out["latitude"]           = geo["lat"]
    out["longitude"]          = geo["lon"]
    out["source_coordonnees"] = "Géocodage automatisé OSM" if geo["statut"] == "OK" else ""

    # numero_inventaire
    brut = row.get("numero_inventaire", "")
    statut_inv = verifier_inventaire(brut, sid)
    out["numero_inventaire"] = brut
    out["numero_inventaire_brut"] = brut
    out["numero_inventaire_statut"] = statut_inv

    # Déterminations (N possibles — colonnes determination_1, determination_2, ...)
    n_det = int(row.get("_n_determinations", "0") or "0")
    if n_det == 0:
        # Compatibilité avec ancien format (colonne unique)
        if row.get("determination"):
            n_det = 1

    dets_raw: list[dict] = []
    for n in range(1, n_det + 1):
        brut_det  = row.get(f"determination_{n}", row.get("determination", "") if n == 1 else "")
        brut_detr = row.get(f"determinateur_{n}", row.get("determinateur", "") if n == 1 else "")
        brut_date = row.get(f"date_determination_{n}", row.get("date_determination", "") if n == 1 else "")

        gbif = valider_gbif(brut_det)
        iso_date, statut_date = normaliser_date(brut_date)

        dets_raw.append({
            "determination":              gbif["nom"],
            "determination_brut":         brut_det,
            "determination_statut":       gbif["statut"],
            "determination_gbif_key":     gbif["key"],
            "determination_gbif_rang":    gbif["rang"],
            "determinateur":              brut_detr,
            "determinateur_brut":         brut_detr,
            "date_determination":         iso_date,
            "date_determination_brut":    brut_date,
            "date_determination_statut":  statut_date,
        })

    dets_numbered = _numeroter_determinations(dets_raw)
    out["n_determinations"] = str(len(dets_numbered))

    for i, d in enumerate(dets_numbered, start=1):
        out[f"determination_{i}"]              = d["determination"]
        out[f"determination_{i}_brut"]         = d["determination_brut"]
        out[f"determination_{i}_statut"]       = d["determination_statut"]
        out[f"determination_{i}_gbif_key"]     = d["determination_gbif_key"]
        out[f"determination_{i}_gbif_rang"]    = d["determination_gbif_rang"]
        out[f"determinateur_{i}"]              = d["determinateur"]
        out[f"determinateur_{i}_brut"]         = d["determinateur_brut"]
        out[f"date_determination_{i}"]         = d["date_determination"]
        out[f"date_determination_{i}_brut"]    = d["date_determination_brut"]
        out[f"date_determination_{i}_statut"]  = d["date_determination_statut"]
        out[f"numero_determination_{i}"]       = d["numero_determination"]
        out[f"numero_determination_{i}_statut"]= d["numero_determination_statut"]

    return out, len(dets_numbered)


# ── Boucle principale ─────────────────────────────────────────────────────────

def run_correction(
    input_csv: Path = HTR_GROUPED_CSV,
    output_csv: Path = CORRECTED_CSV,
    max_rows: int | None = None,
) -> list[dict]:
    if not input_csv.exists():
        print(f"[ERREUR] Fichier introuvable : {input_csv}")
        return []

    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[ERREUR] CSV vide.")
        return []

    if max_rows is not None:
        rows = rows[:max_rows]

    print(f"{len(rows)} spécimens à traiter")

    CORRECT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Première passe : traiter toutes les lignes pour connaître le max de déterminations
    results: list[dict] = []
    max_det = 0
    total = len(rows)

    print("  Passe 1/2 — traitement des champs…")
    for i, row in enumerate(rows):
        out, n_det = process_row(row)
        if n_det > max_det:
            max_det = n_det
        results.append(out)
        done = i + 1
        if done % 10 == 0 or done == total:
            print(f"    {done}/{total}")

    fieldnames = _output_fieldnames(max(max_det, 1))

    print(f"  Passe 2/2 — écriture CSV ({max_det} détermination(s) max)…")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for out in results:
            writer.writerow(out)

    # Résumé des statuts
    for col in ("date_collecte", "localite", "numero_inventaire"):
        col_statut = f"{col}_statut"
        counts: dict[str, int] = {}
        for r in results:
            s = r.get(col_statut, "")
            counts[s] = counts.get(s, 0) + 1
        print(f"  {col_statut} : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    for n in range(1, max_det + 1):
        for col in ("determination", "date_determination"):
            col_statut = f"{col}_{n}_statut"
            counts = {}
            for r in results:
                s = r.get(col_statut, "")
                counts[s] = counts.get(s, 0) + 1
            if any(counts.values()):
                print(f"  {col_statut} : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print(f"Résultats → {output_csv}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",     default=str(HTR_GROUPED_CSV))
    p.add_argument("--output",    default=str(CORRECTED_CSV))
    p.add_argument("--max-rows",  type=int, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    input_csv  = Path(args.input)
    output_csv = Path(args.output)
    if not input_csv.exists():
        sys.exit(f"Fichier introuvable : {input_csv}")
    run_correction(input_csv, output_csv, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
