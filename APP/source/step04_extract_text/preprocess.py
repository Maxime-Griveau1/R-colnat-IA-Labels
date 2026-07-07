"""
Prétraitement des crops step03 avant HTR.

Objectif : maximiser la lisibilité du texte manuscrit ou imprimé
pour le modèle HTR, en compensant :
  - flou de mise au point ou de numérisation
  - contraste faible (papier jauni, encre pâlie)
  - bruit de fond (texture du papier, taches)

Pipeline appliqué dans l'ordre :
  1. Conversion en niveaux de gris
  2. Deskew léger (correction de l'inclinaison ≤ 5°)
  3. Débruitage (filtre bilatéral — préserve les bords du texte)
  4. Rehaussement du contraste (CLAHE)
  5. Netteté (unsharp mask)
  6. Binarisation adaptative (Sauvola) → image N&B finale

Usage :
  from source.step04_extract_text.preprocess import preprocess_crop
  img_processed = preprocess_crop(path_or_pil_image)
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage


# ── Paramètres ────────────────────────────────────────────────────────────────

# Taille minimale (px) en petite dimension — en dessous on upscale avant traitement
MIN_SHORT_SIDE = 64

# Résolution cible pour l'HTR (hauteur en px) — None = pas de redimensionnement
HTR_HEIGHT: int | None = 128


def preprocess_crop(
    src: Path | PILImage.Image | np.ndarray,
    *,
    target_height: int | None = HTR_HEIGHT,
    binarize: bool = True,
    deskew: bool = True,
) -> PILImage.Image:
    """
    Prétraite un crop de champ pour l'HTR.

    Retourne une image PIL en niveaux de gris (L) ou N&B (si binarize=True).
    """
    img = _to_numpy(src)

    # 1. Niveaux de gris
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Upscale si trop petite
    h, w = gray.shape
    if min(h, w) < MIN_SHORT_SIDE:
        scale = MIN_SHORT_SIDE / min(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 3. Deskew (≤ 15° de correction)
    if deskew:
        gray = _deskew(gray)

    # 4. Débruitage bilatéral (préserve les contours)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)

    # 5. CLAHE — rehaussement local du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 6. Unsharp mask — netteté
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    gray = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)

    # 7. Binarisation Sauvola (adaptative locale)
    if binarize:
        gray = _sauvola(gray)

    # 8. Redimensionnement à hauteur cible (HTR attend une hauteur fixe)
    if target_height is not None:
        h, w = gray.shape
        new_w = max(1, int(w * target_height / h))
        gray = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_AREA)

    return PILImage.fromarray(gray, mode="L")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_numpy(src: Path | PILImage.Image | np.ndarray) -> np.ndarray:
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, Path):
        src = PILImage.open(src)
    arr = np.array(src.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Corrige l'inclinaison par projection de Hough sur les lignes de texte."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=80,
                             minLineLength=gray.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:
                angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _sauvola(gray: np.ndarray, window: int = 25, k: float = 0.2) -> np.ndarray:
    """
    Binarisation de Sauvola : seuil local basé sur moyenne et écart-type locaux.
    Meilleure que l'Otsu global sur du texte à contraste variable.
    """
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, -1, (window, window))
    mean_sq = cv2.boxFilter(gray_f ** 2, -1, (window, window))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    # Seuil de Sauvola : T = mean * (1 + k * (std/128 - 1))
    threshold = mean * (1.0 + k * (std / 128.0 - 1.0))
    binary = np.where(gray_f <= threshold, 0, 255).astype(np.uint8)
    return binary
