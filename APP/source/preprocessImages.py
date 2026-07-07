from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _letterbox(img: np.ndarray, size: int = 1024, pad_color: int = 128) -> np.ndarray:
    """Resize proportionnel + padding centré vers un carré size×size."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((size, size, img.shape[2] if img.ndim == 3 else 1),
                     pad_color, dtype=np.uint8)
    if img.ndim == 2:
        canvas = np.full((size, size), pad_color, dtype=np.uint8)

    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE sur le canal L (espace LAB) — améliore le contraste local sans surexposition."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _binarize(img: np.ndarray) -> np.ndarray:
    """
    Binarisation adaptative (Gaussian) sur niveaux de gris.
    Retourne une image BGR (3 canaux) pour uniformiser le pipeline.
    À activer uniquement pour les HTR entraînés sur images binaires (Kraken, etc.).
    Désactiver pour les modèles multimodaux comme GLM-OCR.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def preprocess_single(
    src_path: str | Path,
    dst_path: str | Path,
    size: int = 1024,
    clahe: bool = True,
    binarize: bool = False,
) -> None:
    """Applique le pipeline de prétraitement sur une image."""
    img = cv2.imread(str(src_path))
    if img is None:
        raise ValueError(f"Impossible de lire : {src_path}")

    if clahe:
        img = _apply_clahe(img)

    if binarize:
        img = _binarize(img)

    img = _letterbox(img, size=size)
    cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def preprocessImages(
    in_paths: list[str],
    out_paths: list[str],
    max_images: int | None = None,
    size: int = 1024,
    clahe: bool = True,
    binarize: bool = False,
) -> None:
    """
    Prétraite les images de in_paths vers out_paths.

    Pipeline (dans l'ordre) :
      1. CLAHE (contraste local adaptatif, canal L en LAB)   — si clahe=True
      2. Binarisation adaptative Gaussian                     — si binarize=True
         ⚠ Désactivé par défaut : contre-productif pour GLM-OCR
      3. Resize proportionnel + letterbox gris 128 → size×size

    Args:
        in_paths   : liste de dossiers sources
        out_paths  : liste de dossiers de destination (même ordre)
        max_images : limite par dossier (None = toutes)
        size       : côté du carré de sortie en pixels (défaut 1024)
        clahe      : activer le rehaussement de contraste local
        binarize   : activer la binarisation adaptative
    """
    for in_path, out_path in zip(in_paths, out_paths):
        os.makedirs(out_path, exist_ok=True)

        images = sorted(
            f for f in os.listdir(in_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        )

        if max_images is not None:
            images = images[:max_images]

        total = len(images)
        ops = []
        if clahe:
            ops.append("CLAHE")
        if binarize:
            ops.append("binarisation")
        ops.append(f"letterbox {size}px")
        print(f"  → {os.path.basename(in_path)} : {total} images [{', '.join(ops)}]")

        errors = 0
        for idx, image in enumerate(images, 1):
            src = os.path.join(in_path, image)
            dst = os.path.join(out_path, os.path.splitext(image)[0] + ".jpg")
            try:
                preprocess_single(src, dst, size=size, clahe=clahe, binarize=binarize)
            except Exception as e:
                print(f"    [ERREUR] {image} : {e}")
                errors += 1
            if idx % 100 == 0 or idx == total:
                print(f"    {idx}/{total}" + (f" ({errors} erreurs)" if errors else ""))
