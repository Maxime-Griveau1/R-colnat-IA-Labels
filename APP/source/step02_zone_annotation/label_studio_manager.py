"""
Client Label Studio via l'API REST.

Usage :
    from source.step02_zone_annotation.label_studio_manager import LSManager
    ls = LSManager(url="http://localhost:8080", api_key="...")
    ls.ping()                    # vérifie la connexion
    pid = ls.get_or_create_project()
    ls.import_images(pid, [...])
    stats = ls.project_stats(pid)
    tasks  = ls.export_tasks(pid)
"""

from __future__ import annotations
import base64
import json
import time
from pathlib import Path
from typing import Any

import requests

# ── Config du projet Label Studio ─────────────────────────────────────────────

PROJECT_TITLE = "MNHN - Annotation des zones"

# Classes de zones (ordre = index YOLO)
ZONE_CLASSES: list[dict] = [
    {"name": "collecte",          "color": "#2196F3", "hotkey": "1"},  # étiquette de collecte (lieu, date, collecteur)
    {"name": "determination",     "color": "#E91E63", "hotkey": "2"},  # une boîte par détermination
    {"name": "tampon",            "color": "#9C27B0", "hotkey": "3"},
    {"name": "numero_inventaire", "color": "#FF9800", "hotkey": "4"},  # code-barre ou étiquette MNHN imprimée
    {"name": "graines",           "color": "#8BC34A", "hotkey": "5"},  # sachet de graines (herbier)
    {"name": "notes",             "color": "#4CAF50", "hotkey": "6"},  # annotations manuscrites libres
    {"name": "dessin",            "color": "#795548", "hotkey": "7"},  # dessin botanique / illustration
    {"name": "specimen",          "color": "#F44336", "hotkey": "8"},
    {"name": "logo",              "color": "#607D8B", "hotkey": "9"},
]

CLASS_NAMES: list[str] = [c["name"] for c in ZONE_CLASSES]

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image"
         zoom="true" zoomControl="true" rotateControl="true"
         brightnessControl="true" contrastControl="true"/>
  <RectangleLabels name="zones" toName="image" showInline="true">
""" + "\n".join(
    f'    <Label value="{c["name"]}" background="{c["color"]}" hotkey="{c["hotkey"]}"/>'
    for c in ZONE_CLASSES
) + """
  </RectangleLabels>
</View>
"""


def _is_jwt(token: str) -> bool:
    """Détecte si le token est un JWT (commence par eyJ…)."""
    return token.startswith("eyJ")


def _jwt_token_type(token: str) -> str:
    """
    Décode le payload JWT (sans vérification de signature) et retourne
    le champ 'token_type' ('access', 'refresh', ou inconnu).
    """
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return "unknown"
        # Padding base64
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("token_type", "unknown")
    except Exception:
        return "unknown"


class LSManager:
    """Wrapper minimaliste autour de l'API REST Label Studio."""

    def __init__(self, url: str, api_key: str):
        self.url = url.rstrip("/")
        self._api_key = api_key
        # LS 1.x utilise toujours « Token » même pour les JWT.
        # On part sur Token ; check_auth essaiera Bearer si Token échoue.
        self._scheme = "Bearer" if _is_jwt(api_key) else "Token"
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }

    def token_diagnosis(self) -> dict:
        """
        Retourne un diagnostic du token pour affichage dans l'UI.
        Si LS est accessible, effectue un vrai test des deux schémas.
        """
        if not self._api_key:
            return {"type": "absent", "msg": "Aucune clé API renseignée."}

        if _is_jwt(self._api_key):
            ttype = _jwt_token_type(self._api_key)
            if ttype == "refresh":
                return {
                    "type": "jwt_refresh",
                    "msg": (
                        "C'est un token JWT de type « refresh » — non valide pour l'API. "
                        "Dans Label Studio : Settings → Account & Settings → Access Token "
                        "→ copiez le token affiché (pas le Personal Access Token JWT)."
                    ),
                }
            # JWT access ou inconnu : tenter les deux schémas
            if self.ping():
                for scheme in ("Token", "Bearer"):
                    try:
                        h = {**self.headers, "Authorization": f"{scheme} {self._api_key}"}
                        r = requests.get(f"{self.url}/api/current-user/whoami", headers=h, timeout=5)
                        if r.status_code == 200:
                            return {"type": f"jwt_{ttype}", "msg": f"Token JWT valide (schéma : {scheme})."}
                    except Exception:
                        pass
                return {
                    "type": "jwt_invalid",
                    "msg": (
                        "Token JWT rejeté avec Token et Bearer. "
                        "Vérifiez qu'il s'agit bien du token d'accès dans "
                        "Settings → Account & Settings → Access Token."
                    ),
                }
            return {"type": f"jwt_{ttype}", "msg": f"Token JWT (type: {ttype}) — LS hors ligne, impossible de tester."}

        return {"type": "legacy", "msg": "Token legacy (format Token)."}

    # ── Connectivité ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Retourne True si Label Studio répond."""
        try:
            r = requests.get(f"{self.url}/api/health", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # Endpoints d'authentification testés dans l'ordre (compatibilité multi-versions)
    _AUTH_ENDPOINTS = ["/api/current-user/whoami", "/api/users/me", "/api/users/"]

    def check_auth(self) -> bool:
        """
        Retourne True si la clé API est valide.
        Teste automatiquement les schémas Token/Bearer sur plusieurs
        endpoints (compatibilité LS v1.x → v1.12+).
        """
        for scheme in ("Token", "Bearer"):
            for endpoint in self._AUTH_ENDPOINTS:
                try:
                    headers = {**self.headers, "Authorization": f"{scheme} {self._api_key}"}
                    r = requests.get(f"{self.url}{endpoint}", headers=headers, timeout=5)
                    if r.status_code == 200:
                        self.headers["Authorization"] = f"{scheme} {self._api_key}"
                        self._scheme = scheme
                        return True
                except requests.RequestException:
                    pass
        return False

    # ── Projet ────────────────────────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        r = requests.get(f"{self.url}/api/projects/", headers=self.headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("results", data) if isinstance(data, dict) else data

    def get_or_create_project(self) -> int:
        """Retourne l'id du projet MNHN existant, ou le crée."""
        for p in self.list_projects():
            if p["title"] == PROJECT_TITLE:
                return p["id"]
        # Création
        payload = {
            "title": PROJECT_TITLE,
            "label_config": LABEL_CONFIG,
            "description": "Annotation des zones sur les images de spécimens MNHN",
        }
        r = requests.post(
            f"{self.url}/api/projects/",
            headers=self.headers,
            data=json.dumps(payload),
            timeout=15,
        )
        r.raise_for_status()
        return r.json()["id"]

    def project_stats(self, project_id: int) -> dict:
        """Retourne les statistiques d'annotation du projet."""
        r = requests.get(
            f"{self.url}/api/projects/{project_id}/",
            headers=self.headers, timeout=10
        )
        r.raise_for_status()
        d = r.json()
        return {
            "total_tasks":       d.get("task_number", 0),
            "annotated":         d.get("num_tasks_with_annotations", 0),
            "skipped":           d.get("skipped_annotations_number", 0),
            "total_annotations": d.get("total_annotations_number", 0),
        }

    # ── Import d'images ───────────────────────────────────────────────────────

    def import_images(
        self,
        project_id: int,
        image_urls: list[str],
        batch_size: int = 50,
    ) -> int:
        """Importe des images comme tâches LS. Retourne le nombre importé."""
        imported = 0
        for i in range(0, len(image_urls), batch_size):
            batch = image_urls[i : i + batch_size]
            tasks = [{"data": {"image": url}} for url in batch]
            r = requests.post(
                f"{self.url}/api/projects/{project_id}/import",
                headers=self.headers,
                data=json.dumps(tasks),
                timeout=30,
            )
            r.raise_for_status()
            imported += len(batch)
            time.sleep(0.1)
        return imported

    # ── Export ────────────────────────────────────────────────────────────────

    def export_tasks(self, project_id: int) -> list[dict]:
        """Retourne toutes les tâches annotées (format JSON LS)."""
        r = requests.get(
            f"{self.url}/api/projects/{project_id}/export",
            headers=self.headers,
            params={"exportType": "JSON", "download_all_tasks": "false"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def delete_all_tasks(self, project_id: int) -> None:
        """Supprime toutes les tâches du projet (utile avant un re-import)."""
        r = requests.delete(
            f"{self.url}/api/projects/{project_id}/tasks/",
            headers=self.headers, timeout=30
        )
        r.raise_for_status()
