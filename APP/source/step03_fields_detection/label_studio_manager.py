"""
Client Label Studio pour l'étape 3 : annotation des champs de métadonnées.

Deux projets distincts, chacun avec des classes restreintes à son type de zone :
  - collecte      → collecteur, date_collecte, localite
  - determination → determination, date_determination, determinateur, statut_nomenclatural

Les crops des deux projets alimentent ensuite le même pipeline HTR (step04).
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import requests

# ── Configs par type de zone ──────────────────────────────────────────────────

ZONE_CONFIGS: dict[str, dict] = {
    "collecte": {
        "title":   "MNHN - Champs collecte",
        "env_key": "LABEL_STUDIO_COLLECTE_PROJECT_ID",
        "classes": [
            {"name": "collecteur",   "color": "#F44336", "hotkey": "1"},
            {"name": "date_collecte","color": "#607D8B", "hotkey": "2"},
            {"name": "localite",     "color": "#00BCD4", "hotkey": "3"},
        ],
    },
    "determination": {
        "title":   "MNHN - Champs détermination",
        "env_key": "LABEL_STUDIO_DETERMINATION_PROJECT_ID",
        "classes": [
            {"name": "determination",        "color": "#4CAF50", "hotkey": "1"},
            {"name": "date_determination",   "color": "#FF9800", "hotkey": "2"},
            {"name": "determinateur",        "color": "#9C27B0", "hotkey": "3"},
            {"name": "statut_nomenclatural", "color": "#E91E63", "hotkey": "4"},
        ],
    },
}

# Helpers accès rapide
CLASS_NAMES: dict[str, list[str]] = {
    z: [c["name"] for c in cfg["classes"]]
    for z, cfg in ZONE_CONFIGS.items()
}


def _build_label_config(zone_type: str) -> str:
    classes = ZONE_CONFIGS[zone_type]["classes"]
    labels  = "\n".join(
        f'    <Label value="{c["name"]}" background="{c["color"]}" hotkey="{c["hotkey"]}"/>'
        for c in classes
    )
    return f"""<View>
  <Image name="image" value="$image"
         zoom="true" zoomControl="true" rotateControl="true"
         brightnessControl="true" contrastControl="true"/>
  <RectangleLabels name="champs" toName="image" showInline="true">
{labels}
  </RectangleLabels>
</View>
"""


class LSManager:
    """Wrapper REST Label Studio pour l'étape 3 (un manager par type de zone)."""

    def __init__(self, url: str, api_key: str, zone_type: str):
        if zone_type not in ZONE_CONFIGS:
            raise ValueError(f"zone_type inconnu : {zone_type!r} (attendu: {list(ZONE_CONFIGS)})")
        self.url        = url.rstrip("/")
        self._api_key   = api_key
        self.zone_type  = zone_type
        self._cfg       = ZONE_CONFIGS[zone_type]
        self.headers    = {
            "Authorization": f"Token {api_key}",
            "Content-Type":  "application/json",
        }

    @property
    def project_title(self) -> str:
        return self._cfg["title"]

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/health", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    _AUTH_ENDPOINTS = ["/api/current-user/whoami", "/api/users/me", "/api/users/"]

    def check_auth(self) -> bool:
        for scheme in ("Token", "Bearer"):
            for endpoint in self._AUTH_ENDPOINTS:
                try:
                    headers = {**self.headers, "Authorization": f"{scheme} {self._api_key}"}
                    r = requests.get(f"{self.url}{endpoint}", headers=headers, timeout=5)
                    if r.status_code == 200:
                        self.headers["Authorization"] = f"{scheme} {self._api_key}"
                        return True
                except requests.RequestException:
                    pass
        return False

    def list_projects(self) -> list[dict]:
        r = requests.get(f"{self.url}/api/projects/", headers=self.headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("results", data) if isinstance(data, dict) else data

    def get_or_create_project(self) -> int:
        for p in self.list_projects():
            if p["title"] == self.project_title:
                return p["id"]
        payload = {
            "title":        self.project_title,
            "label_config": _build_label_config(self.zone_type),
            "description":  f"Annotation des champs de métadonnées — zone {self.zone_type}",
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
        r = requests.get(
            f"{self.url}/api/projects/{project_id}/",
            headers=self.headers, timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        return {
            "total_tasks":       d.get("task_number", 0),
            "annotated":         d.get("num_tasks_with_annotations", 0),
            "skipped":           d.get("skipped_annotations_number", 0),
            "total_annotations": d.get("total_annotations_number", 0),
        }

    def import_images(self, project_id: int, image_urls: list[str], batch_size: int = 50) -> int:
        imported = 0
        for i in range(0, len(image_urls), batch_size):
            batch = image_urls[i: i + batch_size]
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

    def get_existing_urls(self, project_id: int) -> set[str]:
        """Retourne l'ensemble des URLs image déjà importées dans le projet."""
        page, existing = 1, set()
        while True:
            r = requests.get(
                f"{self.url}/api/tasks",
                headers=self.headers,
                params={"project": project_id, "page": page, "page_size": 500},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            tasks = data.get("tasks", data) if isinstance(data, dict) else data
            if not tasks:
                break
            for t in tasks:
                url = t.get("data", {}).get("image", "")
                if url:
                    existing.add(url)
            if isinstance(data, dict) and not data.get("next"):
                break
            page += 1
        return existing

    def export_tasks(self, project_id: int) -> list[dict]:
        r = requests.get(
            f"{self.url}/api/projects/{project_id}/export",
            headers=self.headers,
            params={"exportType": "JSON", "download_all_tasks": "false"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def delete_all_tasks(self, project_id: int) -> None:
        r = requests.delete(
            f"{self.url}/api/projects/{project_id}/tasks/",
            headers=self.headers, timeout=30,
        )
        r.raise_for_status()
