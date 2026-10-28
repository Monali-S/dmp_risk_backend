"""Tiny thread-safe JSON-file store for missions (swap for a real DB later)."""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import Mission


class MissionStore:
    def __init__(self, path: str = "data/missions.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._data = {}

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2))

    def create(self, mission: Mission) -> Mission:
        with self._lock:
            mission = mission.model_copy(update={"id": mission.id or uuid.uuid4().hex[:8]})
            self._data[mission.id] = mission.model_dump(mode="json")
            self._flush()
            return mission

    def get(self, mission_id: str) -> Optional[Mission]:
        d = self._data.get(mission_id)
        return Mission(**d) if d else None

    def list(self) -> List[Mission]:
        return [Mission(**d) for d in self._data.values()]

    def update(self, mission_id: str, mission: Mission) -> Optional[Mission]:
        with self._lock:
            if mission_id not in self._data:
                return None
            mission = mission.model_copy(update={"id": mission_id})
            self._data[mission_id] = mission.model_dump(mode="json")
            self._flush()
            return mission

    def delete(self, mission_id: str) -> bool:
        with self._lock:
            if mission_id not in self._data:
                return False
            del self._data[mission_id]
            self._flush()
            return True
