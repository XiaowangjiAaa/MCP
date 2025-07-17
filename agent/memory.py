import os
import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

class MemoryController:
    def __init__(self, filepath: str = "memory_store.jsonl"):
        self.filepath = Path(filepath)
        self.records: List[Dict[str, Any]] = []
        self.alias_map = {
            "最大宽度": "max_width",
            "平均宽度": "avg_width",
            "最大裂缝宽度": "max_width",
            "宽度最大值": "max_width",
            "avgwidth": "avg_width",
            "最大宽": "max_width"
        }
        self._load_memory()

    def _load_memory(self):
        if not self.filepath.exists():
            return
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    self.records.append(obj)
                except Exception:
                    continue

    def _save_record(self, record: Dict):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def normalize(self, s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "")

    def to_standard_metric(self, name: str) -> str:
        name = self.normalize(name)
        for alias, standard in self.alias_map.items():
            if self.normalize(alias) == name:
                return standard
        return name

    def _record_exists(self, subject: str, task: str, pixel_size: float = None) -> bool:
        for r in self.records:
            if r.get("subject") == subject and r.get("context", {}).get("task") == task:
                if pixel_size is None:
                    return True
                if abs(r["context"].get("pixel_size_mm", -1) - pixel_size) < 1e-6:
                    return True
        return False

    def update_context(self, intent: str, indices: List[int], pixel_size: float, results: List[Dict], plan: List[Dict] = None):
        for r in results:
            tool = r.get("tool")
            status = r.get("status")
            if status != "success":
                continue

            args = r.get("args", {})
            subject = Path(args.get("image_path") or args.get("mask_path", "")).stem

            if tool == "segment_crack_image":
                record = {
                    "subject": subject,
                    "context": {
                        "task": "segment",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "observation": {
                        "mask_path": r.get("outputs", {}).get("mask_path", "")
                    }
                }
                if not self._record_exists(subject, "segment"):
                    self.records.append(record)
                    self._save_record(record)

            elif tool == "quantify_crack_geometry":
                pixel_size = args.get("pixel_size_mm", 0.5)
                outputs = r.get("outputs", {})
                visuals = r.get("visualizations", {})

                if outputs:
                    record = {
                        "subject": subject,
                        "context": {
                            "task": "quantify",
                            "pixel_size_mm": pixel_size,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        "observation": outputs
                    }
                    if not self._record_exists(subject, "quantify", pixel_size):
                        self.records.append(record)
                        self._save_record(record)

                if visuals:
                    record = {
                        "subject": subject,
                        "context": {
                            "task": "save",
                            "pixel_size_mm": pixel_size,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        "observation": visuals
                    }
                    if not self._record_exists(subject, "save", pixel_size):
                        self.records.append(record)
                        self._save_record(record)

    def save_mask_path(self, subject_name: str, mask_path: str):
        record = {
            "subject": subject_name,
            "context": {
                "task": "segment",
                "timestamp": datetime.utcnow().isoformat()
            },
            "observation": {
                "mask_path": mask_path
            }
        }
        self.records.append(record)
        self._save_record(record)

    def save_metrics(self, subject_name: str, pixel_size: float, metrics: Dict[str, Any]):
        record = {
            "subject": subject_name,
            "context": {
                "task": "quantify",
                "pixel_size_mm": pixel_size,
                "timestamp": datetime.utcnow().isoformat()
            },
            "observation": metrics
        }
        self.records.append(record)
        self._save_record(record)

    def get_metrics_by_name(self, name: str, pixel_size: float = None) -> Dict[str, Any]:
        matches = [
            r for r in self.records
            if r.get("subject") == name and r.get("context", {}).get("task") == "quantify"
        ]
        if pixel_size is not None:
            matches = [
                r for r in matches
                if abs(r["context"].get("pixel_size_mm", 0) - pixel_size) < 1e-6
            ]
        if not matches:
            return {}
        return matches[-1].get("observation", {})

    def get_mask_path(self, name: str) -> str:
        for r in reversed(self.records):
            if r.get("subject") == name:
                obs = r.get("observation", {})
                if isinstance(obs, dict) and "mask_path" in obs:
                    return obs["mask_path"]
        return ""

    def get_pixel_size(self, subject_name: str) -> float:
        for r in reversed(self.records):
            if r.get("subject") == subject_name and r.get("context", {}).get("task") == "quantify":
                return r.get("context", {}).get("pixel_size_mm")
        return None

    def get_last_metrics(self, count: int = 5) -> Dict[str, Dict[str, Any]]:
        latest = self.records[-count:]
        return {r["subject"]: r["observation"] for r in latest if "observation" in r}

    def has_metrics(self, name: str, requested_metrics: List[str], pixel_size: float = None) -> bool:
        existing = self.get_metrics_by_name(name, pixel_size)
        for m in requested_metrics:
            norm_m = self.normalize(self.to_standard_metric(m))
            if not any(norm_m in self.normalize(k) for k in existing):
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_task": self.records[-1]["context"]["task"] if self.records else None,
            "known_subjects": list({r["subject"] for r in self.records}),
            "recent_metrics": self.get_last_metrics()
        }

    def export_latest_snapshot(self, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def clear(self):
        self.records = []
        if self.filepath.exists():
            self.filepath.unlink()

    def get_visualization_path(self, subject_name: str, visual_type: str) -> str:
        """
        从 observation 中提取视觉图路径，自动匹配 *_overlay 字段。
        例如 visual_type="max_width" → 查找 "max_width_overlay"
        """
        overlay_key = f"{visual_type}_overlay"

        for r in reversed(self.records):
            if r.get("subject") == subject_name:
                vis = r.get("observation", {})
                if isinstance(vis, dict) and overlay_key in vis:
                    return vis[overlay_key]
        return ""

    def update_visualization_path(self, subject_name: str, visual_type: str, path: str):
        for r in reversed(self.records):
            if r.get("subject") == subject_name and r.get("context", {}).get("task") == "save":
                if "observation" not in r or not isinstance(r["observation"], dict):
                    r["observation"] = {}
                r["observation"][visual_type] = path
                self._save_record(r)
                return
        new_record = {
            "subject": subject_name,
            "context": {
                "task": "save",
                "pixel_size_mm": 0.5,
                "timestamp": datetime.utcnow().isoformat()
            },
            "observation": {
                visual_type: path
            }
        }
        self.records.append(new_record)
        self._save_record(new_record)

    def save_visualizations(self, subject_name: str, pixel_size_mm: float, visual_paths: Dict[str, str]):
        for r in reversed(self.records):
            if r.get("subject") == subject_name and r.get("context", {}).get("task") == "save":
                if "observation" not in r or not isinstance(r["observation"], dict):
                    r["observation"] = {}
                r["observation"].update(visual_paths)
                self._save_record(r)
                return
        new_record = {
            "subject": subject_name,
            "context": {
                "task": "save",
                "pixel_size_mm": pixel_size_mm,
                "timestamp": datetime.utcnow().isoformat()
            },
            "observation": visual_paths
        }
        self.records.append(new_record)
        self._save_record(new_record)
