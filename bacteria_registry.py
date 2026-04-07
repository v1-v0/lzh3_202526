"""
Bacteria Registry
Persistent store for bacteria profiles used by both tuner.py and dev0406.py.
Metadata is serialised to  bacteria_configs/registry.json.

Usage
-----
from bacteria_registry import registry           # singleton
registry.register("Escherichia coli", ...)       # add
registry.remove("escherichia_coli")              # delete
registry.get_whitelist()                         # validated keys for multi-scan
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_REGISTRY_PATH = Path("bacteria_configs") / "registry.json"

# Seed data written on first run when no registry.json exists yet.
_BUILTINS: Dict[str, dict] = {
    "proteus_mirabilis": {
        "config_key":    "proteus_mirabilis",
        "display_name":  "Proteus mirabilis",
        "description":   "Rod-shaped, flagellated bacterium",
        "common_in":     "Catheter-associated infections",
        "validated":     True,
        "registered_at": "2026-01-01T00:00:00",
        "registered_by": "System",
        "builtin":       True,
    },
    "klebsiella_pneumoniae": {
        "config_key":    "klebsiella_pneumoniae",
        "display_name":  "Klebsiella pneumoniae",
        "description":   "Gram-negative, encapsulated bacterium",
        "common_in":     "Healthcare-associated infections",
        "validated":     True,
        "registered_at": "2026-01-01T00:00:00",
        "registered_by": "System",
        "builtin":       True,
    },
    "streptococcus_mitis": {
        "config_key":    "streptococcus_mitis",
        "display_name":  "Streptococcus mitis",
        "description":   "Gram-positive cocci in chains",
        "common_in":     "Touch contamination",
        "validated":     False,
        "registered_at": "2026-01-01T00:00:00",
        "registered_by": "System",
        "builtin":       True,
    },
}


def name_to_key(display_name: str) -> str:
    """Convert a human-readable display name to a filesystem-safe config key.

    Examples
    --------
    "Escherichia coli"  →  "escherichia_coli"
    "E. coli O157:H7"   →  "e_coli_o157_h7"
    """
    key = display_name.lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


class BacteriaRegistry:
    """Persistent registry for bacteria configurations.

    Do not instantiate directly — use the module-level ``registry`` singleton.
    """

    def __init__(self, path: Path = _REGISTRY_PATH) -> None:
        self._path = path
        self._data: Dict[str, dict] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
                print(f"[Registry] Loaded {len(self._data)} bacteria.")
                return
            except Exception as exc:
                print(f"[Registry] Could not read registry ({exc}). Seeding defaults.")

        # First-run: seed with built-ins and persist immediately.
        self._data = {k: dict(v) for k, v in _BUILTINS.items()}
        self._save()

    def _save(self) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
            return True
        except Exception as exc:
            print(f"[Registry] Save failed: {exc}")
            return False

    # ── public read API ───────────────────────────────────────────────────────

    def all(self) -> Dict[str, dict]:
        """Return a copy of all registered bacteria, keyed by config_key."""
        return {k: dict(v) for k, v in self._data.items()}

    def get(self, config_key: str) -> Optional[dict]:
        """Return metadata for one bacterium, or None if not found."""
        entry = self._data.get(config_key)
        return dict(entry) if entry else None

    def get_whitelist(self) -> List[str]:
        """Return config_keys of validated bacteria (used in multi-scan)."""
        return [k for k, v in self._data.items() if v.get("validated", False)]

    def key_exists(self, config_key: str) -> bool:
        return config_key in self._data

    def has_json_config(self, config_key: str) -> bool:
        """True when a tuned bacteria_configs/<key>.json file exists on disk."""
        return (Path("bacteria_configs") / f"{config_key}.json").exists()

    # ── public write API ──────────────────────────────────────────────────────

    def register(
        self,
        display_name: str,
        *,
        description:   str            = "",
        common_in:     str            = "",
        validated:     bool           = False,
        config_key:    Optional[str]  = None,
        registered_by: str            = "User",
    ) -> str:
        """Register a new bacterium.  Returns the config_key used.

        Raises ValueError if the key is already taken.
        """
        if not config_key:
            config_key = name_to_key(display_name)
        if not config_key:
            raise ValueError("Cannot derive a valid config key from the display name.")
        if config_key in self._data:
            raise ValueError(f"Config key '{config_key}' is already registered.")

        self._data[config_key] = {
            "config_key":    config_key,
            "display_name":  display_name,
            "description":   description,
            "common_in":     common_in,
            "validated":     validated,
            "registered_at": datetime.now().isoformat(),
            "registered_by": registered_by,
            "builtin":       False,
        }
        self._save()
        print(f"[Registry] Registered '{display_name}' → '{config_key}'")
        return config_key

    def remove(self, config_key: str, *, delete_json: bool = False) -> bool:
        """Remove a bacterium from the registry.

        Parameters
        ----------
        config_key  : registry key to remove
        delete_json : when True, also deletes bacteria_configs/<key>.json
        """
        if config_key not in self._data:
            print(f"[Registry] '{config_key}' not found.")
            return False

        del self._data[config_key]
        self._save()

        if delete_json:
            cfg_file = Path("bacteria_configs") / f"{config_key}.json"
            if cfg_file.exists():
                try:
                    cfg_file.unlink()
                    print(f"[Registry] Deleted config file: {cfg_file.name}")
                except Exception as exc:
                    print(f"[Registry] Could not delete {cfg_file.name}: {exc}")

        print(f"[Registry] Removed '{config_key}'")
        return True

    def set_validated(self, config_key: str, validated: bool) -> bool:
        """Toggle the multi-scan whitelist membership for one bacterium."""
        if config_key not in self._data:
            return False
        self._data[config_key]["validated"] = validated
        self._save()
        return True

    def update(self, config_key: str, **fields) -> bool:
        """Update editable metadata fields for a registered bacterium."""
        if config_key not in self._data:
            return False
        allowed = {"display_name", "description", "common_in", "validated"}
        for k, v in fields.items():
            if k in allowed:
                self._data[config_key][k] = v
        self._save()
        return True


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this in both tuner.py and dev0406.py:
#   from bacteria_registry import registry as _bacteria_registry
registry = BacteriaRegistry()