"""Task037 standalone loader for the authoritative Task021 implementation.

The implementation is retained byte-identically under verification/archived_source_snapshots.
This wrapper keeps the original top-level module name and exposes that implementation
without importing any historical source directory.
"""
from pathlib import Path
import runpy

_ARCHIVE = Path(__file__).resolve().parents[2] / "verification" / "archived_source_snapshots" / "task021_importance_information_retention.py"
_NAMESPACE = runpy.run_path(str(_ARCHIVE), run_name=__name__)
for _name, _value in _NAMESPACE.items():
    if _name not in {"__name__", "__file__", "__cached__", "__loader__", "__package__", "__spec__"}:
        globals()[_name] = _value
