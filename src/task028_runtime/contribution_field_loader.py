"""Load contribution fields exported by ``probe_cstc_contribution_tube.py``."""

import json
import re
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch


_VOLUME_SUFFIX = "_contribution_volumes"
_INDEXED_PREFIX = re.compile(r"^layer_\d+$")


def _load_sidecar_metadata(npz_path: Path) -> Mapping[str, object]:
    """Load the metadata sidecar written beside the NPZ, when available."""
    metadata_path = npz_path.with_name("cstc_probe_metadata.json")
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}")
    return metadata


def _resolve_layer_name(
    prefix: str, metadata: Mapping[str, object], npz_path: Path
) -> str:
    entry = metadata.get(prefix)
    if isinstance(entry, dict):
        layer_name = entry.get("layer")
        if isinstance(layer_name, str) and layer_name:
            return layer_name

    # Newer/direct exports may use the model layer name as the NPZ key prefix,
    # for example ``layers.3.blocks.0.attn_contribution_volumes``.
    if not _INDEXED_PREFIX.fullmatch(prefix):
        return prefix

    raise ValueError(
        f"{npz_path} uses indexed key prefix {prefix!r}, but the sibling "
        "cstc_probe_metadata.json needed to recover the model layer name is "
        "missing or incomplete"
    )


def _load_probe_volume(array: np.ndarray, key: str) -> torch.Tensor:
    """Convert one probe array to a structure-preserving ``[U,T,H,W]`` field.

    Direct ``[U,T,H,W]`` exports are loaded without any reduction, clipping,
    or flattening.  For legacy ``[S,U,T,H,W]`` exports, ``S`` is the probe
    sample axis and is combined by addition only; the unit, temporal, height,
    and width axes remain intact.  Addition is cosine-scale invariant and
    avoids introducing an averaged importance statistic in the loader.
    """
    if array.ndim == 5:
        if array.shape[0] == 0:
            raise ValueError(f"{key} has an empty probe-sample axis")
        field = np.asarray(array, dtype=np.float32).sum(
            axis=0, dtype=np.float32
        )  # [U,T,H,W]; reduce S only
    elif array.ndim == 4:
        field = np.asarray(array, dtype=np.float32)  # [U,T,H,W]
    else:
        raise ValueError(
            f"{key} must have shape [S,U,T,H,W] or [U,T,H,W], "
            f"but received {array.shape}"
        )

    if field.shape[0] == 0 or any(size == 0 for size in field.shape[1:]):
        raise ValueError(f"{key} contains an empty unit, temporal, or spatial axis")
    if not np.isfinite(field).all():
        raise ValueError(f"{key} contains NaN or infinity")
    return torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))


def load_contribution_fields(npz_path: str) -> Dict[str, torch.Tensor]:
    """Load all ``*_contribution_volumes`` arrays from one CSTC NPZ.

    Returns a dictionary mapping the exact model layer name to a CPU tensor
    with shape ``[U,T,H,W]``. ``U`` is the number of pruning units, ``T`` is
    time, and ``H/W`` are the spatial axes. Other CSTC arrays in the NPZ are
    intentionally ignored.
    """
    path = Path(npz_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Contribution-field NPZ not found: {path}")

    metadata = _load_sidecar_metadata(path)
    fields: Dict[str, torch.Tensor] = {}
    with np.load(path, allow_pickle=False) as archive:
        volume_keys = [
            key for key in archive.files if key.endswith(_VOLUME_SUFFIX)
        ]
        if not volume_keys:
            raise ValueError(
                f"{path} contains no keys ending with {_VOLUME_SUFFIX!r}"
            )

        for key in volume_keys:
            prefix = key[: -len(_VOLUME_SUFFIX)]
            layer_name = _resolve_layer_name(prefix, metadata, path)
            if layer_name in fields:
                raise ValueError(
                    f"Multiple contribution arrays resolve to layer {layer_name!r}"
                )
            fields[layer_name] = _load_probe_volume(archive[key], key)

    return fields
