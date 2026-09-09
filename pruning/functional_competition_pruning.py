"""GPU-first functional competition selection for Task014.

The Dynamic3D descriptor and BMS implementation live in :mod:`MC`.  This
module starts only after BMS has produced competition domains.  It audits the
one-to-one descriptor/cache mapping, aligns the nine per-video Contribution
Fields on CUDA, builds one similarity matrix per BMS domain, and selects the
next feasible unit by set-dependent marginal functional coverage loss.

No descriptor value, BMS score, unit type, or parameter cost enters the
functional ranking.  Parameter cost is read only after a feasible unit has
been selected, to accumulate the existing global parameter budget.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


EXPECTED_NPZ_ARRAYS = 192
EXPECTED_PRUNING_LAYERS = 48
EXPECTED_DESCRIPTOR_UNITS = 36_378
EXPECTED_VIDEO_FIELDS = 9
ALIGNED_FIELD_SHAPE = (16, 7, 7)
NORMALIZATION_EPS = 1e-8
FUNCTIONAL_SCORE_MODES = ("domain_average", "domain_total")

# These are bounded implementation chunks, not method hyperparameters.
_POOL_UNIT_CHUNK = 256
_SIMILARITY_ROW_CHUNK = 1024
_VOLUME_SUFFIX = "_contribution_volumes"


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_npy_header(handle) -> tuple[tuple[int, ...], bool, np.dtype]:
    """Read one NPY member header without materializing its array body."""
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version in ((2, 0), (3, 0)):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"Unsupported NPY header version: {version}")
    return tuple(int(size) for size in shape), bool(fortran_order), np.dtype(dtype)


def _normalized_unit_type(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"attention", "attn", "head", "attention_head", "windowattention3d"}:
        return "attention_head"
    if text in {"mlp", "ffn", "neuron", "ffn_neuron"}:
        return "ffn_neuron"
    return text


@dataclass(frozen=True)
class ArrayHeader:
    """Header-only NPZ member description; no array payload is loaded."""

    key: str
    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool


@dataclass(frozen=True)
class LayerFieldSpec:
    """Strict mapping for one pruning module.

    ``field_shape`` is ``[S,U,T,H,W]``: video sample, pruning unit, temporal,
    height, and width.  ``global_start:global_stop`` is the descriptor-row
    range belonging to the layer.
    """

    layer_index: int
    layer: str
    unit_type: str
    cache_key: str
    field_shape: tuple[int, int, int, int, int]
    dtype: str
    global_start: int
    global_stop: int

    @property
    def unit_count(self) -> int:
        return int(self.field_shape[1])


@dataclass(frozen=True)
class UnitFieldMapping:
    """One descriptor row mapped to one cached Contribution Field."""

    global_index: int
    layer_index: int
    layer: str
    unit_type: str
    unit_index: int
    cache_key: str


@dataclass
class MappingAudit:
    """Evidence that descriptor rows and cached fields form a bijection."""

    npz_path: str
    npz_size_bytes: int
    npz_mtime_ns: int
    metadata_path: str | None
    metadata_sha256: str | None
    array_count: int
    contribution_layer_count: int
    cache_units: int
    descriptor_units: int
    mapped_units: int
    missing: int
    duplicate: int
    mapping_sha256: str
    layers: list[LayerFieldSpec]
    units: list[UnitFieldMapping]

    def summary(self) -> dict[str, object]:
        return {
            "status": "passed",
            "npz_path": self.npz_path,
            "npz_size_bytes": self.npz_size_bytes,
            "npz_mtime_ns": self.npz_mtime_ns,
            "metadata_path": self.metadata_path,
            "metadata_sha256": self.metadata_sha256,
            "array_count": self.array_count,
            "contribution_layer_count": self.contribution_layer_count,
            "cache_units": self.cache_units,
            "descriptor_units": self.descriptor_units,
            "mapped_units": self.mapped_units,
            "missing": self.missing,
            "duplicate": self.duplicate,
            "mapping_sha256": self.mapping_sha256,
            "expected_video_fields": EXPECTED_VIDEO_FIELDS,
            "aligned_field_shape_per_video": list(ALIGNED_FIELD_SHAPE),
            "sample_aggregation": "none; nine fields are concatenated in sample order",
            "layer_wise_lazy_loading": True,
        }

    def write(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        _atomic_json(output_dir / "contribution_mapping_audit.json", self.summary())
        _atomic_csv(
            output_dir / "contribution_layer_mapping.csv",
            (
                "layer_index",
                "layer",
                "unit_type",
                "cache_key",
                "sample_count",
                "unit_count",
                "temporal_size",
                "height",
                "width",
                "global_start",
                "global_stop",
            ),
            (
                {
                    "layer_index": spec.layer_index,
                    "layer": spec.layer,
                    "unit_type": spec.unit_type,
                    "cache_key": spec.cache_key,
                    "sample_count": spec.field_shape[0],
                    "unit_count": spec.field_shape[1],
                    "temporal_size": spec.field_shape[2],
                    "height": spec.field_shape[3],
                    "width": spec.field_shape[4],
                    "global_start": spec.global_start,
                    "global_stop": spec.global_stop,
                }
                for spec in self.layers
            ),
        )
        _atomic_csv(
            output_dir / "contribution_unit_mapping.csv",
            (
                "global_index",
                "layer_index",
                "layer",
                "unit_type",
                "unit_index",
                "cache_key",
            ),
            (asdict(unit) for unit in self.units),
        )


@dataclass(frozen=True)
class ZeroFieldUnit:
    """Root-cause evidence for one null or otherwise fatal source unit.

    A source unit has shape ``[S,T,H,W] = [9,16,H_l,W_l]``.  Per-video
    statistics therefore have shape ``[S] = [9]``.  ``classification`` is
    exactly one of ``case_a_exact_zero``, ``case_b_raw_near_zero``,
    ``case_c_pooling_induced_zero``, ``case_d_mapping_error`` or
    ``case_e_nonfinite``.
    """

    global_index: int
    layer_index: int
    layer_name: str
    unit_type: str
    unit_index: int
    source_array_key: str
    raw_shape: tuple[int, int, int, int]
    raw_min: float | None
    raw_max: float | None
    raw_abs_sum: float | None
    raw_l2: float | None
    raw_abs_sum_per_video: tuple[float | None, ...]
    raw_l2_per_video: tuple[float | None, ...]
    pooled_abs_sum_per_video: tuple[float | None, ...]
    pooled_l2_per_video: tuple[float | None, ...]
    concatenated_l2: float | None
    is_raw_all_zero: bool
    is_pooled_all_zero: bool
    classification: str

    def csv_row(self) -> dict[str, object]:
        row = asdict(self)
        for key in (
            "raw_shape",
            "raw_abs_sum_per_video",
            "raw_l2_per_video",
            "pooled_abs_sum_per_video",
            "pooled_l2_per_video",
        ):
            row[key] = json.dumps(row[key], ensure_ascii=False)
        return row


ZERO_FIELD_CSV_FIELDS = (
    "global_index",
    "layer_index",
    "layer_name",
    "unit_type",
    "unit_index",
    "source_array_key",
    "raw_shape",
    "raw_min",
    "raw_max",
    "raw_abs_sum",
    "raw_l2",
    "raw_abs_sum_per_video",
    "raw_l2_per_video",
    "pooled_abs_sum_per_video",
    "pooled_l2_per_video",
    "concatenated_l2",
    "is_raw_all_zero",
    "is_pooled_all_zero",
    "classification",
)


@dataclass
class ZeroFieldAudit:
    """Classification of null/fatal functional fields over all mapped units."""

    units: list[ZeroFieldUnit]
    total_element_count: int
    finite_element_count: int
    negative_element_count: int
    exact_zero_element_count: int
    global_min: float | None
    global_max: float | None

    def summary(self) -> dict[str, object]:
        counts = {
            name: sum(unit.classification == name for unit in self.units)
            for name in (
                "case_a_exact_zero",
                "case_b_raw_near_zero",
                "case_c_pooling_induced_zero",
                "case_d_mapping_error",
                "case_e_nonfinite",
            )
        }
        null_count = (
            counts["case_a_exact_zero"] + counts["case_b_raw_near_zero"]
        )
        fatal_count = (
            counts["case_c_pooling_induced_zero"]
            + counts["case_d_mapping_error"]
            + counts["case_e_nonfinite"]
        )
        finite_denominator = max(self.finite_element_count, 1)
        return {
            "status": "passed" if fatal_count == 0 else "failed",
            "field_semantics": "signed",
            "pooling_semantics": "adaptive_average_preserve_sign",
            "vector_semantics": (
                "flatten_then_concatenate_9_videos_then_l2_normalize_if_norm_gt_eps"
            ),
            "similarity_semantics": "clamp(signed_cosine,0,1)",
            "num_null_function_units": null_count,
            "num_exact_zero_units": counts["case_a_exact_zero"],
            "num_raw_near_zero_units": counts["case_b_raw_near_zero"],
            "num_pooling_induced_zero_units": counts[
                "case_c_pooling_induced_zero"
            ],
            "num_mapping_errors": counts["case_d_mapping_error"],
            "num_nonfinite_units": counts["case_e_nonfinite"],
            "attention_null_count": sum(
                unit.unit_type == "attention_head"
                and unit.classification
                in {"case_a_exact_zero", "case_b_raw_near_zero"}
                for unit in self.units
            ),
            "mlp_null_count": sum(
                unit.unit_type == "ffn_neuron"
                and unit.classification
                in {"case_a_exact_zero", "case_b_raw_near_zero"}
                for unit in self.units
            ),
            "layers_with_null_function_units": sorted(
                {
                    unit.layer_name
                    for unit in self.units
                    if unit.classification
                    in {"case_a_exact_zero", "case_b_raw_near_zero"}
                }
            ),
            "null_global_indices": sorted(
                unit.global_index
                for unit in self.units
                if unit.classification
                in {"case_a_exact_zero", "case_b_raw_near_zero"}
            ),
            "normalization_eps": NORMALIZATION_EPS,
            "unit_null_test": (
                "complete signed pooled and concatenated [9*16*7*7] vector "
                "L2 <= normalization_eps"
            ),
            "null_interpretation": (
                "no reliably measurable functional pattern under the current "
                "functional calibration set"
            ),
            "elementwise_zero_ratio_used_for_null_classification": False,
            "total_element_count": self.total_element_count,
            "finite_element_count": self.finite_element_count,
            "negative_element_count": self.negative_element_count,
            "exact_zero_element_count": self.exact_zero_element_count,
            "global_min": self.global_min,
            "global_max": self.global_max,
            "global_negative_element_ratio": (
                self.negative_element_count / finite_denominator
            ),
            "global_exact_zero_element_ratio": (
                self.exact_zero_element_count / finite_denominator
            ),
        }

    def write(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        _atomic_csv(
            output_dir / "zero_field_units.csv",
            ZERO_FIELD_CSV_FIELDS,
            (unit.csv_row() for unit in self.units),
        )
        _atomic_json(output_dir / "zero_field_summary.json", self.summary())

    def require_valid_null_semantics(self) -> None:
        summary = self.summary()
        if summary["status"] == "passed":
            return
        affected = [
            f"{unit.global_index}:{unit.layer_name}[{unit.unit_index}]="
            f"{unit.classification}"
            for unit in self.units
            if unit.classification
            in {
                "case_c_pooling_induced_zero",
                "case_d_mapping_error",
                "case_e_nonfinite",
            }
        ]
        raise RuntimeError(
            "Task014 functional-energy audit failed; pooling-induced zeros, "
            "mapping errors and non-finite fields are fatal: "
            + ", ".join(affected)
        )


class LayerWiseContributionFieldArchive:
    """Header-first, one-layer-at-a-time reader for a large CSTC NPZ.

    NumPy cannot memory-map compressed members inside an NPZ.  This reader
    therefore inspects all member headers through ``zipfile`` without loading
    payloads and materializes only one ``[S,U,T,H,W]`` layer at a time.  The
    deterministic aligned vectors are written once to a true ``.npy`` memmap.
    """

    def __init__(self, npz_path: str | os.PathLike[str]):
        self.path = Path(npz_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Contribution Field NPZ not found: {self.path}")

        self.metadata_path = self.path.with_name("cstc_probe_metadata.json")
        self.metadata: Mapping[str, object] = {}
        self.metadata_sha256: str | None = None
        if self.metadata_path.is_file():
            raw = self.metadata_path.read_bytes()
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError(f"Expected JSON object in {self.metadata_path}")
            self.metadata = decoded
            self.metadata_sha256 = _sha256_bytes(raw)

        headers: dict[str, ArrayHeader] = {}
        with zipfile.ZipFile(self.path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.endswith(".npy"):
                    continue
                key = member.filename[:-4]
                if key in headers:
                    raise ValueError(f"Duplicate NPZ array key: {key!r}")
                with archive.open(member, "r") as handle:
                    shape, fortran_order, dtype = _read_npy_header(handle)
                if dtype.hasobject:
                    raise ValueError(f"Object dtype is forbidden for {key!r}")
                headers[key] = ArrayHeader(
                    key=key,
                    shape=shape,
                    dtype=dtype.str,
                    fortran_order=fortran_order,
                )
        if not headers:
            raise ValueError(f"No NPY arrays found in {self.path}")
        self.headers = headers

    def _metadata_entry(self, prefix: str) -> Mapping[str, object]:
        entry = self.metadata.get(prefix)
        return entry if isinstance(entry, dict) else {}

    def _resolve_layer(self, prefix: str) -> str:
        entry = self._metadata_entry(prefix)
        layer = entry.get("layer")
        if isinstance(layer, str) and layer:
            return layer
        if not prefix.startswith("layer_"):
            return prefix
        raise ValueError(
            f"Indexed cache prefix {prefix!r} requires a complete layer entry "
            f"in {self.metadata_path}"
        )

    def _metadata_unit_type(self, prefix: str) -> str | None:
        entry = self._metadata_entry(prefix)
        for key in ("unit_type", "module_type", "type"):
            if key in entry:
                return _normalized_unit_type(entry[key])
        return None

    def audit_descriptor_mapping(
        self,
        unit_info: Sequence[Mapping[str, object]],
        valid_layer_names: Sequence[str],
        model_unit_types: Mapping[str, str],
        *,
        expected_total_units: int = EXPECTED_DESCRIPTOR_UNITS,
        expected_layer_count: int = EXPECTED_PRUNING_LAYERS,
        expected_array_count: int = EXPECTED_NPZ_ARRAYS,
        expected_video_fields: int = EXPECTED_VIDEO_FIELDS,
    ) -> MappingAudit:
        """Assert a strict descriptor-row to NPZ-field bijection.

        ``unit_info`` has length ``N`` and each row contains ``layer`` and
        zero-based ``idx``.  The resulting mapping is
        ``global descriptor index -> layer index -> unit index -> [S,T,H,W]``.
        No absent or duplicate unit is tolerated.
        """
        if len(self.headers) != int(expected_array_count):
            raise ValueError(
                f"Expected {expected_array_count} NPZ arrays, found {len(self.headers)}"
            )
        layer_names = [str(name) for name in valid_layer_names]
        layer_name_set = set(layer_names)
        if len(layer_names) != len(set(layer_names)):
            raise ValueError("valid_layer_names contains duplicates")
        if len(layer_names) != int(expected_layer_count):
            raise ValueError(
                f"Expected {expected_layer_count} pruning layers, found {len(layer_names)}"
            )

        volume_entries: dict[str, tuple[str, str, ArrayHeader]] = {}
        for key, header in self.headers.items():
            if not key.endswith(_VOLUME_SUFFIX):
                continue
            prefix = key[: -len(_VOLUME_SUFFIX)]
            layer = self._resolve_layer(prefix)
            if layer in volume_entries:
                raise ValueError(
                    f"Multiple contribution arrays resolve to layer {layer!r}"
                )
            volume_entries[layer] = (prefix, key, header)
        if len(volume_entries) != int(expected_layer_count):
            raise ValueError(
                f"Expected {expected_layer_count} contribution layers, found "
                f"{len(volume_entries)}"
            )
        if set(volume_entries) != layer_name_set:
            missing = sorted(layer_name_set - set(volume_entries))
            extra = sorted(set(volume_entries) - layer_name_set)
            raise ValueError(
                f"Contribution layer mismatch: missing={missing}, extra={extra}"
            )

        descriptor_by_layer: dict[str, list[tuple[int, int]]] = defaultdict(list)
        seen_units: set[tuple[str, int]] = set()
        duplicate = 0
        for global_index, row in enumerate(unit_info):
            layer = str(row.get("layer"))
            if layer not in layer_name_set:
                raise ValueError(
                    f"Descriptor row {global_index} references unknown layer {layer!r}"
                )
            unit_index = int(row.get("idx", -1))
            pair = (layer, unit_index)
            if pair in seen_units:
                duplicate += 1
            seen_units.add(pair)
            descriptor_by_layer[layer].append((unit_index, global_index))
        if duplicate:
            raise ValueError(f"Descriptor mapping contains {duplicate} duplicate units")

        layer_specs: list[LayerFieldSpec] = []
        unit_mappings: list[UnitFieldMapping] = []
        cache_units = 0
        expected_global = 0
        for layer_index, layer in enumerate(layer_names):
            prefix, key, header = volume_entries[layer]
            indexed_prefix = re.fullmatch(r"layer_(\d+)", prefix)
            if indexed_prefix is not None and int(indexed_prefix.group(1)) != layer_index:
                raise ValueError(
                    f"Cache layer index mismatch for {layer!r}: prefix {prefix!r} "
                    f"vs descriptor layer_index={layer_index}"
                )
            metadata_entry = self._metadata_entry(prefix)
            if "layer_index" in metadata_entry and int(
                metadata_entry["layer_index"]
            ) != layer_index:
                raise ValueError(
                    f"Cache metadata layer_index mismatch for {layer!r}: "
                    f"{metadata_entry['layer_index']} vs {layer_index}"
                )
            if len(header.shape) != 5:
                raise ValueError(
                    f"{key} must have [S,U,T,H,W], got {header.shape}"
                )
            samples, units, temporal, height, width = header.shape
            if samples != int(expected_video_fields):
                raise ValueError(
                    f"{key} has {samples} video fields; expected {expected_video_fields}"
                )
            if temporal != ALIGNED_FIELD_SHAPE[0]:
                raise ValueError(
                    f"{key} temporal axis is {temporal}; expected "
                    f"{ALIGNED_FIELD_SHAPE[0]}"
                )
            if height < ALIGNED_FIELD_SHAPE[1] or width < ALIGNED_FIELD_SHAPE[2]:
                raise ValueError(
                    f"{key} native spatial shape {(height, width)} cannot be "
                    f"downsampled to {ALIGNED_FIELD_SHAPE[1:]} without upsampling"
                )

            pairs = sorted(descriptor_by_layer[layer])
            local_indices = [unit_index for unit_index, _ in pairs]
            if local_indices != list(range(len(pairs))):
                raise ValueError(
                    f"Descriptor unit indices for {layer!r} are not contiguous "
                    "from zero"
                )
            global_indices = [global_index for _, global_index in pairs]
            expected_range = list(range(expected_global, expected_global + len(pairs)))
            if global_indices != expected_range:
                raise ValueError(
                    f"Descriptor rows for layer index {layer_index} ({layer}) are "
                    "not a contiguous layer-major range"
                )
            if units != len(pairs):
                raise ValueError(
                    f"{key} has U={units}, descriptor layer has {len(pairs)} units"
                )

            inferred_type = (
                "attention_head" if layer.endswith(".attn")
                else "ffn_neuron" if layer.endswith(".mlp")
                else None
            )
            model_type = _normalized_unit_type(model_unit_types.get(layer))
            metadata_type = self._metadata_unit_type(prefix)
            if inferred_type is None:
                raise ValueError(
                    f"Layer {layer!r} is neither an Attention nor MLP pruning module"
                )
            if model_type != inferred_type:
                raise ValueError(
                    f"Model type mismatch for {layer!r}: {model_type!r} vs "
                    f"{inferred_type!r}"
                )
            if metadata_type is not None and metadata_type != inferred_type:
                raise ValueError(
                    f"Cache metadata type mismatch for {layer!r}: "
                    f"{metadata_type!r} vs {inferred_type!r}"
                )

            # The confirmed full cache contains four arrays per pruning module.
            prefix_array_count = sum(
                key_name.startswith(prefix + "_") for key_name in self.headers
            )
            if prefix_array_count != 4:
                raise ValueError(
                    f"Cache prefix {prefix!r} has {prefix_array_count} arrays; expected 4"
                )

            spec = LayerFieldSpec(
                layer_index=layer_index,
                layer=layer,
                unit_type=inferred_type,
                cache_key=key,
                field_shape=(samples, units, temporal, height, width),
                dtype=header.dtype,
                global_start=expected_global,
                global_stop=expected_global + units,
            )
            layer_specs.append(spec)
            for unit_index, global_index in pairs:
                unit_mappings.append(
                    UnitFieldMapping(
                        global_index=global_index,
                        layer_index=layer_index,
                        layer=layer,
                        unit_type=inferred_type,
                        unit_index=unit_index,
                        cache_key=key,
                    )
                )
            expected_global += units
            cache_units += units

        descriptor_units = len(unit_info)
        mapped_units = len(unit_mappings)
        missing = descriptor_units - mapped_units
        if cache_units != int(expected_total_units):
            raise ValueError(
                f"sum(U_l)={cache_units}, expected {expected_total_units}"
            )
        if descriptor_units != int(expected_total_units):
            raise ValueError(
                f"descriptor_units={descriptor_units}, expected {expected_total_units}"
            )
        if mapped_units != descriptor_units or missing != 0 or duplicate != 0:
            raise ValueError(
                "Contribution mapping is not bijective: "
                f"mapped={mapped_units}, descriptor={descriptor_units}, "
                f"missing={missing}, duplicate={duplicate}"
            )

        mapping_lines = [
            f"L\t{spec.layer_index}\t{spec.layer}\t{spec.unit_type}\t"
            f"{spec.cache_key}\t{spec.field_shape}\t{spec.dtype}\t"
            f"{spec.global_start}\t{spec.global_stop}"
            for spec in layer_specs
        ]
        mapping_lines.extend(
            f"U\t{unit.global_index}\t{unit.layer_index}\t{unit.layer}\t"
            f"{unit.unit_type}\t{unit.unit_index}\t{unit.cache_key}"
            for unit in unit_mappings
        )
        mapping_payload = "\n".join(mapping_lines).encode("utf-8")
        stat = self.path.stat()
        return MappingAudit(
            npz_path=str(self.path),
            npz_size_bytes=int(stat.st_size),
            npz_mtime_ns=int(stat.st_mtime_ns),
            metadata_path=(
                str(self.metadata_path) if self.metadata_path.is_file() else None
            ),
            metadata_sha256=self.metadata_sha256,
            array_count=len(self.headers),
            contribution_layer_count=len(layer_specs),
            cache_units=cache_units,
            descriptor_units=descriptor_units,
            mapped_units=mapped_units,
            missing=missing,
            duplicate=duplicate,
            mapping_sha256=_sha256_bytes(mapping_payload),
            layers=layer_specs,
            units=unit_mappings,
        )

    def _load_layer_array(self, spec: LayerFieldSpec) -> np.ndarray:
        """Materialize exactly one source layer; all other NPZ members stay lazy."""
        with np.load(self.path, allow_pickle=False, mmap_mode="r") as archive:
            array = archive[spec.cache_key]
        if tuple(array.shape) != tuple(spec.field_shape):
            raise RuntimeError(
                f"{spec.cache_key} changed after audit: {array.shape} vs "
                f"{spec.field_shape}"
            )
        return array

    def audit_zero_function_fields(
        self,
        audit: MappingAudit,
        device: torch.device,
        output_dir: Path,
    ) -> ZeroFieldAudit:
        """Classify every zero/non-finite field before enabling null semantics.

        Source chunks have shape ``[S,B,T,H,W]`` and are transferred to
        ``device`` as ``[B,S,T,H,W]``.  Raw statistics and parameter-free
        pooling are computed on that device.  Only compact diagnostic scalars
        return to CPU; the NPZ remains read-only and is loaded one layer at a
        time.
        """
        mapping_by_global = {unit.global_index: unit for unit in audit.units}
        rows: list[ZeroFieldUnit] = []
        total_element_count = 0
        finite_element_count = 0
        negative_element_count = 0
        exact_zero_element_count = 0
        global_min: float | None = None
        global_max: float | None = None
        for spec in audit.layers:
            source = self._load_layer_array(spec)  # [S,U,T,H,W]
            for unit_start in range(0, spec.unit_count, _POOL_UNIT_CHUNK):
                unit_stop = min(unit_start + _POOL_UNIT_CHUNK, spec.unit_count)
                sampled = source[:, unit_start:unit_stop]
                cpu = torch.from_numpy(
                    np.ascontiguousarray(sampled, dtype=np.float32)
                )  # [S,B,T,H,W]
                fields = cpu.permute(1, 0, 2, 3, 4).contiguous().to(
                    device=device, dtype=torch.float32, non_blocking=True
                )  # [B,S,T,H,W]
                finite = torch.isfinite(fields)
                total_element_count += int(fields.numel())
                chunk_finite_count = int(finite.sum().item())
                finite_element_count += chunk_finite_count
                negative_element_count += int((finite & (fields < 0)).sum().item())
                exact_zero_element_count += int((finite & (fields == 0)).sum().item())
                if chunk_finite_count:
                    chunk_min = float(
                        torch.where(finite, fields, torch.inf).min().item()
                    )
                    chunk_max = float(
                        torch.where(finite, fields, -torch.inf).max().item()
                    )
                    global_min = (
                        chunk_min if global_min is None else min(global_min, chunk_min)
                    )
                    global_max = (
                        chunk_max if global_max is None else max(global_max, chunk_max)
                    )
                finite_by_unit = finite.reshape(fields.shape[0], -1).all(dim=1)
                raw_safe = torch.nan_to_num(
                    fields, nan=0.0, posinf=0.0, neginf=0.0
                )
                raw_flat = raw_safe.reshape(
                    raw_safe.shape[0], raw_safe.shape[1], -1
                )
                # Absolute sums are zero-audit diagnostics only.  They never
                # replace the signed field used by pooling or similarity.
                raw_abs_per_video = raw_flat.abs().sum(dim=2)  # [B,S]
                raw_l2_per_video = torch.linalg.vector_norm(
                    raw_flat, ord=2, dim=2
                )  # [B,S]
                pooled = F.adaptive_avg_pool3d(
                    raw_safe.reshape(
                        raw_safe.shape[0] * raw_safe.shape[1],
                        1,
                        raw_safe.shape[2],
                        raw_safe.shape[3],
                        raw_safe.shape[4],
                    ),
                    ALIGNED_FIELD_SHAPE,
                ).reshape(
                    raw_safe.shape[0],
                    raw_safe.shape[1],
                    *ALIGNED_FIELD_SHAPE,
                )
                pooled_flat = pooled.reshape(pooled.shape[0], pooled.shape[1], -1)
                pooled_abs_per_video = pooled_flat.abs().sum(dim=2)  # [B,S]
                pooled_l2_per_video = torch.linalg.vector_norm(
                    pooled_flat, ord=2, dim=2
                )  # [B,S]
                concatenated_l2 = torch.linalg.vector_norm(
                    pooled.reshape(pooled.shape[0], -1), ord=2, dim=1
                )  # [B]

                for chunk_index, unit_index in enumerate(
                    range(unit_start, unit_stop)
                ):
                    global_index = spec.global_start + unit_index
                    mapping = mapping_by_global.get(global_index)
                    mapping_ok = mapping is not None and (
                        mapping.layer_index == spec.layer_index
                        and mapping.layer == spec.layer
                        and mapping.unit_type == spec.unit_type
                        and mapping.unit_index == unit_index
                        and mapping.cache_key == spec.cache_key
                    )
                    unit_finite = bool(finite_by_unit[chunk_index].item())
                    vector_l2 = float(concatenated_l2[chunk_index].item())
                    if unit_finite and mapping_ok and vector_l2 > NORMALIZATION_EPS:
                        continue

                    if unit_finite:
                        raw_abs_values = tuple(
                            float(value)
                            for value in raw_abs_per_video[chunk_index].tolist()
                        )
                        raw_l2_values = tuple(
                            float(value)
                            for value in raw_l2_per_video[chunk_index].tolist()
                        )
                        pooled_abs_values = tuple(
                            float(value)
                            for value in pooled_abs_per_video[chunk_index].tolist()
                        )
                        pooled_l2_values = tuple(
                            float(value)
                            for value in pooled_l2_per_video[chunk_index].tolist()
                        )
                        raw_unit = raw_safe[chunk_index]
                        raw_min = float(raw_unit.min().item())
                        raw_max = float(raw_unit.max().item())
                        raw_abs_sum = float(sum(raw_abs_values))
                        raw_l2 = float(
                            torch.linalg.vector_norm(
                                raw_unit.reshape(-1), ord=2
                            ).item()
                        )
                        is_raw_all_zero = all(value == 0.0 for value in raw_abs_values)
                        is_pooled_all_zero = all(
                            value == 0.0 for value in pooled_abs_values
                        )
                    else:
                        empty_video_values = (None,) * EXPECTED_VIDEO_FIELDS
                        raw_abs_values = empty_video_values
                        raw_l2_values = empty_video_values
                        pooled_abs_values = empty_video_values
                        pooled_l2_values = empty_video_values
                        raw_min = raw_max = raw_abs_sum = raw_l2 = None
                        vector_l2 = None
                        is_raw_all_zero = False
                        is_pooled_all_zero = False

                    if not unit_finite:
                        classification = "case_e_nonfinite"
                    elif not mapping_ok:
                        classification = "case_d_mapping_error"
                    elif is_raw_all_zero:
                        classification = "case_a_exact_zero"
                    elif raw_l2 <= NORMALIZATION_EPS:
                        classification = "case_b_raw_near_zero"
                    else:
                        # Rows arrive here only when their pooled norm is at
                        # most eps, or a mapping/non-finite fatal condition was
                        # detected above.  Hence this remaining finite mapped
                        # case is a pooling-induced loss of functional energy.
                        classification = "case_c_pooling_induced_zero"
                    rows.append(
                        ZeroFieldUnit(
                            global_index=global_index,
                            layer_index=spec.layer_index,
                            layer_name=spec.layer,
                            unit_type=spec.unit_type,
                            unit_index=unit_index,
                            source_array_key=spec.cache_key,
                            raw_shape=(
                                spec.field_shape[0],
                                spec.field_shape[2],
                                spec.field_shape[3],
                                spec.field_shape[4],
                            ),
                            raw_min=raw_min,
                            raw_max=raw_max,
                            raw_abs_sum=raw_abs_sum,
                            raw_l2=raw_l2,
                            raw_abs_sum_per_video=raw_abs_values,
                            raw_l2_per_video=raw_l2_values,
                            pooled_abs_sum_per_video=pooled_abs_values,
                            pooled_l2_per_video=pooled_l2_values,
                            concatenated_l2=vector_l2,
                            is_raw_all_zero=is_raw_all_zero,
                            is_pooled_all_zero=is_pooled_all_zero,
                            classification=classification,
                        )
                    )
                del fields, finite, raw_safe, pooled
            del source

        result = ZeroFieldAudit(
            units=sorted(rows, key=lambda row: row.global_index),
            total_element_count=total_element_count,
            finite_element_count=finite_element_count,
            negative_element_count=negative_element_count,
            exact_zero_element_count=exact_zero_element_count,
            global_min=global_min,
            global_max=global_max,
        )
        result.write(output_dir)
        summary = result.summary()
        print(">>> Null Functional Unit energy audit")
        print(f">>> Null functional units: {summary['num_null_function_units']}")
        print(f">>> Exact raw-zero units: {summary['num_exact_zero_units']}")
        print(f">>> Raw near-zero units: {summary['num_raw_near_zero_units']}")
        print(
            ">>> Pooling-induced zero units: "
            f"{summary['num_pooling_induced_zero_units']}"
        )
        print(f">>> Mapping errors: {summary['num_mapping_errors']}")
        print(f">>> Non-finite units: {summary['num_nonfinite_units']}")
        print(
            ">>> Signed field range: "
            f"[{summary['global_min']}, {summary['global_max']}]"
        )
        print(
            ">>> Negative element ratio (diagnostic only): "
            f"{summary['global_negative_element_ratio']:.12f}"
        )
        print(
            ">>> Exact-zero element ratio (not a null-unit test): "
            f"{summary['global_exact_zero_element_ratio']:.12f}"
        )
        for unit in result.units:
            print(
                f"    {unit.global_index} {unit.layer_name} "
                f"{unit.unit_type} {unit.unit_index}"
            )
        if summary["status"] == "passed":
            print(">>> Null Functional Unit handling: ENABLED")
        return result

    @staticmethod
    def _pool_and_normalize(
        sampled_fields: np.ndarray, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return vectors ``[B,S*T*7*7]`` and validity mask ``Bool[B]``.

        The nine video fields are never averaged.  Each ``[T,H,W]`` field is
        pooled independently to ``[16,7,7]``.  The sample-major values are
        concatenated in their original order with its original sign.  Active
        signed rows with norm above ``NORMALIZATION_EPS`` are L2-normalized.
        Exact-zero and raw-near-zero rows remain exactly zero in this derived
        normalized representation and are marked invalid rather than having
        numerical noise magnified into a unit-length direction.
        """
        if sampled_fields.ndim != 5:
            raise ValueError(
                f"sampled_fields must be [S,B,T,H,W], got {sampled_fields.shape}"
            )
        samples, batch_units, temporal, height, width = sampled_fields.shape
        if samples != EXPECTED_VIDEO_FIELDS:
            raise ValueError(
                f"Expected {EXPECTED_VIDEO_FIELDS} videos, received {samples}"
            )
        cpu = torch.from_numpy(
            np.ascontiguousarray(sampled_fields, dtype=np.float32)
        )  # [S,B,T,H,W]
        fields = cpu.permute(1, 0, 2, 3, 4).contiguous().to(
            device=device, dtype=torch.float32, non_blocking=True
        )  # [B,S,T,H,W]
        if not torch.isfinite(fields).all():
            raise ValueError("Contribution Fields contain NaN or infinity")
        pooled = F.adaptive_avg_pool3d(
            fields.reshape(batch_units * samples, 1, temporal, height, width),
            ALIGNED_FIELD_SHAPE,
        ).reshape(batch_units, samples, *ALIGNED_FIELD_SHAPE)
        vectors = pooled.reshape(batch_units, -1)  # [B,9*16*7*7]
        norms = torch.linalg.vector_norm(vectors, ord=2, dim=1)  # [B]
        valid_function_mask = norms > NORMALIZATION_EPS  # Bool[B]
        normalized = torch.zeros_like(vectors)
        if torch.any(valid_function_mask):
            normalized[valid_function_mask] = (
                vectors[valid_function_mask]
                / norms[valid_function_mask, None]
            )
        return normalized, valid_function_mask

    def _vector_cache_identity(self, audit: MappingAudit) -> dict[str, object]:
        return {
            "format_version": "task014_signed_energy_null_mask_v3",
            "source_npz": audit.npz_path,
            "source_size_bytes": audit.npz_size_bytes,
            "source_mtime_ns": audit.npz_mtime_ns,
            "source_metadata_sha256": audit.metadata_sha256,
            "mapping_sha256": audit.mapping_sha256,
            "descriptor_units": audit.descriptor_units,
            "video_fields": EXPECTED_VIDEO_FIELDS,
            "aligned_field_shape": list(ALIGNED_FIELD_SHAPE),
            "vector_width": EXPECTED_VIDEO_FIELDS * math.prod(ALIGNED_FIELD_SHAPE),
            "dtype": "float32",
            "valid_function_mask_dtype": "bool",
            "sample_aggregation": "none",
            "field_semantics": "signed",
            "pooling_semantics": "adaptive_average_preserve_sign",
            "similarity_semantics": "clamp(signed_cosine,0,1)",
        }

    def prepare_vector_memmap(
        self,
        audit: MappingAudit,
        cache_dir: Path,
        device: torch.device,
    ) -> tuple[Path, Path]:
        """Create/reuse aligned vectors ``[N,D]`` and validity mask ``Bool[N]``.

        Source payloads are read one layer at a time.  Pooling, finite checks,
        non-negativity checks, flattening, and normalization run on ``device``
        (normally ``cuda:0``).  Only the resulting float32 vectors are copied
        back for the mmap file; the 4.5GB source is never expanded globally.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vector_path = cache_dir / "aligned_function_fields.npy"
        mask_path = cache_dir / "aligned_function_valid_mask.npy"
        metadata_path = cache_dir / "aligned_function_fields.json"
        identity = self._vector_cache_identity(audit)
        vector_width = int(identity["vector_width"])

        if vector_path.is_file() and mask_path.is_file() and metadata_path.is_file():
            try:
                saved = json.loads(metadata_path.read_text(encoding="utf-8"))
                mapped = np.load(vector_path, mmap_mode="r", allow_pickle=False)
                valid_mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
                compatible = saved == identity and tuple(mapped.shape) == (
                    audit.descriptor_units,
                    vector_width,
                ) and mapped.dtype == np.float32 and tuple(valid_mask.shape) == (
                    audit.descriptor_units,
                ) and valid_mask.dtype == np.bool_
                del mapped, valid_mask
                if compatible:
                    return vector_path, mask_path
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        temporary = cache_dir / "aligned_function_fields.npy.tmp"
        mask_temporary = cache_dir / "aligned_function_valid_mask.npy.tmp"
        for path in (temporary, mask_temporary):
            if path.exists():
                path.unlink()
        mapped = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(audit.descriptor_units, vector_width),
        )
        mapped_mask = np.lib.format.open_memmap(
            mask_temporary,
            mode="w+",
            dtype=np.bool_,
            shape=(audit.descriptor_units,),
        )
        try:
            for spec in audit.layers:
                source = self._load_layer_array(spec)  # [S,U,T,H,W], one layer
                for unit_start in range(0, spec.unit_count, _POOL_UNIT_CHUNK):
                    unit_stop = min(unit_start + _POOL_UNIT_CHUNK, spec.unit_count)
                    vectors, valid_mask = self._pool_and_normalize(
                        source[:, unit_start:unit_stop], device
                    )  # [B,9*16*7*7], Bool[B]
                    global_start = spec.global_start + unit_start
                    global_stop = spec.global_start + unit_stop
                    mapped[global_start:global_stop] = (
                        vectors.detach().cpu().numpy()
                    )
                    mapped_mask[global_start:global_stop] = (
                        valid_mask.detach().cpu().numpy()
                    )
                    del vectors, valid_mask
                del source
                mapped.flush()
                mapped_mask.flush()
        except Exception:
            del mapped, mapped_mask
            for path in (temporary, mask_temporary):
                if path.exists():
                    path.unlink()
            raise
        del mapped, mapped_mask
        temporary.replace(vector_path)
        mask_temporary.replace(mask_path)
        _atomic_json(metadata_path, identity)
        return vector_path, mask_path

    def load_domain_vectors_direct(
        self,
        audit: MappingAudit,
        global_indices: Sequence[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one domain as vectors ``[G,D]`` and validity mask ``Bool[G]``."""
        indices = [int(index) for index in global_indices]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("A preflight domain must contain unique units")
        if min(indices) < 0 or max(indices) >= audit.descriptor_units:
            raise IndexError("Domain contains a global index outside the audit")

        output = torch.empty(
            (len(indices), EXPECTED_VIDEO_FIELDS * math.prod(ALIGNED_FIELD_SHAPE)),
            dtype=torch.float32,
            device=device,
        )  # [G,9*16*7*7]
        valid_output = torch.empty(
            (len(indices),), dtype=torch.bool, device=device
        )  # Bool[G]
        positions_by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
        unit_map = {unit.global_index: unit for unit in audit.units}
        for position, global_index in enumerate(indices):
            unit = unit_map[global_index]
            positions_by_layer[unit.layer_index].append((position, unit.unit_index))

        for layer_index, position_units in sorted(positions_by_layer.items()):
            spec = audit.layers[layer_index]
            source = self._load_layer_array(spec)
            for start in range(0, len(position_units), _POOL_UNIT_CHUNK):
                rows = position_units[start:start + _POOL_UNIT_CHUNK]
                positions = [position for position, _ in rows]
                unit_indices = [unit_index for _, unit_index in rows]
                vectors, valid_mask = self._pool_and_normalize(
                    source[:, unit_indices], device
                )
                position_tensor = torch.tensor(positions, device=device)
                output[position_tensor] = vectors
                valid_output[position_tensor] = valid_mask
            del source
        return output, valid_output


def validate_domain_partition(
    groups: Sequence[Sequence[int]], num_units: int
) -> list[list[int]]:
    """Assert every global pruning unit belongs to exactly one BMS domain."""
    normalized: list[list[int]] = []
    seen: set[int] = set()
    duplicates: list[int] = []
    for domain_id, group in enumerate(groups):
        members = sorted(int(index) for index in group)
        if not members:
            raise ValueError(f"BMS domain {domain_id} is empty")
        if len(members) != len(set(members)):
            raise ValueError(f"BMS domain {domain_id} contains duplicate members")
        for index in members:
            if index < 0 or index >= num_units:
                raise IndexError(
                    f"BMS domain {domain_id} index {index} outside [0,{num_units})"
                )
            if index in seen:
                duplicates.append(index)
            seen.add(index)
        normalized.append(members)
    missing = sorted(set(range(num_units)) - seen)
    if duplicates or missing:
        raise ValueError(
            f"BMS domains are not a partition: duplicates={duplicates[:10]}, "
            f"missing={missing[:10]}"
        )
    return normalized


def build_functional_similarity(
    normalized_vectors: torch.Tensor,
    valid_function_mask: torch.Tensor,
    *,
    row_chunk: int = _SIMILARITY_ROW_CHUNK,
) -> torch.Tensor:
    """Return domain-only cosine matrix ``A`` with shape ``[G,G]``.

    ``normalized_vectors`` is signed ``[G,D]``, where
    ``D = 9 * 16 * 7 * 7``.
    ``valid_function_mask`` is Boolean ``[G]``.  Active rows have unit L2 norm;
    null-functional rows are exact zeros.  Any row or column involving a null
    unit remains zero, including its diagonal.  Signed cosine is converted to
    same-direction similarity only at the final matrix step as
    ``A_ij = clamp(cosine_ij, 0, 1)``; field absolute values, cosine absolute
    values and affine cosine shifts are forbidden.  Only row chunks of the
    domain matrix are computed; no global ``[N,N]`` allocation is made.
    """
    vectors = torch.as_tensor(normalized_vectors)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError(
            f"normalized_vectors must be non-empty [G,D], got {tuple(vectors.shape)}"
        )
    if not torch.isfinite(vectors).all():
        raise ValueError("normalized_vectors contains NaN or infinity")
    valid = torch.as_tensor(
        valid_function_mask, dtype=torch.bool, device=vectors.device
    )  # Bool[G]
    if valid.ndim != 1 or valid.numel() != vectors.shape[0]:
        raise ValueError("valid_function_mask must have shape [G]")
    norms = torch.linalg.vector_norm(vectors, ord=2, dim=1)
    if torch.any(valid) and not torch.allclose(
        norms[valid], torch.ones_like(norms[valid]), rtol=1e-4, atol=1e-5
    ):
        raise ValueError("Active normalized_vectors rows must have unit L2 norm")
    if torch.any(~valid) and not torch.allclose(
        norms[~valid], torch.zeros_like(norms[~valid]), rtol=0.0, atol=0.0
    ):
        raise ValueError("Null-functional normalized_vectors rows must be zero")
    group_size = vectors.shape[0]
    similarity = torch.empty(
        (group_size, group_size), dtype=torch.float32, device=vectors.device
    )  # [G,G]
    vectors = vectors.to(dtype=torch.float32)
    for start in range(0, group_size, int(row_chunk)):
        stop = min(start + int(row_chunk), group_size)
        similarity[start:stop] = vectors[start:stop] @ vectors.T
    similarity.clamp_(min=0.0, max=1.0)
    diagonal = torch.arange(group_size, device=similarity.device)
    similarity[diagonal, diagonal] = valid.to(dtype=similarity.dtype)
    if not torch.isfinite(similarity).all():
        raise ValueError("Functional similarity contains NaN or infinity")
    return similarity


def functional_coverage(
    similarity: torch.Tensor,
    retained: Iterable[int] | torch.Tensor,
    valid_function_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute active-demand coverage from ``A[G,G]`` and retained set ``S``.

    ``valid_function_mask`` is Boolean ``[G]``.  Only active rows create
    functional demand and only retained active columns can represent it.  An
    all-null domain has coverage one for every nonempty retained set.
    """
    matrix = torch.as_tensor(similarity)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"similarity must be square [G,G], got {tuple(matrix.shape)}")
    if isinstance(retained, torch.Tensor) and retained.dtype == torch.bool:
        if retained.numel() != matrix.shape[0]:
            raise ValueError("retained mask size does not match domain")
        indices = retained.nonzero(as_tuple=True)[0].to(matrix.device)
    else:
        indices = torch.tensor(
            sorted({int(index) for index in retained}),
            dtype=torch.long,
            device=matrix.device,
        )
    valid = torch.as_tensor(
        valid_function_mask, dtype=torch.bool, device=matrix.device
    )  # Bool[G]
    if valid.ndim != 1 or valid.numel() != matrix.shape[0]:
        raise ValueError("valid_function_mask must have shape [G]")
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= matrix.shape[0]
    ):
        raise IndexError("retained index outside domain")
    demand_rows = valid.nonzero(as_tuple=True)[0]  # [G+]
    if demand_rows.numel() == 0:
        return matrix.new_tensor(1.0)
    if indices.numel() == 0:
        return matrix.new_tensor(0.0)
    active_representatives = indices[valid.index_select(0, indices)]  # [R+]
    if active_representatives.numel() == 0:
        return matrix.new_tensor(0.0)
    return (
        matrix.index_select(0, demand_rows)
        .index_select(1, active_representatives)
        .max(dim=1)
        .values.mean()
    )


def marginal_coverage_losses(
    similarity: torch.Tensor,
    retained_mask: torch.Tensor,
    valid_function_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact ``Delta_i(S)`` for retained units using best/second-best.

    ``similarity`` is ``A in R^[G,G]``; ``retained_mask`` and
    ``valid_function_mask`` are Boolean ``[G]``.  Losses are ``[G]`` with
    ``+inf`` at removed positions and zero at retained null positions.  Only
    active demand rows and retained active representatives enter the
    best/second-best identity.  Coverage is scalar ``C(S) in R``.
    """
    matrix = torch.as_tensor(similarity)
    mask = torch.as_tensor(retained_mask, dtype=torch.bool, device=matrix.device)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity must have shape [G,G]")
    if mask.ndim != 1 or mask.numel() != matrix.shape[0]:
        raise ValueError("retained_mask must have shape [G]")
    valid = torch.as_tensor(
        valid_function_mask, dtype=torch.bool, device=matrix.device
    )  # Bool[G]
    if valid.ndim != 1 or valid.numel() != matrix.shape[0]:
        raise ValueError("valid_function_mask must have shape [G]")
    retained = mask.nonzero(as_tuple=True)[0]  # [R]
    if retained.numel() == 0:
        raise ValueError("At least one domain representative must be retained")
    losses = matrix.new_zeros((matrix.shape[0],), dtype=torch.float32)  # [G]
    demand_rows = valid.nonzero(as_tuple=True)[0]  # [G+]
    if demand_rows.numel() == 0:
        losses.masked_fill_(~mask, torch.inf)
        return losses, matrix.new_tensor(1.0)
    active_retained = (mask & valid).nonzero(as_tuple=True)[0]  # [R+]
    if active_retained.numel() == 0:
        losses.masked_fill_(~mask, torch.inf)
        return losses, matrix.new_tensor(0.0)

    current = (
        matrix.index_select(0, demand_rows)
        .index_select(1, active_retained)
        .clone()
    )  # [G+,R+]
    best_values, best_positions = current.max(dim=1)  # [G+], [G+]
    coverage = best_values.mean()
    if active_retained.numel() == 1:
        second_values = torch.zeros_like(best_values)
    else:
        current.scatter_(1, best_positions[:, None], -torch.inf)
        second_values = current.max(dim=1).values
    best_representatives = active_retained[best_positions]  # [G+]
    losses.scatter_add_(
        0,
        best_representatives,
        (best_values - second_values).clamp_min_(0.0).to(torch.float32),
    )
    losses /= demand_rows.numel()
    losses.masked_fill_(~mask, torch.inf)
    if not torch.isfinite(losses[mask]).all() or not torch.isfinite(coverage):
        raise ValueError("Marginal functional loss is not finite")
    return losses, coverage


def validate_functional_score_mode(value: str) -> str:
    """Return a canonical Task016 functional score mode."""
    mode = str(value).strip().lower()
    aliases = {"average": "domain_average", "total": "domain_total"}
    mode = aliases.get(mode, mode)
    if mode not in FUNCTIONAL_SCORE_MODES:
        raise ValueError(
            f"functional_score must be one of {FUNCTIONAL_SCORE_MODES}, got {value!r}"
        )
    return mode


def domain_total_losses(
    average_losses: torch.Tensor,
    valid_function_mask: torch.Tensor,
) -> torch.Tensor:
    """Remove Task014's fixed active-demand averaging factor.

    ``average_losses`` and ``valid_function_mask`` have shape ``[G]``.  The
    active functional demand set ``G+`` is the immutable non-null mask used by
    Task014's coverage denominator.  Therefore ``L_i=|G+|*Delta_i`` is exactly
    the direct summed coverage loss, while removed ``+inf`` entries remain
    ``+inf`` even for an all-null domain.
    """
    losses = torch.as_tensor(average_losses)
    valid = torch.as_tensor(
        valid_function_mask, dtype=torch.bool, device=losses.device
    )
    if losses.ndim != 1 or valid.shape != losses.shape:
        raise ValueError("average_losses and valid_function_mask must have shape [G]")
    active_domain_size = int(valid.sum().item())
    scaled = losses * float(active_domain_size)
    return torch.where(torch.isfinite(losses), scaled, losses)


def direct_summed_coverage_losses(
    similarity: torch.Tensor,
    retained_mask: torch.Tensor,
    valid_function_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference ``sum_j(max_S A[j]-max_(S\\i) A[j])`` for tests/audits.

    This deliberately direct implementation is not used by production
    selection.  All tensors use domain axes ``[G,G]`` or ``[G]``.
    """
    matrix = torch.as_tensor(similarity)
    retained = torch.as_tensor(
        retained_mask, dtype=torch.bool, device=matrix.device
    )
    valid = torch.as_tensor(
        valid_function_mask, dtype=torch.bool, device=matrix.device
    )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity must have shape [G,G]")
    if retained.shape != valid.shape or retained.shape != (matrix.shape[0],):
        raise ValueError("retained and valid masks must have shape [G]")
    if not bool(retained.any().item()):
        raise ValueError("At least one domain representative must be retained")
    output = matrix.new_full((matrix.shape[0],), torch.inf, dtype=torch.float32)
    demand = valid.nonzero(as_tuple=True)[0]
    if demand.numel() == 0:
        output[retained] = 0.0
        return output
    active_representatives = (retained & valid).nonzero(as_tuple=True)[0]
    if active_representatives.numel():
        current = (
            matrix.index_select(0, demand)
            .index_select(1, active_representatives)
            .max(dim=1)
            .values
        )
    else:
        current = matrix.new_zeros((demand.numel(),))
    for local_index in retained.nonzero(as_tuple=True)[0].tolist():
        if not bool(valid[local_index].item()):
            output[local_index] = 0.0
            continue
        after = retained.clone()
        after[local_index] = False
        after_active = (after & valid).nonzero(as_tuple=True)[0]
        if after_active.numel():
            after_values = (
                matrix.index_select(0, demand)
                .index_select(1, after_active)
                .max(dim=1)
                .values
            )
        else:
            after_values = matrix.new_zeros((demand.numel(),))
        output[local_index] = (current - after_values).sum().to(torch.float32)
    return output


@dataclass
class DomainState:
    """Mutable retained set for one fixed BMS domain similarity matrix."""

    domain_id: int
    global_indices: list[int]
    similarity: torch.Tensor
    valid_function_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.global_indices != sorted(self.global_indices):
            raise ValueError("Domain global indices must be sorted")
        if len(self.global_indices) != len(set(self.global_indices)):
            raise ValueError("Domain global indices must be unique")
        if tuple(self.similarity.shape) != (
            len(self.global_indices), len(self.global_indices)
        ):
            raise ValueError("Domain similarity shape does not match membership")
        self.valid_function_mask = torch.as_tensor(
            self.valid_function_mask,
            dtype=torch.bool,
            device=self.similarity.device,
        )  # Bool[G]
        if self.valid_function_mask.ndim != 1 or self.valid_function_mask.numel() != len(
            self.global_indices
        ):
            raise ValueError("Domain valid_function_mask must have shape [G]")
        self.retained = torch.ones(
            len(self.global_indices), dtype=torch.bool, device=self.similarity.device
        )  # [G]
        self.version = 0
        self.removed_losses: list[float] = []
        self.losses, coverage = marginal_coverage_losses(
            self.similarity, self.retained, self.valid_function_mask
        )
        self.initial_coverage = float(coverage.item())
        self.current_coverage = self.initial_coverage
        if not math.isclose(
            self.initial_coverage, 1.0, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError(
                f"Domain {self.domain_id} initial coverage is "
                f"{self.initial_coverage}, expected approximately 1"
            )

    @property
    def initial_size(self) -> int:
        return len(self.global_indices)

    @property
    def retained_count(self) -> int:
        return int(self.retained.sum().item())

    def best_feasible_candidate(
        self, feasible: Callable[[int], bool]
    ) -> tuple[float, int, int] | None:
        """Return ``(loss, global_index, local_index)`` with deterministic ties."""
        feasible_local = [
            local_index
            for local_index, global_index in enumerate(self.global_indices)
            if bool(self.retained[local_index].item()) and feasible(global_index)
        ]
        if not feasible_local:
            return None
        # Local membership is sorted by global index. torch.min returns the
        # first position for an exact loss tie, hence the smaller global index.
        local_tensor = torch.tensor(
            feasible_local, dtype=torch.long, device=self.similarity.device
        )
        candidate_losses = self.losses.index_select(0, local_tensor)
        loss, position = candidate_losses.min(dim=0)
        local_index = feasible_local[int(position.item())]
        return (
            float(loss.item()),
            self.global_indices[local_index],
            local_index,
        )

    def remove(self, local_index: int) -> dict[str, object]:
        if local_index < 0 or local_index >= self.initial_size:
            raise IndexError("local domain index is out of range")
        if not bool(self.retained[local_index].item()):
            raise ValueError("candidate has already been removed")
        before_losses = self.losses.detach().clone()
        coverage_before = self.current_coverage
        selected_loss = float(before_losses[local_index].item())
        self.retained[local_index] = False
        self.version += 1
        self.removed_losses.append(selected_loss)
        if self.retained_count:
            self.losses, coverage = marginal_coverage_losses(
                self.similarity, self.retained, self.valid_function_mask
            )
            self.current_coverage = float(coverage.item())
        else:
            self.losses = torch.full_like(before_losses, torch.inf)
            self.current_coverage = (
                1.0 if not bool(self.valid_function_mask.any().item()) else 0.0
            )
        return {
            "coverage_before": coverage_before,
            "coverage_after": self.current_coverage,
            "selected_loss": selected_loss,
            "losses_before": before_losses,
            "losses_after": self.losses.detach().clone(),
        }


def _domain_state_from_vectors(
    domain_id: int,
    members: Sequence[int],
    vectors: torch.Tensor,
    valid_function_mask: torch.Tensor,
) -> DomainState:
    similarity = build_functional_similarity(vectors, valid_function_mask)
    return DomainState(
        domain_id, list(members), similarity, valid_function_mask
    )


def run_one_domain_numerical_preflight(
    archive: LayerWiseContributionFieldArchive,
    audit: MappingAudit,
    zero_field_audit: ZeroFieldAudit,
    groups: Sequence[Sequence[int]],
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    """Validate a real null-containing BMS domain before full selection."""
    domains = validate_domain_partition(groups, audit.descriptor_units)
    null_global_indices = {
        unit.global_index
        for unit in zero_field_audit.units
        if unit.classification
        in {"case_a_exact_zero", "case_b_raw_near_zero"}
    }
    if null_global_indices:
        eligible = [
            index
            for index, members in enumerate(domains)
            if null_global_indices.intersection(members)
        ]
        if not eligible:
            raise ValueError("Zero-field audit units are absent from all BMS domains")
        domain_id = eligible[0]
    else:
        domain_id = next(
            (index for index, members in enumerate(domains) if len(members) > 1),
            0,
        )
    members = domains[domain_id]
    vectors, valid_function_mask = archive.load_domain_vectors_direct(
        audit, members, device
    )
    state = _domain_state_from_vectors(
        domain_id, members, vectors, valid_function_mask
    )
    finite_losses = state.losses[state.retained]
    null_local = (~state.valid_function_mask).nonzero(as_tuple=True)[0]
    active_local = state.valid_function_mask.nonzero(as_tuple=True)[0]
    payload = {
        "status": "passed",
        "field_semantics": "signed",
        "similarity_definition": "clamp(signed_cosine,0,1)",
        "domain_id": domain_id,
        "domain_size": len(members),
        "device": str(state.similarity.device),
        "similarity_shape": list(state.similarity.shape),
        "similarity_min": float(state.similarity.min().item()),
        "similarity_max": float(state.similarity.max().item()),
        "initial_coverage": state.initial_coverage,
        "active_functional_units": int(active_local.numel()),
        "null_functional_units": int(null_local.numel()),
        "minimum_marginal_loss": float(finite_losses.min().item()),
        "maximum_marginal_loss": float(finite_losses.max().item()),
        "all_finite": bool(torch.isfinite(finite_losses).all().item()),
    }
    if not payload["all_finite"]:
        raise ValueError("Real-domain preflight produced non-finite marginal loss")
    if not 0.0 <= payload["similarity_min"] <= payload["similarity_max"] <= 1.0:
        raise ValueError("Real-domain preflight similarity is outside [0,1]")

    null_payload: dict[str, object]
    if null_local.numel():
        diagonal = torch.diagonal(state.similarity)
        null_diagonal_zero = bool(
            torch.equal(
                diagonal.index_select(0, null_local),
                torch.zeros_like(diagonal.index_select(0, null_local)),
            )
        )
        active_diagonal_one = bool(
            active_local.numel() == 0
            or torch.allclose(
                diagonal.index_select(0, active_local),
                torch.ones_like(diagonal.index_select(0, active_local)),
                rtol=0.0,
                atol=0.0,
            )
        )
        null_losses_zero = bool(
            torch.allclose(
                state.losses.index_select(0, null_local),
                torch.zeros_like(state.losses.index_select(0, null_local)),
                rtol=0.0,
                atol=0.0,
            )
        )
        active_losses_finite = bool(
            active_local.numel() == 0
            or torch.isfinite(state.losses.index_select(0, active_local)).all().item()
        )
        deleted_local = int(null_local[0].item())
        coverage_before = state.current_coverage
        deletion = state.remove(deleted_local)
        coverage_after = state.current_coverage
        coverage_unchanged = math.isclose(
            coverage_before, coverage_after, rel_tol=0.0, abs_tol=1e-7
        )
        null_payload = {
            "status": "passed",
            "field_semantics": "signed",
            "similarity_definition": "clamp(signed_cosine,0,1)",
            "domain_id": domain_id,
            "domain_size": len(members),
            "active_functional_units": int(active_local.numel()),
            "null_functional_units": int(null_local.numel()),
            "all_null_domain": bool(active_local.numel() == 0),
            "null_diagonal_similarity_zero": null_diagonal_zero,
            "active_diagonal_similarity_one": active_diagonal_one,
            "null_marginal_losses_zero": null_losses_zero,
            "active_marginal_losses_finite": active_losses_finite,
            "initial_coverage": payload["initial_coverage"],
            "deleted_null_global_index": members[deleted_local],
            "deleted_null_marginal_loss": deletion["selected_loss"],
            "coverage_before_null_deletion": coverage_before,
            "coverage_after_null_deletion": coverage_after,
            "coverage_unchanged_after_null_deletion": coverage_unchanged,
        }
        required = (
            null_diagonal_zero,
            active_diagonal_one,
            null_losses_zero,
            active_losses_finite,
            coverage_unchanged,
            math.isclose(
                float(payload["initial_coverage"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5,
            ),
        )
        if not all(required):
            null_payload["status"] = "failed"
            _atomic_json(
                Path(output_dir) / "null_domain_numerical_preflight.json",
                null_payload,
            )
            raise ValueError(
                "Null-containing real-domain numerical preflight failed"
            )
    else:
        null_payload = {
            "status": "not_applicable",
            "field_semantics": "signed",
            "similarity_definition": "clamp(signed_cosine,0,1)",
            "reason": "zero-field audit found no null-functional units",
            "domain_id": domain_id,
            "domain_size": len(members),
        }
    _atomic_json(Path(output_dir) / "one_domain_numerical_preflight.json", payload)
    _atomic_json(
        Path(output_dir) / "null_domain_numerical_preflight.json", null_payload
    )
    return payload


@dataclass
class FunctionalSelectionResult:
    registry: dict[str, set[int]]
    removed_parameter_cost: int
    target_parameter_budget: float
    budget_overshoot: float
    trace: list[dict[str, object]]
    domain_summary: list[dict[str, object]]
    set_dependency_examples: list[dict[str, object]]
    similarity_devices: dict[str, int]
    score_mode: str = "domain_average"
    type_competition_trace: list[dict[str, object]] | None = None
    domain_calibration_statistics: list[dict[str, object]] | None = None
    initial_candidate_scores: list[dict[str, object]] | None = None
    min_keep_rejections: list[dict[str, object]] | None = None
    budget_skips: list[dict[str, object]] | None = None


_CANDIDATE_ALL = 0
_CANDIDATE_ATTENTION = 1
_CANDIDATE_FFN = 2
_CANDIDATE_KINDS = 3


class _FunctionalDomainCandidateCache:
    """CUDA-backed best-candidate cache with shape ``[3,K]``.

    Axis zero is all/Attention/FFN and axis one is BMS domain.  Full marginal
    arrays remain on their domain device.  Refreshing one domain transfers
    only three average losses, three total losses and three local indices.
    """

    def __init__(
        self,
        states: Sequence[DomainState],
        unit_records: Mapping[int, UnitFieldMapping],
        capacities: Mapping[str, int],
    ) -> None:
        self.states = list(states)
        self.unit_records = unit_records
        self.layer_names = sorted(capacities)
        self.layer_to_id = {
            layer: index for index, layer in enumerate(self.layer_names)
        }
        self.capacities = np.asarray(
            [capacities[layer] for layer in self.layer_names], dtype=np.int64
        )  # [L]
        self.pruned = np.zeros(len(self.layer_names), dtype=np.int64)  # [L]
        self.domain_globals: list[np.ndarray] = []
        self.domain_layer_ids: list[torch.Tensor] = []
        self.domain_attention: list[torch.Tensor] = []
        self.domain_ffn: list[torch.Tensor] = []
        self.layer_domains: list[set[int]] = [set() for _ in self.layer_names]
        self.feasible_by_device: dict[str, torch.Tensor] = {}
        self.active_size = np.asarray(
            [int(state.valid_function_mask.sum().item()) for state in self.states],
            dtype=np.int64,
        )  # [K], fixed Task014 active-demand counts
        for state in self.states:
            device = state.similarity.device
            device_key = str(device)
            if device_key not in self.feasible_by_device:
                self.feasible_by_device[device_key] = torch.as_tensor(
                    self.capacities > 0, dtype=torch.bool, device=device
                )  # Bool[L]
            globals_ = np.asarray(state.global_indices, dtype=np.int64)  # [G]
            layers = np.asarray(
                [
                    self.layer_to_id[self.unit_records[int(index)].layer]
                    for index in globals_
                ],
                dtype=np.int64,
            )  # [G]
            types = [self.unit_records[int(index)].unit_type for index in globals_]
            self.domain_globals.append(globals_)
            self.domain_layer_ids.append(
                torch.as_tensor(layers, dtype=torch.long, device=device)
            )
            self.domain_attention.append(
                torch.as_tensor(
                    [value == "attention_head" for value in types],
                    dtype=torch.bool,
                    device=device,
                )
            )
            self.domain_ffn.append(
                torch.as_tensor(
                    [value == "ffn_neuron" for value in types],
                    dtype=torch.bool,
                    device=device,
                )
            )
            for layer_id in np.unique(layers):
                self.layer_domains[int(layer_id)].add(state.domain_id)
        domains = len(self.states)
        self.average = np.full(
            (_CANDIDATE_KINDS, domains), np.inf, dtype=np.float32
        )
        self.total = np.full(
            (_CANDIDATE_KINDS, domains), np.inf, dtype=np.float32
        )
        self.global_index = np.full(
            (_CANDIDATE_KINDS, domains), -1, dtype=np.int64
        )
        self.local_index = np.full(
            (_CANDIDATE_KINDS, domains), -1, dtype=np.int64
        )

    def eligible_mask(self, domain_id: int) -> torch.Tensor:
        state = self.states[domain_id]
        feasible = self.feasible_by_device[str(state.similarity.device)].index_select(
            0, self.domain_layer_ids[domain_id]
        )  # Bool[G]
        return state.retained & feasible  # Bool[G]

    def refresh(self, domain_id: int) -> None:
        state = self.states[domain_id]
        eligible = self.eligible_mask(domain_id)
        masks = torch.stack(
            (
                eligible,
                eligible & self.domain_attention[domain_id],
                eligible & self.domain_ffn[domain_id],
            ),
            dim=0,
        )  # Bool[3,G]
        average = state.losses.unsqueeze(0).expand(
            _CANDIDATE_KINDS, -1
        ).masked_fill(~masks, torch.inf)  # [3,G]
        average_values, local_indices = average.min(dim=1)  # [3], [3]
        total_values = torch.where(
            torch.isfinite(average_values),
            average_values * float(self.active_size[domain_id]),
            average_values,
        )  # [3]
        packet = torch.cat(
            (
                average_values.to(torch.float32),
                total_values.to(torch.float32),
                local_indices.to(torch.float32),
            )
        ).detach().cpu().numpy()  # float32[9], one device synchronization
        average_cpu = packet[:_CANDIDATE_KINDS]
        total_cpu = packet[_CANDIDATE_KINDS:2 * _CANDIDATE_KINDS]
        local_cpu = packet[2 * _CANDIDATE_KINDS:].astype(np.int64)
        global_cpu = np.full(_CANDIDATE_KINDS, -1, dtype=np.int64)
        for kind in range(_CANDIDATE_KINDS):
            if np.isfinite(average_cpu[kind]):
                global_cpu[kind] = self.domain_globals[domain_id][local_cpu[kind]]
            else:
                local_cpu[kind] = -1
        self.average[:, domain_id] = average_cpu
        self.total[:, domain_id] = total_cpu
        self.global_index[:, domain_id] = global_cpu
        self.local_index[:, domain_id] = local_cpu

    def refresh_all(self) -> None:
        for domain_id in range(len(self.states)):
            self.refresh(domain_id)

    def best(
        self, kind: int, score_mode: str
    ) -> tuple[float, int, int, int] | None:
        mode = validate_functional_score_mode(score_mode)
        scores = self.average[kind] if mode == "domain_average" else self.total[kind]
        finite = np.flatnonzero(np.isfinite(scores) & (self.global_index[kind] >= 0))
        if finite.size == 0:
            return None
        order = np.lexsort(
            (self.global_index[kind, finite], scores[finite].astype(np.float64))
        )
        domain_id = int(finite[int(order[0])])
        return (
            float(scores[domain_id]),
            int(self.global_index[kind, domain_id]),
            domain_id,
            int(self.local_index[kind, domain_id]),
        )

    def mark_removed(self, layer: str) -> tuple[bool, set[int]]:
        layer_id = self.layer_to_id[layer]
        self.pruned[layer_id] += 1
        if self.pruned[layer_id] > self.capacities[layer_id]:
            raise RuntimeError(f"Layer {layer} exceeded its Task014 min-keep capacity")
        became_full = self.pruned[layer_id] == self.capacities[layer_id]
        if became_full:
            for feasible in self.feasible_by_device.values():
                feasible[layer_id] = False
            return True, set(self.layer_domains[layer_id])
        return False, set()


def _finite_candidate_payload(
    candidate: tuple[float, int, int, int] | None,
) -> tuple[float, int]:
    if candidate is None:
        return math.inf, -1
    return float(candidate[0]), int(candidate[1])


def _quantile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: math.nan for key in ("mean", "median", "q25", "q75", "q90")}
    q25, median, q75, q90 = np.quantile(array, (0.25, 0.5, 0.75, 0.9))
    return {
        "mean": float(array.mean()),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
    }


class FunctionalCompetitionPruner:
    """Global dynamic selector over current per-domain marginal losses."""

    def __init__(
        self,
        *,
        groups: Sequence[Sequence[int]],
        audit: MappingAudit,
        vector_memmap: Path,
        vector_mask_memmap: Path,
        unit_costs: Sequence[int],
        max_prunable_by_layer: Mapping[str, int],
        target_parameter_budget: float,
        preferred_device: torch.device,
        functional_score: str = "domain_average",
        total_original_parameters: int | None = None,
    ):
        self.audit = audit
        self.groups = validate_domain_partition(groups, audit.descriptor_units)
        if len(unit_costs) != audit.descriptor_units:
            raise ValueError("unit_costs length does not match mapped units")
        self.unit_costs = [int(cost) for cost in unit_costs]
        if any(cost <= 0 for cost in self.unit_costs):
            raise ValueError("Every pruning unit must have positive parameter cost")
        self.capacities = {
            str(layer): int(value)
            for layer, value in max_prunable_by_layer.items()
        }
        if any(value < 0 for value in self.capacities.values()):
            raise ValueError("Layer pruning capacities must be non-negative")
        self.target_budget = float(target_parameter_budget)
        if self.target_budget <= 0.0:
            raise ValueError("target_parameter_budget must be positive")
        self.preferred_device = torch.device(preferred_device)
        self.functional_score = validate_functional_score_mode(functional_score)
        self.total_original_parameters = int(
            total_original_parameters
            if total_original_parameters is not None
            else math.ceil(self.target_budget)
        )
        if self.total_original_parameters <= 0:
            raise ValueError("total_original_parameters must be positive")
        self.vector_memmap = Path(vector_memmap)
        self.vector_mask_memmap = Path(vector_mask_memmap)
        self.unit_records = {unit.global_index: unit for unit in audit.units}
        self.states: list[DomainState] = []

    def _build_state(
        self,
        domain_id: int,
        members: Sequence[int],
        mapped_vectors: np.ndarray,
        mapped_valid_mask: np.ndarray,
    ) -> DomainState:
        cpu_vectors = torch.from_numpy(
            np.asarray(mapped_vectors[list(members)], dtype=np.float32).copy()
        )  # [G,9*16*7*7]
        cpu_valid_mask = torch.from_numpy(
            np.asarray(mapped_valid_mask[list(members)], dtype=np.bool_).copy()
        )  # Bool[G]
        try:
            vectors = cpu_vectors.to(
                device=self.preferred_device, dtype=torch.float32, non_blocking=True
            )
            valid_mask = cpu_valid_mask.to(
                device=self.preferred_device, dtype=torch.bool, non_blocking=True
            )
            state = _domain_state_from_vectors(
                domain_id, members, vectors, valid_mask
            )
            del vectors
            return state
        except RuntimeError as error:
            if (
                self.preferred_device.type != "cuda"
                or "out of memory" not in str(error).lower()
            ):
                raise
            if "vectors" in locals():
                del vectors
            torch.cuda.empty_cache()
            print(
                f"  Warning: domain {domain_id} size {len(members)} exceeded "
                "cuda:0 memory; using documented CPU similarity fallback"
            )
            return _domain_state_from_vectors(
                domain_id, members, cpu_vectors, cpu_valid_mask
            )

    def _initialize_states(self) -> None:
        vectors = np.load(self.vector_memmap, mmap_mode="r", allow_pickle=False)
        valid_mask = np.load(
            self.vector_mask_memmap, mmap_mode="r", allow_pickle=False
        )
        expected_shape = (
            self.audit.descriptor_units,
            EXPECTED_VIDEO_FIELDS * math.prod(ALIGNED_FIELD_SHAPE),
        )
        if tuple(vectors.shape) != expected_shape or vectors.dtype != np.float32:
            raise ValueError(
                f"Aligned vector mmap has {vectors.shape}/{vectors.dtype}, "
                f"expected {expected_shape}/float32"
            )
        if tuple(valid_mask.shape) != (self.audit.descriptor_units,) or (
            valid_mask.dtype != np.bool_
        ):
            raise ValueError(
                f"Aligned valid-function mmap has {valid_mask.shape}/"
                f"{valid_mask.dtype}, expected "
                f"({self.audit.descriptor_units},)/bool"
            )
        self.states = [
            self._build_state(domain_id, members, vectors, valid_mask)
            for domain_id, members in enumerate(self.groups)
        ]
        del vectors, valid_mask

    def select(self) -> FunctionalSelectionResult:
        """Select with Task015's domain-cached GPU engine until budget stop."""
        self._initialize_states()
        registry = {layer: set() for layer in self.capacities}
        removed_cost = 0
        trace: list[dict[str, object]] = []
        type_trace: list[dict[str, object]] = []
        calibration_statistics: list[dict[str, object]] = []
        initial_candidate_scores: list[dict[str, object]] = []
        min_keep_rejections: list[dict[str, object]] = []
        budget_skips: list[dict[str, object]] = []
        dependency_examples: list[dict[str, object]] = []
        candidates = _FunctionalDomainCandidateCache(
            self.states, self.unit_records, self.capacities
        )
        candidates.refresh_all()
        target_sparsity = self.target_budget / float(self.total_original_parameters)
        snapshot_targets = [
            value for value in (0.0, 0.05, 0.10, 0.20, 0.30)
            if value <= target_sparsity + 1e-12
        ]
        next_snapshot = 0

        def candidate_snapshot(progress: float) -> None:
            nonlocal initial_candidate_scores
            rows: list[dict[str, object]] = []
            for domain_id, state in enumerate(self.states):
                local = candidates.eligible_mask(domain_id).nonzero(
                    as_tuple=True
                )[0]  # [F_k]
                if local.numel() == 0:
                    continue
                average = state.losses.index_select(0, local)  # [F_k]
                total = average * float(candidates.active_size[domain_id])  # [F_k]
                packet = torch.stack((average, total), dim=0).detach().cpu().numpy()
                local_cpu = local.detach().cpu().numpy().astype(np.int64, copy=False)
                for position, local_index in enumerate(local_cpu):
                    global_index = state.global_indices[int(local_index)]
                    unit = self.unit_records[global_index]
                    rows.append(
                        {
                            "progress_sparsity": progress,
                            "global_index": global_index,
                            "domain_id": domain_id,
                            "layer": unit.layer,
                            "unit_type": unit.unit_type,
                            "unit_index": unit.unit_index,
                            "domain_active_size": int(candidates.active_size[domain_id]),
                            "delta_average": float(packet[0, position]),
                            "delta_total": float(packet[1, position]),
                        }
                    )
            if progress == 0.0:
                initial_candidate_scores = rows
            for unit_type in ("attention_head", "ffn_neuron"):
                typed = [row for row in rows if row["unit_type"] == unit_type]
                size_summary = _quantile_summary(
                    [float(row["domain_active_size"]) for row in typed]
                )
                average_summary = _quantile_summary(
                    [float(row["delta_average"]) for row in typed]
                )
                total_summary = _quantile_summary(
                    [float(row["delta_total"]) for row in typed]
                )
                calibration_statistics.append(
                    {
                        "progress_sparsity": progress,
                        "unit_type": unit_type,
                        "candidate_count": len(typed),
                        **{f"active_domain_size_{key}": value for key, value in size_summary.items()},
                        **{f"delta_average_{key}": value for key, value in average_summary.items()},
                        **{f"delta_total_{key}": value for key, value in total_summary.items()},
                    }
                )

        candidate_snapshot(0.0)
        next_snapshot = 1
        print(">>> Functional selection started")
        print(f">>> Functional score: {self.functional_score}")
        started = time.perf_counter()
        last_heartbeat = started
        removed_attention = 0
        removed_ffn = 0
        while removed_cost < self.target_budget:
            selected = candidates.best(_CANDIDATE_ALL, self.functional_score)
            if selected is None:
                raise RuntimeError(
                    "No feasible functional candidate remains before the global "
                    f"parameter budget was reached ({removed_cost} < "
                    f"{self.target_budget})"
                )
            selected_score, global_index, domain_id, local_index = selected
            state = self.states[domain_id]
            unit = self.unit_records[global_index]
            retained_before = state.retained_count
            budget_before = removed_cost
            delta_average = float(state.losses[local_index].item())
            domain_active_size = int(candidates.active_size[domain_id])
            delta_total = delta_average * domain_active_size
            best_attention_average = candidates.best(
                _CANDIDATE_ATTENTION, "domain_average"
            )
            best_ffn_average = candidates.best(_CANDIDATE_FFN, "domain_average")
            best_attention_total = candidates.best(
                _CANDIDATE_ATTENTION, "domain_total"
            )
            best_ffn_total = candidates.best(_CANDIDATE_FFN, "domain_total")
            update = state.remove(local_index)
            cost = self.unit_costs[global_index]
            registry[unit.layer].add(unit.unit_index)
            removed_cost += cost
            step = len(trace) + 1
            if unit.unit_type == "attention_head":
                removed_attention += 1
            else:
                removed_ffn += 1
            trace.append(
                {
                    "step": step,
                    "budget_progress_before": budget_before / self.target_budget,
                    "budget_progress_after": removed_cost / self.target_budget,
                    "global_index": global_index,
                    "layer": unit.layer,
                    "stage": unit.layer.split(".")[1] if "." in unit.layer else "",
                    "unit_type": unit.unit_type,
                    "unit_index": unit.unit_index,
                    "is_null_functional": not bool(
                        state.valid_function_mask[local_index].item()
                    ),
                    "domain_id": domain_id,
                    "domain_size_initial": state.initial_size,
                    "domain_retained_before": retained_before,
                    "domain_active_size": domain_active_size,
                    "delta_average": delta_average,
                    "delta_total": delta_total,
                    "functional_score_mode": self.functional_score,
                    "functional_score": selected_score,
                    "marginal_functional_loss": delta_average,
                    "parameter_cost": cost,
                    "cumulative_removed_parameters": removed_cost,
                    "estimated_parameter_sparsity": (
                        removed_cost / float(self.total_original_parameters)
                    ),
                    "coverage_before": update["coverage_before"],
                    "coverage_after": update["coverage_after"],
                    "selected": True,
                }
            )
            attention_average_value, attention_average_index = _finite_candidate_payload(
                best_attention_average
            )
            ffn_average_value, ffn_average_index = _finite_candidate_payload(
                best_ffn_average
            )
            attention_total_value, attention_total_index = _finite_candidate_payload(
                best_attention_total
            )
            ffn_total_value, ffn_total_index = _finite_candidate_payload(
                best_ffn_total
            )
            type_trace.append(
                {
                    "step": step,
                    "estimated_sparsity": removed_cost / float(self.total_original_parameters),
                    "best_attention_delta_average": attention_average_value,
                    "best_ffn_delta_average": ffn_average_value,
                    "best_attention_delta_total": attention_total_value,
                    "best_ffn_delta_total": ffn_total_value,
                    "best_attention_global_index": (
                        attention_average_index
                        if self.functional_score == "domain_average"
                        else attention_total_index
                    ),
                    "best_ffn_global_index": (
                        ffn_average_index
                        if self.functional_score == "domain_average"
                        else ffn_total_index
                    ),
                    "selected_type": unit.unit_type,
                    "selected_global_index": global_index,
                }
            )

            if state.retained_count and len(dependency_examples) < 128:
                before_losses = update["losses_before"]
                after_losses = update["losses_after"]
                retained_local = state.retained.nonzero(as_tuple=True)[0]
                changes = after_losses[retained_local] - before_losses[retained_local]
                maximum_change, position = changes.max(dim=0)
                if float(maximum_change.item()) > 0.0:
                    survivor_local = int(retained_local[int(position.item())].item())
                    survivor_global = state.global_indices[survivor_local]
                    survivor = self.unit_records[survivor_global]
                    dependency_examples.append(
                        {
                            "domain_id": domain_id,
                            "step": step,
                            "global_index": survivor_global,
                            "unit_type": survivor.unit_type,
                            "marginal_loss_before": float(
                                before_losses[survivor_local].item()
                            ),
                            "marginal_loss_after_partner_removal": float(
                                after_losses[survivor_local].item()
                            ),
                            "change": float(maximum_change.item()),
                        }
                    )
            became_full, affected_domains = candidates.mark_removed(unit.layer)
            if became_full:
                rejected = 0
                for affected_domain in affected_domains:
                    affected_state = self.states[affected_domain]
                    for affected_local, affected_global in enumerate(
                        affected_state.global_indices
                    ):
                        if (
                            self.unit_records[affected_global].layer == unit.layer
                            and bool(affected_state.retained[affected_local].item())
                        ):
                            rejected += 1
                min_keep_rejections.append(
                    {
                        "step": step,
                        "layer": unit.layer,
                        "unit_type": unit.unit_type,
                        "max_prunable": self.capacities[unit.layer],
                        "rejected_retained_candidates": rejected,
                        "reason": "existing_min_keep_capacity_reached",
                    }
                )
                for affected_domain in affected_domains:
                    candidates.refresh(affected_domain)
            else:
                candidates.refresh(domain_id)

            estimated_sparsity = removed_cost / float(self.total_original_parameters)
            while (
                next_snapshot < len(snapshot_targets)
                and estimated_sparsity + 1e-12 >= snapshot_targets[next_snapshot]
            ):
                candidate_snapshot(float(snapshot_targets[next_snapshot]))
                next_snapshot += 1

            now = time.perf_counter()
            if now - last_heartbeat >= 10.0 or removed_cost >= self.target_budget:
                elapsed = max(now - started, 1e-9)
                steps_per_second = step / elapsed
                mean_cost = removed_cost / step
                remaining_steps = max(
                    (self.target_budget - removed_cost) / max(mean_cost, 1.0), 0.0
                )
                eta_seconds = remaining_steps / max(steps_per_second, 1e-9)
                best_attention = candidates.best(
                    _CANDIDATE_ATTENTION, self.functional_score
                )
                best_ffn = candidates.best(_CANDIDATE_FFN, self.functional_score)
                print(
                    f"  target={target_sparsity:.1%} cost={removed_cost} "
                    f"estimated={estimated_sparsity:.2%} selected={unit.unit_type} "
                    f"attention={removed_attention} ffn={removed_ffn} "
                    f"best_attention={_finite_candidate_payload(best_attention)[0]:.6g} "
                    f"best_ffn={_finite_candidate_payload(best_ffn)[0]:.6g} "
                    f"steps/s={steps_per_second:.2f} ETA={eta_seconds:.0f}s"
                )
                last_heartbeat = now

        domain_summary = []
        for state in self.states:
            records = [self.unit_records[index] for index in state.global_indices]
            attention_count = sum(
                record.unit_type == "attention_head" for record in records
            )
            removed_count = state.initial_size - state.retained_count
            null_count = int((~state.valid_function_mask).sum().item())
            domain_summary.append(
                {
                    "domain_id": state.domain_id,
                    "initial_size": state.initial_size,
                    "attention_count": attention_count,
                    "mlp_count": state.initial_size - attention_count,
                    "active_functional_count": state.initial_size - null_count,
                    "null_functional_count": null_count,
                    "mixed_type": bool(
                        attention_count and attention_count < state.initial_size
                    ),
                    "initial_coverage": state.initial_coverage,
                    "final_retained_count": state.retained_count,
                    "final_coverage": state.current_coverage,
                    "coverage_loss": state.initial_coverage - state.current_coverage,
                    "removed_count": removed_count,
                    "mean_removed_marginal_loss": (
                        sum(state.removed_losses) / len(state.removed_losses)
                        if state.removed_losses else 0.0
                    ),
                    "max_removed_marginal_loss": (
                        max(state.removed_losses) if state.removed_losses else 0.0
                    ),
                }
            )

        devices: dict[str, int] = defaultdict(int)
        for state in self.states:
            devices[str(state.similarity.device)] += 1
        return FunctionalSelectionResult(
            registry=registry,
            removed_parameter_cost=removed_cost,
            target_parameter_budget=self.target_budget,
            budget_overshoot=removed_cost - self.target_budget,
            trace=trace,
            domain_summary=domain_summary,
            set_dependency_examples=dependency_examples,
            similarity_devices=dict(devices),
            score_mode=self.functional_score,
            type_competition_trace=type_trace,
            domain_calibration_statistics=calibration_statistics,
            initial_candidate_scores=initial_candidate_scores,
            min_keep_rejections=min_keep_rejections,
            budget_skips=budget_skips,
        )


FUNCTIONAL_TRACE_FIELDS = (
    "step",
    "budget_progress_before",
    "budget_progress_after",
    "global_index",
    "layer",
    "stage",
    "unit_type",
    "unit_index",
    "is_null_functional",
    "domain_id",
    "domain_size_initial",
    "domain_retained_before",
    "domain_active_size",
    "delta_average",
    "delta_total",
    "functional_score_mode",
    "functional_score",
    "marginal_functional_loss",
    "parameter_cost",
    "cumulative_removed_parameters",
    "estimated_parameter_sparsity",
    "coverage_before",
    "coverage_after",
    "selected",
)

DOMAIN_SUMMARY_FIELDS = (
    "domain_id",
    "initial_size",
    "attention_count",
    "mlp_count",
    "active_functional_count",
    "null_functional_count",
    "mixed_type",
    "initial_coverage",
    "final_retained_count",
    "final_coverage",
    "coverage_loss",
    "removed_count",
    "mean_removed_marginal_loss",
    "max_removed_marginal_loss",
)

SET_DEPENDENCY_FIELDS = (
    "domain_id",
    "step",
    "global_index",
    "unit_type",
    "marginal_loss_before",
    "marginal_loss_after_partner_removal",
    "change",
)

TYPE_COMPETITION_FIELDS = (
    "step",
    "estimated_sparsity",
    "best_attention_delta_average",
    "best_ffn_delta_average",
    "best_attention_delta_total",
    "best_ffn_delta_total",
    "best_attention_global_index",
    "best_ffn_global_index",
    "selected_type",
    "selected_global_index",
)

DOMAIN_CALIBRATION_FIELDS = (
    "progress_sparsity",
    "unit_type",
    "candidate_count",
    "active_domain_size_mean",
    "active_domain_size_median",
    "active_domain_size_q25",
    "active_domain_size_q75",
    "active_domain_size_q90",
    "delta_average_mean",
    "delta_average_median",
    "delta_average_q25",
    "delta_average_q75",
    "delta_average_q90",
    "delta_total_mean",
    "delta_total_median",
    "delta_total_q25",
    "delta_total_q75",
    "delta_total_q90",
)

INITIAL_CANDIDATE_FIELDS = (
    "progress_sparsity",
    "global_index",
    "domain_id",
    "layer",
    "unit_type",
    "unit_index",
    "domain_active_size",
    "delta_average",
    "delta_total",
)

MIN_KEEP_REJECTION_FIELDS = (
    "step",
    "layer",
    "unit_type",
    "max_prunable",
    "rejected_retained_candidates",
    "reason",
)

BUDGET_SKIP_FIELDS = (
    "step",
    "global_index",
    "parameter_cost",
    "remaining_budget",
    "reason",
)


def write_functional_selection_artifacts(
    output_dir: Path, result: FunctionalSelectionResult
) -> None:
    output_dir = Path(output_dir)
    _atomic_csv(
        output_dir / "functional_selection_trace.csv",
        FUNCTIONAL_TRACE_FIELDS,
        result.trace,
    )
    _atomic_csv(
        output_dir / "functional_domain_summary.csv",
        DOMAIN_SUMMARY_FIELDS,
        result.domain_summary,
    )
    _atomic_csv(
        output_dir / "set_dependency_examples.csv",
        SET_DEPENDENCY_FIELDS,
        result.set_dependency_examples,
    )
    _atomic_csv(
        output_dir / "type_competition_trace.csv",
        TYPE_COMPETITION_FIELDS,
        result.type_competition_trace or [],
    )
    _atomic_csv(
        output_dir / "domain_calibration_statistics.csv",
        DOMAIN_CALIBRATION_FIELDS,
        result.domain_calibration_statistics or [],
    )
    _atomic_csv(
        output_dir / "initial_candidate_scores.csv",
        INITIAL_CANDIDATE_FIELDS,
        result.initial_candidate_scores or [],
    )
    _atomic_csv(
        output_dir / "min_keep_rejections.csv",
        MIN_KEEP_REJECTION_FIELDS,
        result.min_keep_rejections or [],
    )
    _atomic_csv(
        output_dir / "budget_skip_trace.csv",
        BUDGET_SKIP_FIELDS,
        result.budget_skips or [],
    )
    _atomic_json(
        output_dir / "functional_selection_summary.json",
        {
            "status": "complete",
            "target_parameter_budget": result.target_parameter_budget,
            "removed_parameter_cost": result.removed_parameter_cost,
            "budget_overshoot": result.budget_overshoot,
            "functional_score": result.score_mode,
            "domain_size_semantics": (
                "fixed non-null functional demand count from Task014 coverage"
            ),
            "parameter_cost_in_ranking": False,
            "budget_policy": "select feasible unit, then accumulate actual parameter cost",
            "budget_skipped_candidates": len(result.budget_skips or []),
            "type_specific_rule": False,
            "new_method_hyperparameter": False,
            "removed_units": len(result.trace),
            "removed_null_functional_units": sum(
                bool(row["is_null_functional"]) for row in result.trace
            ),
            "null_functional_units": sum(
                int(row["null_functional_count"])
                for row in result.domain_summary
            ),
            "similarity_devices": result.similarity_devices,
            "full_fine_tuning_executed": False,
        },
    )
