"""Disk-backed signed Contribution Field archive and Task037 coverage math."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .slowfast_unit_adapter import UnitInventory


def normalize_fields(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Oracle helper: concatenate all videos per unit, then apply one L2 norm."""
    raw = np.asarray(fields, dtype=np.float32)
    if raw.ndim != 5:
        raise ValueError("fields must be [N,C,T,H,W]")
    flat = raw.transpose(1, 0, 2, 3, 4).reshape(raw.shape[1], -1)
    norms = np.linalg.norm(flat, axis=1).astype(np.float32)
    valid = norms > 1e-12
    normalized = np.zeros_like(flat, dtype=np.float32)
    np.divide(flat, norms[:, None], out=normalized, where=valid[:, None])
    return normalized, valid


@dataclass
class FieldManifestEntry:
    layer_name: str
    global_start: int
    global_end: int
    shape: list[int]
    fields_path: str
    valid_path: str


class ContributionFieldArchive:
    def __init__(self, root: str | Path, inventory: UnitInventory):
        self.root = Path(root)
        self.inventory = inventory
        manifest_path = self.root / "field_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def create(
        root: str | Path,
        inventory: UnitInventory,
        fields_by_layer: dict[str, np.ndarray],
        sample_identity: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for layer_name, fields in fields_by_layer.items():
            units = [u for u in inventory.units if u.module_name == layer_name]
            if len(units) != int(fields.shape[1]):
                raise ValueError(f"field width mismatch for {layer_name}")
            raw = np.asarray(fields, dtype=np.float32)
            if raw.ndim != 5:
                raise ValueError("fields must be [N,C,T,H,W]")
            valid = (np.linalg.norm(
                raw.reshape(raw.shape[0], raw.shape[1], -1), axis=2
            ).astype(np.float32) > 0.0)
            stem = layer_name.replace(".", "__")
            fpath = root / f"{stem}.fields.npy"
            vpath = root / f"{stem}.valid.npy"
            np.save(fpath, raw, allow_pickle=False)
            np.save(vpath, valid.astype(np.bool_), allow_pickle=False)
            entries.append(
                {
                    "layer_name": layer_name,
                    "global_start": units[0].global_index,
                    "global_end": units[-1].global_index + 1,
                    "shape": list(raw.shape),
                    "fields_path": fpath.name,
                    "valid_path": vpath.name,
                    "normalization": "raw signed pooled float32; one L2 after cross-video concatenation",
                }
            )
        manifest = {
            "field_semantics": "signed true-class raw logit X*d z_y/dX",
            "sample_count": len(sample_identity),
            "pooled_shape": [16, 7, 7],
            "feature_dimension": len(sample_identity) * 16 * 7 * 7,
            "unit_count": inventory.num_units,
            "entries": entries,
            "sample_identity": list(sample_identity),
        }
        (root / "field_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (root / "sample_identity.json").write_text(
            json.dumps(list(sample_identity), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    def materialize_aligned(self) -> Path:
        """Build one aligned mmap by streaming layers, never raw fields."""
        target = self.root / "aligned_vectors.npy"
        valid_target = self.root / "aligned_valid.npy"
        if target.exists() and valid_target.exists():
            return target
        sample_count = int(self.manifest["sample_count"])
        feature_dim = sample_count * 16 * 7 * 7
        aligned = np.lib.format.open_memmap(
            target, mode="w+", dtype=np.float32,
            shape=(self.inventory.num_units, feature_dim)
        )
        valid_aligned = np.lib.format.open_memmap(
            valid_target, mode="w+", dtype=np.bool_,
            shape=(self.inventory.num_units,)
        )
        for entry in self.manifest["entries"]:
            start, end = int(entry["global_start"]), int(entry["global_end"])
            array = np.load(self.root / entry["fields_path"], mmap_mode="r", allow_pickle=False)
            valid = np.load(self.root / entry["valid_path"], mmap_mode="r", allow_pickle=False)
            block = np.asarray(array, dtype=np.float32).transpose(1, 0, 2, 3, 4).reshape(end - start, feature_dim)
            aligned[start:end] = block
            valid_aligned[start:end] = np.asarray(valid, dtype=np.bool_).any(axis=0)
        aligned.flush()
        valid_aligned.flush()
        return target

    def _entry(self, layer_name: str) -> dict[str, Any]:
        for entry in self.manifest["entries"]:
            if entry["layer_name"] == layer_name:
                return entry
        raise KeyError(layer_name)

    def load_vectors(
        self, global_indices: Sequence[int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wanted = {int(i) for i in global_indices}
        if not wanted:
            raise ValueError("empty domain")
        aligned_path = self.materialize_aligned()
        if aligned_path.exists():
            aligned = np.load(aligned_path, mmap_mode="r", allow_pickle=False)
            valid_mmap = np.load(self.root / "aligned_valid.npy", mmap_mode="r", allow_pickle=False)
            order = sorted(wanted)
            array = np.asarray(aligned[order], dtype=np.float32)
            valid = torch.from_numpy(np.asarray(valid_mmap[order], dtype=np.bool_)).to(device=device)
            tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
            norms = torch.linalg.norm(tensor, dim=1)
            tensor = torch.where(valid[:, None], tensor / norms.clamp_min(1e-12)[:, None], torch.zeros_like(tensor))
            if not torch.isfinite(tensor).all(): raise RuntimeError("non-finite normalized contribution vector")
            return tensor, valid
        by_layer: dict[str, list[Any]] = {}
        for unit in self.inventory.units:
            if unit.global_index in wanted:
                by_layer.setdefault(unit.module_name, []).append(unit)
        vectors: list[np.ndarray] = []
        valid: list[bool] = []
        order = sorted(global_indices)
        lookup = {}
        for layer_name, units in by_layer.items():
            entry = self._entry(layer_name)
            array = np.load(self.root / entry["fields_path"], mmap_mode="r")
            mask = np.load(self.root / entry["valid_path"], mmap_mode="r")
            for unit in units:
                row = np.asarray(array[:, unit.local_channel_index], dtype=np.float32)
                lookup[unit.global_index] = row.reshape(-1)
        for index in order:
            unit = self.inventory.units[index]
            row = lookup[index]
            # A function is active if at least one video has nonzero field.
            vectors.append(row)
            valid.append(bool(np.linalg.norm(row) > 1e-12))
        array = np.stack(vectors).astype(np.float32, copy=False)
        tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
        valid_tensor = torch.tensor(valid, dtype=torch.bool, device=device)
        norms = torch.linalg.norm(tensor, dim=1)
        tensor = torch.where(
            valid_tensor[:, None], tensor / norms.clamp_min(1e-12)[:, None],
            torch.zeros_like(tensor),
        )
        if not torch.isfinite(tensor).all():
            raise RuntimeError("non-finite normalized contribution vector")
        return tensor, valid_tensor


def build_functional_similarity(
    normalized_vectors: torch.Tensor, valid_function_mask: torch.Tensor
) -> torch.Tensor:
    vectors = torch.as_tensor(normalized_vectors, dtype=torch.float32)
    valid = torch.as_tensor(valid_function_mask, dtype=torch.bool, device=vectors.device)
    if vectors.ndim != 2 or valid.shape != (vectors.shape[0],):
        raise ValueError("vectors and valid mask shape mismatch")
    norms = torch.linalg.norm(vectors, dim=1)
    if bool(valid.any()) and not torch.allclose(
        norms[valid], torch.ones_like(norms[valid]), rtol=1e-4, atol=1e-5
    ):
        raise ValueError("active vectors must be normalized")
    if bool((~valid).any()) and bool((norms[~valid] != 0).any()):
        raise ValueError("null vectors must be exact zero")
    result = (vectors @ vectors.T).clamp(0.0, 1.0)
    result = result * (valid[:, None] & valid[None, :]).to(torch.float32)
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def functional_coverage(
    similarity: torch.Tensor, retained_mask: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    retained = torch.as_tensor(retained_mask, dtype=torch.bool, device=similarity.device)
    valid = torch.as_tensor(valid, dtype=torch.bool, device=similarity.device)
    demand = valid.nonzero(as_tuple=True)[0]
    reps = (retained & valid).nonzero(as_tuple=True)[0]
    if demand.numel() == 0:
        return similarity.new_tensor(1.0)
    if reps.numel() == 0:
        return similarity.new_tensor(0.0)
    return similarity[demand][:, reps].max(1).values.mean()


def marginal_coverage_losses(
    similarity: torch.Tensor, retained_mask: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    retained = torch.as_tensor(retained_mask, dtype=torch.bool, device=similarity.device)
    valid = torch.as_tensor(valid, dtype=torch.bool, device=similarity.device)
    if not bool(retained.any()):
        raise ValueError("domain must retain at least one unit")
    losses = torch.zeros(similarity.shape[0], dtype=torch.float32, device=similarity.device)
    demand = valid.nonzero(as_tuple=True)[0]
    active = (retained & valid).nonzero(as_tuple=True)[0]
    if demand.numel() == 0:
        losses[~retained] = float("inf")
        return losses, similarity.new_tensor(1.0)
    if active.numel() == 0:
        losses[~retained] = float("inf")
        return losses, similarity.new_tensor(0.0)
    values = similarity[demand][:, active].clone()
    best, pos = values.max(1)
    if active.numel() == 1:
        second = torch.zeros_like(best)
    else:
        values.scatter_(1, pos[:, None], -float("inf"))
        second = values.max(1).values
    losses.scatter_add_(0, active[pos], (best - second).clamp_min(0.0))
    losses /= demand.numel()
    losses[~retained] = float("inf")
    return losses, best.mean()


def domain_total_losses(average_losses: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    active_count = int(torch.as_tensor(valid, dtype=torch.bool).sum().item())
    # The multiplication remains float32 before any Python float conversion.
    return torch.where(
        torch.isfinite(average_losses),
        average_losses * torch.tensor(active_count, dtype=torch.float32, device=average_losses.device),
        average_losses,
    )


@dataclass
class DomainState:
    domain_id: int
    members: list[int]
    similarity: torch.Tensor
    valid: torch.Tensor
    retained: torch.Tensor
    losses: torch.Tensor
    coverage: torch.Tensor

    @classmethod
    def create(cls, domain_id, members, vectors, valid):
        similarity = build_functional_similarity(vectors, valid)
        retained = torch.ones(len(members), dtype=torch.bool, device=similarity.device)
        losses, coverage = marginal_coverage_losses(similarity, retained, valid)
        state = cls(domain_id, list(members), similarity, valid, retained, losses, coverage)
        state._demand_indices = None
        state._active_representatives = None
        state._ranked_global = None
        state._ranked_values = None
        return state

    def _initialize_incremental(self) -> None:
        demand = self.valid.nonzero(as_tuple=True)[0]
        active = self.valid.nonzero(as_tuple=True)[0]
        self._demand_indices = demand
        self._active_representatives = active
        if demand.numel() == 0 or active.numel() == 0:
            self._ranked_global = None
            self._ranked_values = None
            return
        scores = self.similarity.index_select(0, demand).index_select(1, active)
        order = torch.argsort(scores, dim=1, descending=True)
        self._ranked_global = active[order]
        self._ranked_values = scores.gather(1, order)

    def _refresh_incremental(self) -> None:
        self.losses.zero_()
        demand = self._demand_indices
        if demand.numel() == 0:
            self.losses[~self.retained] = float("inf")
            self.coverage = self.coverage.new_tensor(1.0)
            return
        ranked_global = self._ranked_global
        ranked_values = self._ranked_values
        if ranked_global is None or ranked_values is None:
            self.losses[~self.retained] = float("inf")
            self.coverage = self.coverage.new_tensor(0.0)
            return
        available = self.retained[ranked_global]
        has_best = available.any(dim=1)
        first_pos = available.to(torch.int64).argmax(dim=1)
        rows = torch.arange(demand.numel(), device=self.similarity.device)
        best_values = ranked_values[rows, first_pos]
        remaining = available.clone()
        remaining[rows, first_pos] = False
        has_second = remaining.any(dim=1)
        second_pos = remaining.to(torch.int64).argmax(dim=1)
        second_values = ranked_values[rows, second_pos]
        best_values = torch.where(has_best, best_values, torch.zeros_like(best_values))
        second_values = torch.where(has_second, second_values, torch.zeros_like(second_values))
        delta = (best_values - second_values).clamp_min(0.0)
        best_global = ranked_global[rows, first_pos]
        if bool(has_best.any()):
            self.losses.scatter_add_(
                0,
                best_global[has_best],
                delta[has_best] / float(demand.numel()),
            )
        self.losses[~self.retained] = float("inf")
        self.coverage = best_values.mean()

    def remove(self, local_index: int) -> dict[str, float]:
        if not self.retained[local_index]:
            raise ValueError("unit is already removed")
        before = self.coverage
        loss = self.losses[local_index]
        self.retained[local_index] = False
        if bool(self.retained.any()):
            if self._ranked_global is None and self._demand_indices is None:
                self._initialize_incremental()
            self._refresh_incremental()
        else:
            self.losses.fill_(float("inf"))
            self.coverage = torch.where(
                self.valid.any(), self.coverage.new_tensor(0.0), self.coverage.new_tensor(1.0)
            )
        return {
            "coverage_before": float(before.item()),
            "coverage_after": float(self.coverage.item()),
            "selected_loss": float(loss.item()),
        }


def validate_partition(groups: Sequence[Sequence[int]], num_units: int) -> list[list[int]]:
    norm = [sorted(int(x) for x in group) for group in groups]
    seen = [x for group in norm for x in group]
    if sorted(seen) != list(range(num_units)):
        raise ValueError("BMS domains do not partition all candidate units")
    if any(not group for group in norm):
        raise ValueError("BMS domain is empty")
    if len(seen) != len(set(seen)):
        raise ValueError("BMS domains overlap")
    return norm
