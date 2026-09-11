from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from task040_htor_probe import temporary_unit_mask


def parameter_snapshots(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


class TestTask040Masking(unittest.TestCase):
    def test_real_production_ffn_neuron_mask_restores_exactly(self) -> None:
        fc2 = nn.Linear(3, 3, bias=False).eval()
        with torch.no_grad():
            fc2.weight.copy_(torch.eye(3))
        spec = types.SimpleNamespace(
            unit_type="neuron",
            num_units=3,
            module=fc2,
            hook_module=fc2,
        )
        inputs = torch.tensor([[1.0, 2.0, 3.0]])
        before_parameters = parameter_snapshots(fc2)
        baseline = fc2(inputs).detach().clone()
        with temporary_unit_mask(spec, 1):
            masked = fc2(inputs).detach().clone()
        restored = fc2(inputs).detach().clone()

        self.assertTrue(torch.equal(masked, torch.tensor([[1.0, 0.0, 3.0]])))
        self.assertFalse(torch.equal(masked, baseline))
        self.assertTrue(torch.equal(restored, baseline))
        for before, after in zip(before_parameters, fc2.parameters()):
            self.assertTrue(torch.equal(before, after))

    def test_real_production_attention_head_mask_restores_exactly(self) -> None:
        attention = nn.Module()
        attention.head_dim = 2
        attention.proj = nn.Linear(6, 6, bias=False).eval()
        with torch.no_grad():
            attention.proj.weight.copy_(torch.eye(6))
        spec = types.SimpleNamespace(
            unit_type="head",
            num_units=3,
            module=attention,
            hook_module=attention.proj,
        )
        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        before_parameters = parameter_snapshots(attention)
        baseline = attention.proj(inputs).detach().clone()
        with temporary_unit_mask(spec, 1):
            masked = attention.proj(inputs).detach().clone()
        restored = attention.proj(inputs).detach().clone()

        self.assertTrue(torch.equal(masked, torch.tensor([[1.0, 2.0, 0.0, 0.0, 5.0, 6.0]])))
        self.assertFalse(torch.equal(masked, baseline))
        self.assertTrue(torch.equal(restored, baseline))
        for before, after in zip(before_parameters, attention.parameters()):
            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()


