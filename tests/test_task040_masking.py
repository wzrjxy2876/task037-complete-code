from __future__ import annotations

import unittest
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def temporary_neuron_mask(module: nn.Linear, unit_index: int):
    def hook(_module, inputs):
        values = inputs[0].clone()
        values[..., unit_index] = 0.0
        return (values,) + tuple(inputs[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class TestTask040Masking(unittest.TestCase):
    def test_mask_context_restores_model_and_parameters_exactly(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1.0)
        inputs = torch.ones(3, 4)
        parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
        baseline = model(inputs).detach().clone()
        with temporary_neuron_mask(model[2], 1):
            masked = model(inputs).detach().clone()
        restored = model(inputs).detach().clone()
        self.assertFalse(torch.equal(masked, baseline))
        self.assertTrue(torch.equal(restored, baseline))
        for before, after in zip(parameters_before, model.parameters()):
            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()

