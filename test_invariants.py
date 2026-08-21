from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import torch

from config import ExperimentConfig
from federated import PrototypeBank, _prototype_losses, update_bank
from models import SpectralEncoder, assign_flat_parameters, flatten_parameters


def test_flat_roundtrip():
    model = SpectralEncoder(4)
    vector = flatten_parameters(model)
    assign_flat_parameters(model, vector + 0.1)
    assert torch.allclose(flatten_parameters(model), vector + 0.1)


def test_spectral_distribution_and_gradients():
    model = SpectralEncoder(4)
    x = torch.randn(5, 1, 64, 96)
    logits, z, spectral = model(x)
    assert torch.allclose(spectral.sum(1), torch.ones(5), atol=1e-5)
    bank = PrototypeBank.empty(4, 96, 8, "cpu")
    bank.valid[:] = True
    bank.embedding.normal_()
    proto, transport = _prototype_losses(z, spectral, torch.arange(5) % 4, bank)
    (logits.mean() + proto + transport).backward()
    assert model.band_attention[-1].weight.grad is not None


def test_bank_is_valid_probability():
    bank = PrototypeBank.empty(2, 3, 4, "cpu")
    stats = [{0: {"count": torch.tensor(5.0), "embedding": torch.ones(3),
                   "spectral": torch.tensor([0.1, 0.2, 0.3, 0.4]), "dispersion": torch.tensor(0.2)}}]
    bank = update_bank(bank, stats)
    assert bank.valid[0]
    assert torch.allclose(bank.spectral[0].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(bank.spectral[0] >= 0)


if __name__ == "__main__":
    test_flat_roundtrip()
    test_spectral_distribution_and_gradients()
    test_bank_is_valid_probability()
    print("all invariants passed")

