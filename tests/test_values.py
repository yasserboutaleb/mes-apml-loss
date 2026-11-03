"""Value and shape tests with independently written mathematical references."""
import numpy as np
import pytest
import torch

from mes_apml import apml_loss, mes_loss, combined_loss

DTYPE = torch.float64


def test_apml_matches_direct_normalized_cauchy_affinities():
    rng = np.random.default_rng(2026)
    z_np = rng.normal(size=(6, 3))
    w_np = rng.normal(size=(4, 3))
    labels = np.array([0, 1, 3, 2, 1, 0])
    d2 = ((z_np[:, None] - w_np[None, :]) ** 2).sum(axis=-1)
    affinities = 1.0 / (1.0 + d2)
    probs = affinities / affinities.sum(axis=1, keepdims=True)
    expected = -np.log(probs[np.arange(len(labels)), labels]).mean()
    actual = apml_loss(torch.tensor(z_np), torch.tensor(labels), torch.tensor(w_np))
    np.testing.assert_allclose(actual.item(), expected, rtol=0, atol=1e-13)


def test_mes_uses_three_unique_squared_distances_and_population_variance():
    # q=[4, 1, 5], mean=10/3 => population variance=26/9
    w = torch.tensor([[0, 0], [2, 0], [0, 1]], dtype=DTYPE)
    np.testing.assert_allclose(mes_loss(w).item(), 26/9, atol=1e-13, rtol=0)


def test_combined_uses_both_weights():
    z = torch.tensor([[0.1, 0.4], [1.0, -0.2]], dtype=DTYPE)
    y = torch.tensor([0, 1])
    w = torch.tensor([[0.0, 0.0], [1.0, 0.4], [-0.5, 0.8]], dtype=DTYPE)
    a, b = 0.7, 0.13
    val = combined_loss(z, y, w, alpha=a, beta=b)
    torch.testing.assert_close(val, a*apml_loss(z,y,w) + b*mes_loss(w))


def test_rejects_invalid_labels_and_shapes():
    z = torch.randn(3, 2, dtype=DTYPE)
    w = torch.randn(4, 2, dtype=DTYPE)
    with pytest.raises(TypeError):
        apml_loss(z, torch.tensor([0.0, 1.0, 2.0]), w)
    with pytest.raises(ValueError):
        apml_loss(z, torch.tensor([0, 1]), w)
    with pytest.raises(ValueError):
        apml_loss(z, torch.tensor([0, 1, 2]), torch.randn(4, 3, dtype=DTYPE))
    with pytest.raises(ValueError):
        mes_loss(torch.randn(1, 2))
    with pytest.raises(ValueError):
        combined_loss(z, torch.tensor([0,1,2]), w, beta=-1)
