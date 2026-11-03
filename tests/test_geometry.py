"""MES invariances and exact geometric boundary cases."""
import numpy as np
import torch
from mes_apml import mes_loss


def test_translation_rotation_and_scaling():
    rng = np.random.default_rng(17)
    w_np = rng.normal(size=(5,3))
    w = torch.tensor(w_np, dtype=torch.float64)
    offset = torch.tensor([5.0, -9.0, 1.4], dtype=torch.float64)
    base = mes_loss(w)
    torch.testing.assert_close(mes_loss(w + offset), base, atol=1e-12, rtol=1e-12)
    factor = 2.3
    torch.testing.assert_close(mes_loss(factor*w), (factor**4)*base, atol=1e-10, rtol=1e-12)
    a = rng.normal(size=(3,3))
    q,_ = np.linalg.qr(a)
    torch.testing.assert_close(mes_loss(w@torch.tensor(q, dtype=torch.float64)), base,
                               atol=1e-12, rtol=1e-12)


def test_exact_regular_tetrahedron_zero_variance():
    w = torch.tensor([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]], dtype=torch.float64)
    pairs = torch.triu_indices(4, 4, offset=1)
    d2 = (w[pairs[0]] - w[pairs[1]]).square().sum(-1)
    torch.testing.assert_close(d2, torch.full_like(d2, 8.0), atol=0, rtol=0)
    assert mes_loss(w).item() == 0.0


def test_all_centers_collapsed_zero_mes_and_zero_gradient():
    w = torch.ones(5,3, dtype=torch.float64, requires_grad=True)
    loss = mes_loss(w)
    gradient = torch.autograd.grad(loss, w)[0]
    assert loss.item() == 0.0 and torch.count_nonzero(gradient) == 0


def test_two_classes_mes_always_zero_even_when_far_apart():
    w = torch.tensor([[0.,0.], [40., -10.]], dtype=torch.float64, requires_grad=True)
    loss = mes_loss(w)
    gradient = torch.autograd.grad(loss, w)[0]
    assert loss.item() == 0.0 and torch.count_nonzero(gradient) == 0


def test_six_octahedral_vertices_have_unequal_pairwise_distances():
    w = torch.cat((torch.eye(3, dtype=torch.float64), -torch.eye(3, dtype=torch.float64)))
    unique = torch.unique(torch.pdist(w).square())
    torch.testing.assert_close(unique, torch.tensor([2.0, 4.0], dtype=torch.float64))
    assert mes_loss(w).item() > 0
