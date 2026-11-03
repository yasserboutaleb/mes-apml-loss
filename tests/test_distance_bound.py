"""Check the sharp lower bound for the smallest squared inter-proxy distance."""
import math
import torch
from mes_apml import mes_loss


def test_distance_uniformity_bound_across_random_proxy_configurations():
    for c, d in ((3, 2), (4, 3), (8, 5)):
        generator = torch.Generator().manual_seed(100 + c)
        for _ in range(6):
            w = torch.randn((c, d), dtype=torch.float64, generator=generator)
            q = torch.pdist(w, p=2).square()
            mean = q.mean()
            variance = mes_loss(w)
            M = c * (c - 1) // 2
            lower = mean - torch.sqrt((M - 1) * variance)
            assert float(q.min()) + 1e-11 >= float(lower)
            assert math.isclose(float(variance), float((q - mean).square().mean()), abs_tol=1e-12)


def test_tetrahedron_saturates_zero_variance_bound():
    w = torch.tensor([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]], dtype=torch.float64)
    q = torch.pdist(w).square()
    assert math.isclose(float(q.min()), 8.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(float(q.mean()), 8.0, rel_tol=0, abs_tol=1e-12)
    assert float(mes_loss(w)) == 0.0
