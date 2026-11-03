"""Analytic and finite-difference verification in float64."""
import numpy as np
import torch

from mes_apml import apml_loss, mes_loss, combined_loss


def _analytic_mes_grad(w):
    n, d = w.shape
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    q = np.array([np.dot(w[i]-w[j], w[i]-w[j]) for i,j in pairs])
    center = float(np.mean(q))
    grad = np.zeros_like(w)
    for (i,j), value in zip(pairs, q):
        contribution = (4.0 / len(pairs)) * (value - center) * (w[i]-w[j])
        grad[i] += contribution
        grad[j] -= contribution
    return grad


def test_mes_autograd_matches_analytic_gradient():
    w_np = np.random.default_rng(1337).normal(size=(5,3))
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    grad = torch.autograd.grad(mes_loss(w), w)[0].detach().numpy()
    np.testing.assert_allclose(grad, _analytic_mes_grad(w_np), rtol=1e-12, atol=1e-12)


def test_mes_gradient_matches_central_finite_differences():
    w_np = np.random.default_rng(23).normal(size=(4,3))
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    grad = torch.autograd.grad(mes_loss(w), w)[0].detach().numpy()
    h = 1e-6
    estimates = np.zeros_like(w_np)
    for idx in np.ndindex(w_np.shape):
        plus = w_np.copy(); minus = w_np.copy()
        plus[idx] += h; minus[idx] -= h
        estimates[idx] = (mes_loss(torch.tensor(plus)).item()
                          - mes_loss(torch.tensor(minus)).item()) / (2*h)
    np.testing.assert_allclose(grad, estimates, rtol=1e-8, atol=1e-8)


def test_beta_does_not_change_embedding_gradient_but_changes_proxy_gradient():
    gen = torch.Generator().manual_seed(19)
    z0 = torch.randn(7, 3, generator=gen, dtype=torch.float64)
    w0 = torch.randn(4, 3, generator=gen, dtype=torch.float64)
    y = torch.tensor([0,1,2,3,0,1,2], dtype=torch.int64)

    def get_grad(beta):
        z = z0.detach().clone().requires_grad_(True)
        w = w0.detach().clone().requires_grad_(True)
        l = combined_loss(z, y, w, alpha=1, beta=beta)
        return torch.autograd.grad(l, (z, w))

    z0_grad, w0_grad = get_grad(0)
    z1_grad, w1_grad = get_grad(0.2)
    torch.testing.assert_close(z0_grad, z1_grad, atol=1e-12, rtol=1e-12)
    assert torch.linalg.norm(w1_grad-w0_grad).item() > 1e-6


def test_apml_backprop_reaches_both_embedding_and_proxy_weights():
    z = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float64))
    loss = apml_loss(z, torch.tensor([0,1,2,1]), w)
    grad_z, grad_w = torch.autograd.grad(loss, (z, w))
    assert torch.isfinite(grad_z).all() and grad_z.abs().sum() > 0
    assert torch.isfinite(grad_w).all() and grad_w.abs().sum() > 0


def test_apml_gradients_match_closed_form_for_embeddings_and_proxies():
    rng = np.random.default_rng(2027)
    z_np = rng.normal(size=(5,2))
    w_np = rng.normal(size=(3,2))
    y_np = np.array([0,1,2,0,1])
    z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    g_z, g_w = torch.autograd.grad(apml_loss(z, torch.tensor(y_np), w), (z,w))
    diffs = z_np[:,None,:]-w_np[None,:,:]
    squared = np.sum(diffs**2,axis=-1)
    affinity=1/(1+squared)
    probs=affinity/affinity.sum(axis=1,keepdims=True)
    onehot=np.eye(3)[y_np]
    coeff=(2/len(z_np))*(onehot-probs)/(1+squared)
    expected_z=(coeff[:,:,None]*diffs).sum(axis=1)
    expected_w=-(coeff[:,:,None]*diffs).sum(axis=0)
    np.testing.assert_allclose(g_z.detach().numpy(),expected_z,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(g_w.detach().numpy(),expected_w,atol=1e-12,rtol=1e-12)


def test_apml_gradients_match_central_finite_differences():
    rng = np.random.default_rng(216)
    z_np = rng.normal(size=(3,2))
    w_np = rng.normal(size=(3,2))
    y = torch.tensor([0,2,1])
    z = torch.tensor(z_np,dtype=torch.float64,requires_grad=True)
    w = torch.tensor(w_np,dtype=torch.float64,requires_grad=True)
    ga,gw = torch.autograd.grad(apml_loss(z,y,w),(z,w))
    h=1e-6
    for values,expected,which in ((z_np,ga.detach().numpy(),"z"),(w_np,gw.detach().numpy(),"w")):
        for idx in np.ndindex(values.shape):
            a=values.copy();b=values.copy();a[idx]+=h;b[idx]-=h
            if which=="z":
                fplus=apml_loss(torch.tensor(a),y,torch.tensor(w_np)).item()
                fminus=apml_loss(torch.tensor(b),y,torch.tensor(w_np)).item()
            else:
                fplus=apml_loss(torch.tensor(z_np),y,torch.tensor(a)).item()
                fminus=apml_loss(torch.tensor(z_np),y,torch.tensor(b)).item()
            np.testing.assert_allclose(expected[idx],(fplus-fminus)/(2*h),atol=1e-8,rtol=1e-8)
