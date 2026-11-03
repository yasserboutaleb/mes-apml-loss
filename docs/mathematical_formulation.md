# MES-loss: concise mathematical formulation

This note describes the **original 2023 APML+MES objective**, not a modified or scale-normalized regularizer. The notation and identities below are mathematical consequences of the stated loss. The code's implementation can be checked in [torch_backend.py](../src/mes_apml/torch_backend.py).

## Setup: one set of shared, trainable class proxies

For $N$ labeled examples $(x_i,y_i)$, let $z_i=f_\theta(x_i)\in\mathbb R^d$. Let $W=(w_1,\ldots,w_C)$ be $C$ learnable class-proxy vectors, $w_k\in\mathbb R^d$. These proxies are the classification-head parameters **rather than empirical minibatch centroids**.

For each example and proxy, set

$$
r_{ik}=\|z_i-w_k\|_2^2,\qquad
\ell_{ik}=-\log(1+r_{ik}).
$$

The normalized Cauchy affinity is

$$
p_{ik}=\frac{e^{\ell_{ik}}}{\sum_{j=1}^C e^{\ell_{ij}}}
      =\frac{(1+r_{ik})^{-1}}{\sum_{j=1}^C(1+r_{ij})^{-1}}.
$$

Thus, with $\operatorname{CE}$ denoting cross-entropy,

$$
\boxed{\mathcal L_{\rm APML}
=-\frac1N\sum_{i=1}^N\log p_{i,y_i}
=\frac1N\sum_{i=1}^N\operatorname{CE}(\ell_i,y_i).}
$$

This is a logit-based *implementation* of the original Cauchy-affinity objective; it adds **no** new temperature, fixed margin, or L2 normalization.

## MES: variance of squared distances

Let $\mathcal P=\{(j,k):1\le j<k\le C\}$ and $M=|\mathcal P|=\binom C2$. Define

$$
q_{jk}=\|w_j-w_k\|_2^2,\qquad
\mu=\frac1M\sum_{j<k}q_{jk}.
$$

The original MES term is

$$
\boxed{
V(W)=\mathcal L_{\rm MES}(W)
=\frac1M\sum_{j<k}(q_{jk}-\mu)^2.
}
$$

Each unordered pair is counted **once**; the divisor is $M$, not $M-1$. Squared distances are essential: this is **not** the variance of ordinary Euclidean distances.

The published composite loss is

$$
\boxed{\mathcal L(\theta,W)
=\alpha\mathcal L_{\rm APML}(\theta,W)+\beta V(W),\qquad
\alpha,\beta\ge0.}
$$

Both terms act on the **same** $W$, making the regularizer directly relevant to the classification geometry.

## Gradients: how the two objectives influence learning

Define $h_{ik}=\mathbf1[y_i=k]$. For the embeddings and proxies, respectively,

$$
\boxed{\frac{\partial \mathcal L_{\rm APML}}{\partial z_i}
=\frac2N\sum_{k=1}^C
(h_{ik}-p_{ik})\frac{z_i-w_k}{1+r_{ik}}}
$$

and

$$
\boxed{\frac{\partial \mathcal L_{\rm APML}}{\partial w_k}
=\frac2N\sum_{i=1}^N
(h_{ik}-p_{ik})\frac{w_k-z_i}{1+r_{ik}}.}
$$

MES has an exact center gradient:

$$
\boxed{
\nabla_{w_k}V
=\frac4M\sum_{j\ne k}(q_{kj}-\mu)(w_k-w_j).
}
$$

The contribution from differentiating $\mu$ cancels because $\sum_{j<k}(q_{jk}-\mu)=0$. MES contains no $z_i$, so for independently parametrized $W$ and $\theta$, it contributes **no direct** gradient to the encoder. The encoder can still be affected indirectly via later APML updates as the learned proxies change.

## Why a tetrahedron? Four classes in 3D

Consider $C$ **distinct** points with one common positive squared distance $q$. Their squared-distance matrix is $D=q(\mathbf1\mathbf1^\top-I)$. Introduce the centering projector

$$
J=I-\frac1C\mathbf1\mathbf1^\top.
$$

The centered Gram matrix is

$$
\boxed{G=-\tfrac12 JDJ=\tfrac q2 J.}
$$

Because $\operatorname{rank}(G)=\operatorname{rank}(J)=C-1$ and vectors in $\mathbb R^d$ have Gram rank no greater than $d$, one must have

$$
\boxed{d\ge C-1.}
$$

Conversely, the vertices of a regular $(C-1)$-simplex realize that configuration. Thus $C=4$ and $d=3$ give a **regular tetrahedron**. One exact coordinate example is

$$
(1,1,1),\ (1,-1,-1),\ (-1,1,-1),\ (-1,-1,1),
$$

for which all six $q_{jk}=8$ and $V=0$.

This is a *geometric feasibility result*, **not a convergence theorem** for APML+MES. In particular, six distinct equidistant points cannot exist in 3D; the octahedron is neither equidistant nor established here as an MES optimum.

## An exact bound that explains distance uniformity

Let $q_{\min}=\min_{j<k}q_{jk}$ and $\mu>0$. Since the $M$ deviations from the mean sum to zero, Cauchy–Schwarz implies

$$
\boxed{q_{\min}\ge\mu-\sqrt{(M-1)V(W)}.}
$$

*Proof.* For the smallest distance, write $a=q_{\min}-\mu\le0$. The other $M-1$ deviations sum to $-a$, so their squared sum is at least $a^2/(M-1)$. Thus $M V\ge a^2+a^2/(M-1)=M a^2/(M-1)$, or $|a|\le\sqrt{(M-1)V}$. $\square$

This is a **relative uniformity statement at a given mean distance**; it does not imply that decreasing $V$ increases $q_{\min}$, because the mean $\mu$ can change simultaneously.

For figures only, the dimensionless coefficient of variation

$$
\mathrm{CV}_q=\frac{\sqrt{V}}{\mu}\qquad(\mu>0)
$$

helps compare *geometry* across different center scales. **It is a diagnostic, not the original MES loss and not a new training objective.**

## Essential mathematical limits

1. **Translation invariance:** $V(W+a)=V(W)$ for the same $a\in\mathbb R^d$ added to each proxy.
2. **Rotation/reflection invariance:** any common orthogonal transform preserves $V$.
3. **Fourth-order scaling:** $V(sW)=s^4V(W)$.
4. **Degenerate global minima:** coincident centers give $V=0$ regardless of class count. The MES term does **not** enforce positive separation.
5. **Two-class limitation:** for $C=2$ the only squared pair distance equals its mean, so $V\equiv0$.
6. **Uniformity is not a packing guarantee:** minimizing variance is not equivalent to maximizing the minimum distance without constraints on the scale.
7. **Optimization and generalization are separate questions:** geometry alone does not certify better retrieval, clustering, or performance on unseen classes.

The mathematical results can be checked against [unit tests](../tests/), [analytic figures](../figures/README.md) and the [journal paper](https://doi.org/10.1016/j.patrec.2023.06.005).
