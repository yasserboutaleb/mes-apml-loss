# Mathematical formulation — APML and original MES

> Working mathematical note. Audited against the designated reference code; comparison to the full thesis and publisher's complete article is still pending. This note distinguishes equations from empirical claims.

## Setup

Consider $N\ge1$ labeled inputs $(x_i,y_i)$ with $y_i\in\{0,\dots,C-1\}$, $C\ge2$. An encoder $f_\theta$ produces $z_i=f_\theta(x_i)\in\mathbb R^d$. The classifier has $C$ **trainable** class-proxy vectors $w_k\in\mathbb R^d$, collected in $W$. These are learned parameters, **not** mini-batch empirical class centroids.

## APML

With $r_{ik}=\lVert z_i-w_k\rVert_2^2$ and affinity $a_{ik}=(1+r_{ik})^{-1}$,

$$p_{ik}=\frac{a_{ik}}{\sum_{j=0}^{C-1}a_{ij}},\qquad
\mathcal L_{\mathrm{APML}}=-\frac1N\sum_i \log p_{i,y_i}.$$

Numerically stable computation uses $\ell_{ik}=\log(a_{ik})=-\log(1+r_{ik})$ and softmax cross-entropy over $\ell$. There is no cosine normalization, margin, or temperature in this formulation.

Define $h_{ik}=\mathbf 1[y_i=k]$. By differentiating the log-normalized affinities,

$$
\frac{\partial\mathcal L_{\mathrm{APML}}}{\partial z_i}
=\frac2N\sum_k(h_{ik}-p_{ik})\frac{z_i-w_k}{1+r_{ik}},
$$

$$
\frac{\partial\mathcal L_{\mathrm{APML}}}{\partial w_k}
=-\frac2N\sum_i(h_{ik}-p_{ik})\frac{z_i-w_k}{1+r_{ik}}.
$$

The sample-specific forces depend on the Cauchy denominator and the normalized probabilities. Describing APML as simply pulling positives and pushing negatives is an intuition, not a constant-force law.

## Original MES

Let $\mathcal P=\{(j,k):0\le j<k<C\}$ be the unordered set of distinct proxy pairs and $M=|\mathcal P|=C(C-1)/2$. Set

$$q_{jk}=\lVert w_j-w_k\rVert_2^2,\qquad\mu=\frac1M\sum_{(j,k)\in\mathcal P}q_{jk}.$$

Then

$$\mathcal L_{\mathrm{MES}}=\frac1M\sum_{(j,k)\in\mathcal P}(q_{jk}-\mu)^2.$$

**This is the population variance of squared Euclidean distances**, with divisor $M$ and each unordered pair counted once. It should not be replaced by the variance of ordinary distances or a variance with denominator $M-1$.

Since $\sum_{(j,k)\in\mathcal P}(q_{jk}-\mu)=0$, differentiating the mean does not contribute an extra term to the final expression:

$$
\boxed{\nabla_{w_k}\mathcal L_{\mathrm{MES}}=\frac4M\sum_{j\ne k}(q_{kj}-\mu)(w_k-w_j).}
$$

It is zero when every pairwise squared distance is equal. In particular, all-coincident weights are a zero-loss configuration. For $C=2$, the sole squared distance always equals its mean, so the loss and its gradient are identically zero.

## Joint training

$$\mathcal L=\alpha\mathcal L_{\mathrm{APML}}+\beta\mathcal L_{\mathrm{MES}},\qquad \alpha,\beta\ge0.$$

MES depends on $W$ but not directly on any $z_i$; hence $\partial\mathcal L_{\mathrm{MES}}/\partial z_i=0$. If $z_i=f_\theta(x_i)$ and $W$ is a separate trainable head parameter, the MES term has no **direct** gradient to $\theta$. In training, changing $W$ can indirectly alter later APML gradients to $\theta$.

## Geometric properties

1. **Common translation:** replacing every $w_k$ by $w_k+a$ leaves each $q_{jk}$ and MES unchanged.
2. **Uniform scaling:** replacing every $w_k$ by $sw_k$ sends $q_{jk}$ to $s^2q_{jk}$, $\mu$ to $s^2\mu$, and therefore $\mathcal L_{\mathrm{MES}}$ to $s^4\mathcal L_{\mathrm{MES}}$.
3. **Orthogonal transformation:** $W\mapsto WQ$ for an orthogonal $Q$ also leaves all distances unchanged.
4. **Zero variance is not a positive separation constraint:** $W=(w,\dots,w)$ gives MES $0$; at any fixed positive common distance, a nondegenerate equidistant configuration also gives MES $0$.
5. **Variance is not max-min packing:** the loss contains no radius constraint, direct lower bound on $\min_{j<k}\|w_j-w_k\|$, or positive target edge length.

## Dimension bound for equidistant distinct centers

Suppose $C$ distinct points have the same positive squared pairwise distance $q>0$. The squared-distance matrix is $D=q(\mathbf 1\mathbf 1^\top-I)$, with zero diagonal. Let $J=I-C^{-1}\mathbf 1\mathbf 1^\top$ center coordinates. Classical distance geometry gives their centered Gram matrix

$$G=-\tfrac12 JDJ=\tfrac q2 J.$$

As $\mathrm{rank}(J)=C-1$ and every Gram matrix of points in $\mathbb R^d$ has rank at most $d$, necessarily $d\ge C-1$. Conversely, a centered regular simplex with $C$ vertices in $\mathbb R^{C-1}$ attains equality.

For $C=4,d=3$, one exact construction uses $(1,1,1)$, $(1,-1,-1)$, $(-1,1,-1)$, $(-1,-1,1)$: all six pairwise squared distances equal $8$. For $C=6$ or $8$ in $\mathbb R^3$, pairwise equidistance of *distinct* centers is impossible. This statement does not determine the shape reached by a learned model, and MES can still be zero at collapse.

## Verification checklist

- Compare APML to a direct normalized-affinity calculation in float64.
- Compare MES to an independently enumerated list of $M$ unique squared distances.
- Compare analytic MES gradients to autograd and centered finite differences.
- Verify $C=2$, collapse, exact tetrahedron, translation, $s^4$ scaling, and shared-$W$ gradient flow.
- For any future TensorFlow backend, compare both values and both gradients on the same numeric arrays.

[Back to README](../README.md)