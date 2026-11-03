# Mathematical formulation — APML and MES

This document describes the **original MES-loss method** in the 2023 paper. It first gives the intuition, then the exact definitions, and finally the useful geometric properties. See the [trained four-class MNIST example](mnist_c4_evidence.md).

## 1. Shared learned class proxies

A network maps an input to an embedding. Each class has a **trainable proxy** (a model parameter, not a minibatch centroid).

```math
z_i=f_{\theta}(x_i), \qquad w_k\in\mathbb{R}^{d}
```

There are $N$ labeled training samples, $C$ classes, and $C$ proxy vectors.

## 2. APML — discriminate

APML uses a Cauchy-like affinity: samples close to a proxy receive a larger affinity.

```math
a_{ik}=\frac{1}{1+\|z_i-w_k\|^2}
```

Normalize these affinities to obtain class probabilities:

```math
p_{ik}=\frac{a_{ik}}{\sum_{j=1}^{C}a_{ij}}
```

APML penalizes low probability for the true class $y_i$:

```math
L_{\mathrm{APML}}=-\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}
```

**Implementation note:** using logits $\ell_{ik}=-\log(1+\|z_i-w_k\|^2)$ and standard cross-entropy gives the same objective. No additional normalization, temperature, or margin is used.

## 3. MES — organize the same class proxies

For each unordered pair of different class proxies, compute the **squared Euclidean distance**:

```math
q_{jk}=\|w_j-w_k\|^2
```

Every pair is counted **once**. With $C$ classes there are $M=C(C-1)/2$ pairs; for four classes, there are six. Label these squared distances $q_1,\ldots,q_M$.

Their mean is

```math
\mu=\frac{1}{M}\sum_{r=1}^{M}q_r
```

The **original MES loss** is the population variance of these distances:

```math
L_{\mathrm{MES}}=\frac{1}{M}\sum_{r=1}^{M}(q_r-\mu)^2
```

This uses the squared distances, not the ordinary Euclidean lengths. The divisor is $M$, not $M-1$. MES encourages uniform pairwise distances without specifying a target radius or a minimum separation.

A regular tetrahedron of four **distinct** centers has six equal distances and zero MES, but **fully collapsed centers also have zero MES**. APML provides the data-dependent discriminative term.

## 4. Joint training — discrimination plus geometry

Both objectives act on the **same** proxy parameters:

```math
L=\alpha L_{\mathrm{APML}}+\beta L_{\mathrm{MES}}
```

APML directly trains both the embeddings and proxies; MES acts directly on proxies and influences the encoder **indirectly** through subsequent APML optimization.

## 5. What the gradients do (technical detail)

For $\ell_{ik}=-\log(1+\|z_i-w_k\|^2)$ and $h_{ik}=1$ when $y_i=k$ (zero otherwise), the APML gradient on an embedding is

```math
\frac{\partial L_{\mathrm{APML}}}{\partial z_i}
=\frac{2}{N}\sum_{k=1}^{C}
(h_{ik}-p_{ik})\frac{z_i-w_k}{1+\|z_i-w_k\|^2}
```

The APML gradient on proxy $k$ is

```math
\frac{\partial L_{\mathrm{APML}}}{\partial w_k}
=\frac{2}{N}\sum_{i=1}^{N}
(h_{ik}-p_{ik})\frac{w_k-z_i}{1+\|z_i-w_k\|^2}
```

For MES, the gradient on the proxy is

```math
\frac{\partial L_{\mathrm{MES}}}{\partial w_k}
=\frac{4}{M}\sum_{j\ne k}(q_{jk}-\mu)(w_k-w_j)
```

These expressions assume proxy vectors and encoder parameters are independently parameterized. The derivative of $\mu$ cancels in the MES gradient because the deviations from their mean sum to zero.

## 6. Why four classes can form a tetrahedron in 3D

A regular tetrahedron has **four different vertices and six equal distances**. More generally, $C$ distinct equidistant points require at least $C-1$ dimensions.

A compact proof uses their centered squared-distance matrix. If every off-diagonal squared distance equals the same positive number $q$, define

```math
D_{jk}=q\quad(j\ne k), \qquad D_{jj}=0
```

and the centering matrix

```math
J=I-\frac{1}{C}\mathbf{1}\mathbf{1}^{T}
```

The centered Gram matrix is

```math
G=-\frac{1}{2}JDJ=\frac{q}{2}J
```

Because this Gram matrix has rank $C-1$, the embedding dimension must obey

```math
d\ge C-1
```

For $C=4$, the lower bound is $d=3$, so a regular tetrahedron is **geometrically feasible**. This does not guarantee gradient-based learning will converge to it.

## 7. How uniformity relates to the smallest distance

Writing $V=L_{\mathrm{MES}}$ and $q_{\min}$ for the smallest squared pair distance gives the bound

```math
q_{\min}\ge \mu-\sqrt{(M-1)V}
```

The bound says that *if the mean squared distance is held fixed*, a small MES variance limits how far the smallest distance can lie below the mean. **If the mean shrinks, reducing MES does not necessarily increase the smallest distance.**

A scale-free measure suitable for **displaying** geometry is the coefficient of variation

```math
\mathrm{CV}_{q}=\frac{\sqrt{V}}{\mu}
```

where $\mu>0$. This is only an **analysis metric**, not a replacement for the original MES training loss.

## 8. Scope and mathematical limitations

- Translation and rotation of all proxies together leave MES unchanged.
- Scaling all proxies by $s$ multiplies the **original** MES by $s^4$.
- MES is identically zero for $C=2$ because there is only one pair.
- Collapsed proxies and nondegenerate equal-distance configurations both attain zero MES.
- Uniformity of pairwise distances does not establish improved generalization or test retrieval.
- Three paired MNIST seeds show reproducible *training geometry* in the supplied study; they do not prove universal convergence.

[Return to README](../README.md) · [Trained MNIST figures and provenance](mnist_c4_evidence.md) · [Original publication](https://doi.org/10.1016/j.patrec.2023.06.005)
