# MES-loss

**Mutually Equidistant Separation Metric Learning Loss Function**

Yasser Boutaleb, Catherine Soladié, Nam-Duong Duong, Amine Kacete, Jérôme Royan, Renaud Séguier  
*Pattern Recognition Letters*, 172, 58–64 (2023) · [Read the paper](https://doi.org/10.1016/j.patrec.2023.06.005) · [BibTeX](CITATION.bib)

> **Private scientific preview.** This educational implementation has not yet been approved for redistribution. It does not claim to reproduce the paper's benchmark results.

## The idea

**Learn to distinguish classes, while encouraging their learned representatives to form a more uniform geometry.**

The method combines **two complementary objectives using the same trainable class centers**:

| **APML — Discriminate** | **MES — Organize** |
|:---|:---|
| Learns embeddings that favor the correct class through distance-based Cauchy probabilities. | Reduces differences between the squared distances of every pair of class centers. |
| **Question:** Which class does this sample belong to? | **Question:** How evenly are the class centers arranged? |

There are no extra margins, normalization constraints, or fixed target distances in the original formulation.

## How it works — three steps

Let $z_i=f_\theta(x_i)$ be the embedding of input $x_i$, and let $w_k$ be the **learnable** representative (proxy) of class $k$. These are model parameters, not minibatch averages.

### 1. APML: learn discriminative embeddings

Each embedding is assigned a Cauchy-affinity probability for every class:

$$
p_{ik}=\frac{(1+\|z_i-w_k\|^2)^{-1}}{\sum_j(1+\|z_i-w_j\|^2)^{-1}}
$$

APML penalizes low probability for the correct class:

$$
\mathcal{L}_{\mathrm{APML}}=-\frac{1}{N}\sum_i\log p_{i,y_i}
$$

In PyTorch, the same probabilities can be implemented stably using logits $-\log(1+\|z_i-w_k\|^2)$ with cross-entropy.

### 2. MES: encourage uniform distances between class centers

Consider every distinct pair of trainable centers, and compute its **squared** Euclidean distance:

$$
q_{jk}=\|w_j-w_k\|^2 \qquad (j<k)
$$

MES is simply the population variance of these pairwise squared distances:

$$
\mathcal{L}_{\mathrm{MES}}=\operatorname{Var}_{j<k}(q_{jk})
$$

That is, the loss becomes small when the pairwise squared distances become similar. **MES does not prescribe how large those distances must be.** The exact finite-sum definition is in [the mathematical notes](docs/mathematical_formulation.md).

### 3. Train the two objectives together

$$
\mathcal{L}=\alpha\,\mathcal{L}_{\mathrm{APML}}+\beta\,\mathcal{L}_{\mathrm{MES}}
$$

**One set of class centers, two jobs:** APML learns discrimination; MES encourages a more regular arrangement of those same centers.

## Why a tetrahedron appears with four classes in 3D

Four distinct points *can* have six equal pairwise distances in three dimensions. Their shape is a **regular tetrahedron**.

![Analytic regular tetrahedron showing four class-proxy positions and six equal edges](figures/03_ideal_tetrahedron.svg)

*Geometric reference, not trained model weights: the four displayed vertices have six identical squared distances, all equal to 8.*

More generally, $C$ **distinct equidistant** points require at least $C-1$ dimensions. This explains why a tetrahedron is geometrically possible for **4 classes in 3D**. It does **not** guarantee that training will reach one.

**Important distinction:** MES alone also assigns zero loss to *collapsed* centers. APML provides the data-dependent discriminative objective, while MES regularizes their arrangement. Neither tetrahedral convergence nor better retrieval on unseen classes is mathematically guaranteed. [See the proof and limitations](docs/mathematical_formulation.md).

## Reproduce the method

Start with Python 3.10+ and a compatible PyTorch installation. The first two examples run on **CPU**, without downloading a dataset.

**Check the loss and gradients:**

~~~bash
python -m pip install -e ".[dev]"
python scripts/reproduce.py check
~~~

**Run a tiny synthetic model and regenerate the mathematical figures:**

~~~bash
python scripts/reproduce.py quickstart
python scripts/reproduce.py figures
~~~

**Optional: record real four-class MNIST training in 3D:**

~~~bash
python -m pip install -e ".[dev,movie]"
python scripts/reproduce.py mnist --out runs/mnist_1234_seed42 --video
~~~

This last command trains **APML vs APML + MES**, starting from the same weights and using the same minibatches, and captures genuine embedding and center positions throughout training. It uses the MNIST *training split only*; the video is an **educational example**, not a benchmark reproduction. Encoding MP4 requires FFmpeg.

For a fast dataset-free training smoke test, use the smoke command in [the reproduction instructions](docs/usage_and_reproducibility.md). Existing run directories are protected against overwriting.

[Reproduction instructions](docs/usage_and_reproducibility.md) · [Video protocol](docs/cluster_movie_guide.md) · [Python implementation](src/mes_apml/torch_backend.py)

## Learn more

- [Complete mathematical formulation and gradients](docs/mathematical_formulation.md)
- [Geometry and dimensionality](docs/geometric_intuition.md)
- [Figure provenance: analytic vs synthetic vs trained](figures/README.md)
- [Scientific limitations](docs/limitations.md)
- [Paper reference](docs/references.md)

**Citation:** Boutaleb *et al.*, “MES-loss: Mutually Equidistant Separation Metric Learning Loss Function,” *Pattern Recognition Letters* **172**, 58–64 (2023). [DOI](https://doi.org/10.1016/j.patrec.2023.06.005).

**Publication status:** no public release, merger, or software license is approved by this draft. See [publication checklist](docs/GIT_PUBLICATION_CHECKLIST.md).
