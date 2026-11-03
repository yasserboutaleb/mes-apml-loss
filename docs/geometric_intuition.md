# Geometric intuition

APML treats each class proxy as a learned classifier weight. Embeddings are scored by their relative Cauchy affinities to all class proxies. MES views the **same** proxies as a point cloud and measures how uneven their **squared** pairwise distances are. Minimizing the dispersion does not set their absolute scale.

## Four proxies in three dimensions

An exact regular tetrahedron is possible with four points in 3D. Its six edges all have equal length. The analytic reference shown in [Figure 3](../figures/03_ideal_tetrahedron.svg) has squared edge length $8$ and thus MES zero. A trained model is not guaranteed to recover that construction.

A six-bar view of the unordered pairs makes the loss explicit. The squared distances in [Figure 4](../figures/04_six_squared_distances.svg) are computed from synthetic proxies trained under two objectives with identical fixed synthetic embeddings and initial weights. They do not indicate benchmark performance.

## Why six distinct centers cannot be perfectly equidistant in 3D

Six equidistant points would require a regular simplex with affine dimension five, so embedding it without distortion in 3D is impossible. For example, the six vertices of an octahedron have different distances for opposing and non-opposing vertex pairs; this is an illustration, not an optimizer's guaranteed result.

See [the rank proof](mathematical_formulation.md#dimension-bound-for-equidistant-distinct-centers) and [limitations](limitations.md).