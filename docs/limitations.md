# Limitations and interpretation guardrails

1. **MES does not enforce positive separation.** All identical proxies yield zero MES, regardless of number of classes.
2. **Scale matters.** Replacing $W$ by $sW$ scales MES by $s^4$; loss values across runs with different proxy scales cannot be interpreted without context.
3. **$C=2$ is uninformative for MES.** One squared distance has zero population variance, so APML alone determines training under this loss.
4. **No max-min guarantee.** More uniform pairwise distances do not guarantee a larger smallest distance, improved embeddings, or higher retrieval accuracy.
5. **Dimension constrains exact geometry.** Distinct pairwise equidistant centers require $d\ge C-1$; no exact six-point equidistant configuration exists in 3D.
6. **Do not conflate metrics.** Classification accuracy, clustering scores, and retrieval scores such as Recall@K and mAP@R measure different outcomes.
7. **Toy figures are not experiments on real datasets.** Synthetic examples illustrate formulas; they do not support empirical superiority claims.
8. **No automatic normalization, margin, or temperature.** Adding any of these changes the baseline objective and needs to be labelled as a new choice.
9. **Optimization is not globally characterized.** The MES objective does not guarantee any particular trained polyhedron, convergence rate, or generalization performance.
10. **Compute and memory cost.** Naively enumerating $\binom C2$ pairs requires quadratic work and storage in the number of class proxies; for very large class counts, memory-efficient approaches may be considered without changing the objective.