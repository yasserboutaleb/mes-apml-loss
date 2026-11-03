# Publication checklist — private staging branch

**Status:** private review only. The repository is not approved for public release.

## Separation from the research lab

- This repository has an independent history; do not merge, mirror, or cherry-pick unpublished research-lab commits.
- Do not include internal infrastructure, private CUB/Cars experiments, datasets, model checkpoints, tokens, or proprietary assets.
- Only the original APML and MES loss formulations are in scope; experimental variants remain private.

## Before making the repository public

- [ ] Review the origin, ownership, and contribution rights of all source code and figures.
- [ ] Confirm employer, institution, collaborator, and publisher obligations where applicable.
- [ ] Approve or change the MIT `LICENSE` file that currently exists on `main`. Presence of this file does **not** establish that rights have been cleared.
- [ ] Cross-check equations and claimed properties against the published article and thesis.
- [ ] Approve exact citation metadata and add `CITATION.cff`.
- [ ] Regenerate the synthetic illustrations and review their provenance.
- [ ] Verify no keys, private paths, training data, checkpoints, or unpublished results are staged.
- [ ] Review the exact changes between `review/prepublication` and `main`.
- [ ] Explicitly authorize merge and public visibility.

## Local checks

```bash
python -m pip install -e '.[dev,movie]'
python -m pytest -q
python scripts/make_figures.py --out figures
python scripts/cluster_movie.py --help
```

For MP4 export install `ffmpeg`; the procedural-shapes mode permits offline testing without MNIST. The official MNIST test split is not used for training or selection. Synthetic demonstration results must not be represented as benchmark outcomes.

**Do not merge to main or make public without explicit approval.**
