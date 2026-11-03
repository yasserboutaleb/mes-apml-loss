# MES-loss public repository: publication gate

**Status: PRIVATE REVIEW ONLY.** This branch does not authorize making the repository public or releasing code under a software license.

## Scientific and technical status

- [x] Primary article metadata and DOI checked against the supplied 2023 journal manuscript.
- [x] Original APML Cauchy-affinity and MES squared-distance population-variance equations retained.
- [x] Center-gradient and simplex-rank mathematics presented explicitly.
- [x] Separate synthetic/analytic illustrations from real educational training and benchmark evidence.
- [x] Reproducibility commands centralized under scripts/reproduce.py.
- [x] Static mathematical tests, gradient tests, CLI no-overwrite tests and offline movie tests committed.
- [x] GitHub Actions workflow configured for offline tests and figure generation.
- [ ] Confirm latest GitHub Actions run **completed successfully**; commits alone are not test results.
- [ ] Execute the genuine MNIST demo on at least one supported desktop/CPU or GPU environment; do not claim it ran here.
- [ ] Review the rendered README figures on GitHub and test the output from a clean checkout.
- [ ] Optionally provide a separate *genuine* Softmax vs APML+MES experiment if that comparison is used in public promotion.

## Authorship, rights, and public release: explicit owner decision required

- [ ] Determine who holds the copyright and redistribution rights for this **new code and generated figures**.
- [ ] Check publication, institutional, employer, collaborator and publisher obligations.
- [ ] Choose and approve a software license; **main currently contains an MIT LICENSE**, while the current review branch deliberately contains **no LICENSE**. Do not infer licensing approval from the default branch.
- [ ] Approve the software authorship metadata before adding CITATION.cff (the existing CITATION.bib cites **the journal article**, not software authorship).
- [ ] Review the exact branch diff and approve any merge to main.
- [ ] Independently approve switching the repository from PRIVATE to PUBLIC.
- [ ] Confirm no internal CUB/Cars data, private infrastructure, unpublished MES variants, trained research checkpoints, hidden credentials, or proprietary figure assets were included.

## Reproduce locally

~~~bash
python -m pip install -e ".[dev,movie]"
python scripts/reproduce.py check
python scripts/reproduce.py quickstart
python scripts/reproduce.py figures
python scripts/reproduce.py smoke --out runs/shapes_smoke_42
~~~

The smoke output path must not exist beforehand. The MNIST demo is optional and downloads training data only. Consult docs/usage_and_reproducibility.md.

**Do not merge or make public until the owner has explicitly approved the remaining publication gates.**
