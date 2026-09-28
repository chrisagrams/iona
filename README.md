# Iona

Iona is a transformer foundation model for tandem mass spectra (MS/MS). Each centroided peak is a
token, and attention is biased by the m/z difference between every pair of peaks.

## Model weights

### Base weights
| Model | Weights |
|---|---|
| iona-base-25m | <https://anonymous-hf.com/a/z6u85kwe7lca/> |
| iona-base-50m | <https://anonymous-hf.com/a/62zrxaphrn9l/> |
| iona-base-100m | <https://anonymous-hf.com/a/oye56ejrifj2/> |
| iona-base-200m | <https://anonymous-hf.com/a/bzjg2q0sldh2/> |
| iona-base-400m | <https://anonymous-hf.com/a/zt15jot0z4j5/> |

### Denoising weights
| Model | Weights |
|---|---|
| iona-denoise-50m | <https://anonymous-hf.com/a/okz7mkmprv9c/> |
| iona-denoise-100m | <https://anonymous-hf.com/a/wo8iy10zz0b0/> |
| iona-denoise-200m | <https://anonymous-hf.com/a/s45ryj496v0h/> |
| iona-denoise-400m | <https://anonymous-hf.com/a/0zflp5r5t0va/> |

### Contrastive weights
| Model | Weights |
|---|---|
| iona-contrastive-50m | <https://anonymous-hf.com/a/us7v8ygzdb5u/> |
| iona-contrastive-400m | <https://anonymous-hf.com/a/okrgczwxmuk6/> |

### Peptide embedder
| Model | Weights |
|---|---|
| iona-peptide-embedder-400m | <https://anonymous-hf.com/a/4ilud8sqip5n/> |
## Releasing

Releases are automated with [release-please](https://github.com/googleapis/release-please). PRs are
squash-merged, and their titles must follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, ...). Pushes to `main` keep a release PR up to date with the next version and
`CHANGELOG.md`. Merging that PR tags the release, creates a GitHub release and publishes the package to
PyPI.
