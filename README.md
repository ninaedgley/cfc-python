# cfc-python

A Python translation of the confidence forced-choice (CFC) model of Mamassian & de Gardelle (2022), originally implemented in MATLAB.

The original MATLAB toolbox is by Pascal Mamassian and is available at <https://github.com/mamassian/cfc>. This repository is a translation of that toolbox, and keeps the generative model, its parameters, and its predictions unchanged.

> **Status.** Not yet publicly released. Shared directly with collaborators for review. A versioned DOI will be assigned at release, accompanying the forthcoming experimental preprint.

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ninaedgley/cfc-python.git
cd cfc-python
uv sync
```

## Equivalence tests

The test suite refits the two worked examples shared in the MATLAB toolbox (`cfc_example_1`, `cfc_example_2`), and compares the Python estimates against the MATLAB `cfc_struct` saved from the same data.

```bash
uv run pytest -v
```

`sens_noise`, `sens_crit`, `efficiency` and `loglike` are asserted against the MATLAB reference. `conf_noise` and `conf_boost` are reported but not asserted: they trade off against each other along a ridge in the Type-2 likelihood, so they are not separately identified at a single optimum.

The fixtures in `tests/` are the MATLAB outputs (`.mat`) for those two examples. They contain simulated data only — no participant data is included.

## Changes from the MATLAB implementation

Two changes were made. Both are described in full, with derivations, in an accompanying document (*A Python-translation of the confidence forced-choice model, CFC*), which will be released at a later date.

**1. Closed-form evaluation of P(C = 1 | s1, s2, D1, D2).** The original evaluates equations 25–28 by two-dimensional adaptive numerical integration (`integral2`). Writing confidence evidence as an affine function of the standard-normal sensory deviate makes the same quantity a trivariate Gaussian orthant probability, evaluated here via Owen's T function and Gauss–Legendre quadrature. This is exact for every parameter value, and since both constructions are deterministic, the likelihood remains a fixed, reproducible function of the parameters.

**2. Equal-width stimulus binning.** Motivated by continuous-stimulus designs, where cell counts become sparse. This module is held back from the present release and will be published separately.


### Optimisation

`cfc_fit` uses multiple optimisation starts where the MATLAB original uses a single start: three scaled starts for the equivalent-confidence-noise fits (steps 3–4), and five `conf_boost` starts by default in the full-model fit (step 5, `boost_init_list`). Bounds are handled by SciPy's Nelder–Mead rather than by `fminsearchbnd`'s reparameterisation. These change which optimum is found on a given dataset, not the model being fitted.

## Module map

| Module | Corresponds to |
| --- | --- |
| `cfc_core.py` | `cfc_core.m` — P(C = 1) per trial kind |
| `cfc_fit.py` | `cfc_fit.m` — maximum-likelihood fit, efficiency |
| `cfc_group.py` | `cfc_group.m` — grouping of raw trials |
| `cfc_simul_discrim.py` | `cfc_simul_discrim.m` — simulation |
| `cfc_plot.py` | `cfc_plot.m` — data-aggregation layer |
| `closed_form.py` | new — closed-form P(C = 1), see above |
| `latent_variables.py` | new — trial-level model-implied quantities |

## Citation

Please cite the original model:

> Mamassian, P., & de Gardelle, V. (2022). Modeling perceptual confidence and the confidence forced-choice paradigm. *Psychological Review*, 129(5), 976–998. https://doi.org/10.1037/rev0000312

## Acknowledgements

Supported by ANR-24-RRII-0004 *Temporal Metacognition*, within the CEA Audace! programme, awarded to Virginie van Wassenhove.