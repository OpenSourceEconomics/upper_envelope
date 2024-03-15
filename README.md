# Fast Upper Envelope Scan (FUES)

[![Continuous Integration Workflow](https://github.com/OpenSourceEconomics/upper-envelope/actions/workflows/main.yml/badge.svg)](https://github.com/OpenSourceEconomics/upper-envelope/actions/workflows/main.yml)
[![Codecov](https://codecov.io/gh/OpenSourceEconomics/upper-envelope/branch/main/graph/badge.svg)](https://app.codecov.io/gh/OpenSourceEconomics/upper-envelope)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Extension of the Fast Upper-Envelope Scan (FUES) for solving discrete-continuous dynamic programming problems based on Dobrescu & Shanker (2022). Both
`jax` and `numba` versions are available.

## References

1. Iskhakov, Jorgensen, Rust, & Schjerning (2017).
   [The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks](http://onlinelibrary.wiley.com/doi/10.3982/QE643/full).
   *Quantitative Economics*


1. Loretti I. Dobrescu & Akshay Shanker (2022).
   [Fast Upper-Envelope Scan for Discrete-Continuous Dynamic Programming](https://dx.doi.org/10.2139/ssrn.4181302).