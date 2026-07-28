# Getting started

## Installation

hrRSF can currently be installed from the source repository:

```bash
git clone <repository-url>
cd hrRSF
python -m pip install -e .


### `docs/theory.md`

```markdown
# Theoretical foundation

## Resource selection

Let \(D \subseteq \mathbb{R}^2\) denote the geographic domain available to an
animal. The relative intensity of use at location \(s \in D\) may be represented
as

\[
\lambda(s) = \exp\left(\beta_0 + \boldsymbol{\beta}^{\mathsf T}
\mathbf{x}(s)\right),
\]

where \(\mathbf{x}(s)\) contains environmental covariates and
\(\boldsymbol{\beta}\) contains the corresponding selection coefficients.

## From telemetry to inference

Telemetry records provide a finite and imperfect observation of an animal's
underlying spatial distribution. hrRSF therefore distinguishes between:

1. the latent process governing space use;
2. the observation process generating telemetry records;
3. the numerical representation of the environment;
4. inference about individual and population-level selection.