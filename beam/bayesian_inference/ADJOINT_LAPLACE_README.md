# Phase 2 by adjoint optimisation + Laplace posterior

## What this adds

A Phase 2 path that replaces sampling with an adjoint-based optimisation, following
the CVaR setup (`MeasurementResidualResponseFunction`, `SystemIdentificationStaticAnalysis`,
realisation loop with weighted combination), and adds the Laplace approximation that
the CVaR setup did not have.

```
parameter guess -> Kratos primal -> predicted u -> J -> adjoint gradient
-> parameter update -> ... -> optimum (MAP) -> Hessian from adjoint sensitivities
-> Laplace posterior N(MAP, H^-1)
```

Two modes, selected by `"mode"` in `AdjointLaplaceParameters.json`:

| mode | unknowns | data | Kratos solves per iteration |
|---|---|---|---|
| `u_mean` | alpha per zone | `value` (= u_hat_mean) of `measured_data_collapsed.csv` | 1 primal + 1 misfit adjoint + 1 adjoint per sensor |
| `gauss_hermite` | population mean and sd of alpha per zone | `u_hat_mean`, `u_hat_std`, `n_samples` of the same file | per Gauss-Hermite point: 1 primal + 1 adjoint per sensor (3 points for 1 zone, 3^z for z zones) |

No finite differences are used anywhere in the method. Phase 2 reads displacements only.

No existing file is modified. `MainBayesian.py`, `BayesianParameters.json`, the SMC sampler,
the three-point and batch controllers all work exactly as before.

## Files

| File | Role |
|---|---|
| `MainAdjointLaplace.py` | runner: `python MainAdjointLaplace.py` |
| `AdjointLaplaceParameters.json` | settings of this method; model, zones and alpha prior are read from `BayesianParameters.json` (`base_parameters_file`) |
| `adjoint_laplace_analysis.py` | controller: settings, data, startup checks, optimisation, Laplace, output |
| `adjoint_sensitivity_model.py` | Kratos: existing `KratosForwardModel` (primal) + SI adjoint on a node-sharing `AdjointStructure` + SI sensors |
| `expectation_evaluator.py` | objectives: `UMeanObjective`, `GaussHermiteMomentObjective` (pure numpy) |
| `adjoint_laplace_core.py` | prior, Gauss-Hermite rule, Gauss-Newton / steepest-descent-BB, Laplace (pure numpy) |
| `AdjointParametersBayes.json` | adjoint analysis settings (like `DamageResponseParametersCase2A.json`) |
| `AdjointStructuralMaterials.json` | empty materials file for the adjoint (as in the CVaR setup) |
| `check_adjoint_setup.py` | step 1: verifies the Kratos adjoint wiring, no optimisation |
| `test_adjoint_laplace_core.py` | math tests with a closed-form mock, no Kratos needed |

## How the pieces map to the CVaR setup

| CVaR setup | here |
|---|---|
| `DamageDetectionResponse` / `MeasurementResidualResponseFunction(p)` | same class, `p = 1`: J = sum 0.5 (u - d)^2, divided by sigma^2 |
| `adjoint_analysis.CalculateGradient(response)` then `-GetGradient(YOUNG_MODULUS_SENSITIVITY)` | same calls; element values summed per zone and multiplied by E_ref |
| `connectivity_preserving_model_part_controller` (Structure <-> AdjointStructure) | `ConnectivityPreserveModeler` from Structure to AdjointStructure: shared nodes, same element ids |
| loop over xi quadrature points, weights | loop over Gauss-Hermite points, weights (1/6, 2/3, 1/6) |
| steepest descent + BB step | `"type": "steepest_descent_bb"` (same BB formulas); default is `"gauss_newton"` |
| (none) | Laplace posterior from the Gauss-Newton Hessian |

One deliberate difference: CVaR averages the *losses* of the realisations. The
Gauss-Hermite mode averages the *predictions* first (expected response) and then compares
them with the data. Each point's adjoint then has to be weighted by the combined residual,
so the per-sensor adjoint sensitivities are used (a Kratos `Sensor` is itself an adjoint
response function).

## Run it step by step

All commands from `beam/bayesian_inference`.

**Step 0: math test (no Kratos, seconds)**
```
python test_adjoint_laplace_core.py
```
Expect `all checks passed`.

**Step 1: check the Kratos adjoint wiring**
```
python check_adjoint_setup.py
```
Expect, at alpha = 1.0 and at alpha = 0.8:
- check 1: sensor value difference around 1e-12 or smaller
- check 2: homogeneity ratio `[1.]` with status `ok` (exact identity sum alpha du/dalpha = -u)
- check 3: gradient difference around 1e-8 or smaller (a warning is printed above
  `gradient_tolerance`, 1e-6)
- `du/dalpha` equal to `-u/alpha`

If check 2 reports `sign_flipped`, the run corrects the sign automatically and says so.
If it reports `failed`, stop and send the console output.

**Step 2: stage 1, u_mean**

In `AdjointLaplaceParameters.json` keep `"mode": "u_mean"`, then
```
python MainAdjointLaplace.py
```
Converges in a few Gauss-Newton iterations. Expect alpha about 1 % below the mean of the
Phase 1 alphas: that is the 1/alpha nonlinearity (u_mean is not the response of the mean
structure), the same bias `collapse_phase1.py` prints.

**Step 3: stage 2, Gauss-Hermite population**

Set `"mode": "gauss_hermite"` and run again. Expect the population mean and sd of alpha
close to the Phase 1 sample values (in a closed-form test with the clean 983-sample
statistics: mean -0.03 %, sd -0.7 %).

Setting `"match": ["mean"]` shows why the sd must be matched too: the Hessian becomes
singular (mu and sigma perfectly correlated) and the run warns that the population sd is
not identified.

To compare the update strategies, set `"optimizer": {"type": "steepest_descent_bb"}` and
raise `"max_iterations"`.

## Outputs (`output_adjoint_laplace/<mode>/`)

| File | Content |
|---|---|
| `summary.json` | MAP, sd, covariance, correlation, Hessian eigenvalues, gradient norm and Newton decrement at the optimum, solve counts, startup checks, data fit, validation |
| `laplace_posterior.npz` | MAP, covariance, Hessian, 100 000 Laplace samples, iteration history |
| `iteration_history.csv` | theta, J, gradient norm per iteration |
| `posterior.xlsx` | samples (first 20 000), statistics, iterations (needs pandas + openpyxl) |
| `posterior_alpha_laplace.png` | u_mean mode: Laplace posterior of E per zone |
| `population_laplace.png`, `laplace_parameters.png` | Gauss-Hermite mode: recovered population with 95 % band, and the (mu, sigma) posterior |

`summary.json` keys for zones follow `posterior_output_process.py` (`alpha_mean`,
`alpha_std`, `E_mean_Pa`, ...) so existing plotting can read them.

`validation_reference_file` (Phase 1 `phase1_response_summary.json`) is read after the
run only, to report the error; it is never used in the inference. Set it to `""` to skip.

## Laplace validity checks (printed as warnings when they fail)

- gradient norm and Newton decrement at the optimum ~ 0 (a true optimum)
- Hessian positive definite and not extremely ill-conditioned (identifiable)
- MAP more than 3 posterior sd away from any uniform prior bound

## Limits

- `gauss_hermite` uses a tensor grid: 3^z points for z zones (81 for 4 zones). Fine for the
  beam, costly for many zones.
- Matching sd per sensor assumes the Phase 1 responses are roughly normal; the standard
  error of the sample sd uses the normal-theory formula v / sqrt(2 (n - 1)).
- `sensor_data.json` weights must be 1.0 (checked), because the likelihood is iid Gaussian.
