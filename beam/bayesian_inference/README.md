# Bayesian inference of Young's modulus — beam Phase 2

JSON-configured, Kratos-native MCMC pipeline. The forward solver is called at every
sample; nothing is hardcoded; the model is built once and evaluated many times.

## Run

```bash
python MainBayesian.py                                   # uses BayesianParameters.json
python MainBayesian.py --sampler smc_acs                 # one-line sampler switch
python MainBayesian.py --forward toy --sampler metropolis  # Stage-A validation mode
```

Everything lives in `BayesianParameters.json`: parameter definitions (target sub
model part, control variable, reference value, prior), sensor + measured data files,
noise sigma, and per-sampler settings. A second damage zone is a second entry under
`"parameters"` (requires a zone-partitioned mdpa, as in the plate project).

## Layout

| File | Role |
|---|---|
| `MainBayesian.py` | orchestration: config -> forward model -> sampler -> outputs |
| `kratos_forward_model.py` | **the deliverable**: build-once / evaluate-many wrapper |
| `toy_forward_model.py` | same interface, `u = u_ref/alpha`; sampler validation only |
| `likelihood.py` | likelihood-ONLY log-likelihood (SMC contract), prior, eval cache |
| `metropolis.py` | plain Metropolis + calculation table (log space, plain-number preview) |
| `closed_form.py` | Gaussian-in-`s` closed form + exact quadrature ground truth |
| `field_stats.py` | Welford running mean/std for displacement and E fields |
| `vtk_writer.py` | legacy-VTK writer for DISPLACEMENT_MEAN/STD, YOUNG_MODULUS_MEAN/STD |
| `era/` | unmodified ERA code (SMC_aCS, SMC_GM, EMGM, Distributions) + license |

## Sampler contracts (do not mix)

- plain Metropolis <- `log_likelihood + log_prior`
- ERA `SMC_aCS`    <- `log_likelihood` ONLY. The ERADist/ERANataf object holds the
  prior; the pCN proposal preserves it, so it cancels from the acceptance ratio.
  Passing lik+prior double-counts the prior.
- The log-likelihood handed to SMC_aCS must return a **plain Python float**:
  with numpy >= 2, a length-1 array crashes `logLk[k] = ...` inside SMC_aCS.
- `SMC_aCS` option `opc='b'` is broken upstream; the default `'a'` is used.

## Per-level VTK without touching SMC_aCS

The likelihood caches every (theta -> sensors, displacement field) evaluation.
`SMC_aCS` returns its particle populations per tempering level (`samplesX`), so the
per-level statistics are assembled *after* sampling by cache lookup (hits are
guaranteed — every particle was evaluated). ERA code stays a cited black box.

`u(E[alpha]) != E[u(alpha)]`: all VTK statistics come from the Welford accumulator
over samples, never from a solve at the posterior mean.

VtkOutputProcess cannot be used for the statistics fields because it only outputs
*registered* (compiled) Kratos variables and `DISPLACEMENT_MEAN` etc. do not exist;
`vtk_writer.py` emits the same mesh with correctly named arrays (legacy ASCII,
loads in ParaView next to the deterministic Phase-2 output).

## Hard-won correctness notes

1. **Pin the linear solver.** With no `linear_solver_settings`, Kratos auto-picks
   the "fastest available" solver, which silently returned garbage (relative error
   up to 24, no exception) for some stiffness values in this repeated-solve setting.
   `skyline_lu_factorization` is pinned in `PrimalParametersBayes.json` and gives
   machine-precision results across 400 solves in alpha in [0.25, 1.95]. Revisit the
   choice (e.g. `sparse_lu`) for realistic meshes, but never leave it implicit.
2. **Clear the strategy every sample.** `solver.Clear()` before each re-solve forces
   a clean rebuild of DOF set, LHS and RHS. Without it, both `reform_dofs` settings
   eventually corrupt state (stale LHS drift, or a structurally singular system).
   At 2-3 ms per solve on this mesh the cost is irrelevant; it removes the single
   most dangerous failure mode of the build-once design.
3. **The self-check runs at construction** (`self_check_on_init`): determinism,
   theta-sensitivity, nonzero/non-NaN sensors, timing. It exists because failure
   modes 1 and 2 are invisible downstream — the sampler would happily explore a
   wrong posterior.

## Validation results (this configuration)

Exact posterior (quadrature, uniform prior [0.2, 2.0]): alpha = 0.7057 +/- 0.0120.

| Run | mean | std | extra |
|---|---|---|---|
| toy + Metropolis (2000 counted) | 0.7058 | 0.0122 | acc 0.470 |
| toy + SMC_aCS (N=1000) | 0.7066 | 0.0123 | logcE = -5.599 |
| Kratos + Metropolis | 0.7058 | 0.0122 | 2204 solves, 2.2 ms each |
| Kratos + SMC_aCS | 0.7066 | 0.0123 | 10004 solves, 4 levels |

Kratos and toy runs are bit-identical for equal seeds because the two forward maps
agree to ~1e-12 — which simultaneously validates the sampler layer and the wrapper.
Closed form in s = 1/alpha: 1.4183 +/- 0.0241 vs sampled 1.4173 +/- 0.0245.
