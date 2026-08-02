"""Plain Metropolis sampler — validation sampler and producer of the calculation table.

- Symmetric Gaussian random walk in physical space (theta = alpha vector).
- Uses log_likelihood + log_prior (unlike SMC_aCS, which must get likelihood only).
- Works in log space throughout: with 4 sensors and small sigma, plain scores
  underflow to exactly 0.0 in double precision. A few rows are converted back to
  plain numbers in the preview so they match the worked example in the notes.
- One table row per PROPOSAL; all model columns refer to the proposed value.
  A rejection re-votes for the current position. The first n_burn rows are burn-in.
- Welford accumulators produce burnin.vtk / converged.vtk / posterior.vtk.
"""
import numpy as np

from field_stats import FieldStats


def run_metropolis(likelihood, prior, settings, rng, field_hook=None):
    """field_hook(theta) -> (disp_field or None, E_vector) for the VTK accumulators."""
    n_steps = int(settings["n_steps"])
    n_burn = int(settings["n_burn"])
    prop_std = np.atleast_1d(np.asarray(settings["proposal_std"], float))
    theta = np.atleast_1d(np.asarray(settings["initial_theta"], float))

    def log_w(th):
        # log weight = log likelihood + log prior  (Metropolis needs both)
        return likelihood.log_likelihood(th), prior.log_pdf(th)

    logL_cur, logP_cur = log_w(theta)
    logw_cur = logL_cur + logP_cur

    rows = []
    chain = np.zeros((n_steps, theta.size))
    accepted_post_burn = 0

    stats = {"burn": (FieldStats(), FieldStats()),
             "conv": (FieldStats(), FieldStats())}

    for it in range(n_steps + n_burn):
        burn = it < n_burn
        theta_prop = theta + rng.normal(0.0, prop_std)

        u_prop = likelihood.forward(theta_prop)
        r = likelihood.u_hat - u_prop
        norm2 = float(r @ r)
        logL_prop = likelihood.log_likelihood(theta_prop)
        logP_prop = prior.log_pdf(theta_prop)
        logw_prop = logL_prop + logP_prop
        logR = logw_prop - logw_cur

        dice = rng.uniform()
        accept = (logR >= 0.0) or (np.log(dice) < logR)
        if accept:
            theta = theta_prop
            logw_cur = logw_prop
        # a rejection re-votes for the current position

        rows.append(_row(it + 1, burn, theta_prop, u_prop, r, norm2,
                         logL_prop, logP_prop, logR, dice, accept, theta))

        if not burn:
            chain[it - n_burn] = theta
            accepted_post_burn += int(accept)

        if field_hook is not None:
            disp, E = field_hook(theta)
            seg = "burn" if burn else "conv"
            if disp is not None:
                stats[seg][0].update(disp)
                stats[seg][1].update(E)

    acc_rate = accepted_post_burn / n_steps
    return chain, rows, acc_rate, stats


def _row(it, burn, theta_prop, u_prop, r, norm2, logL, logP, logR, dice, accept, position):
    row = {"It": it, "phase": "burn-in" if burn else "sample"}
    for j, a in enumerate(np.atleast_1d(theta_prop), 1):
        row[f"prop_alpha_{j}"] = a
    for j, u in enumerate(u_prop, 1):
        row[f"u{j}"] = u
    for j, rr in enumerate(r, 1):
        row[f"r{j}"] = rr
    row.update({"norm_r_sq": norm2, "log_L": logL, "log_prior": logP,
                "log_w": logL + logP, "log_R": logR, "dice": dice,
                "decision": "accept" if accept else "reject"})
    for j, a in enumerate(np.atleast_1d(position), 1):
        row[f"position_alpha_{j}"] = a
    return row


def write_table(rows, csv_path, preview_rows=12):
    """Full CSV + a plain-number preview of the first rows (score/weight/ratio
    converted out of log space so they can be checked against the notes)."""
    import csv as _csv
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines = ["First proposals converted back to plain numbers "
             "(score = exp(log L), ratio = exp(log R)):", ""]
    hdr = f"{'It':>4} {'phase':>8} {'prop a':>8} {'||r||^2':>11} {'log L':>10} " \
          f"{'score':>10} {'ratio':>10} {'dice':>6} {'decision':>8} {'position':>9}"
    lines.append(hdr)
    for row in rows[:preview_rows]:
        score = np.exp(row["log_L"])
        ratio = np.exp(min(row["log_R"], 700.0))
        lines.append(
            f"{row['It']:>4} {row['phase']:>8} {row['prop_alpha_1']:>8.4f} "
            f"{row['norm_r_sq']:>11.3e} {row['log_L']:>10.3f} {score:>10.3e} "
            f"{ratio:>10.3e} {row['dice']:>6.3f} {row['decision']:>8} "
            f"{row['position_alpha_1']:>9.4f}")
    preview = "\n".join(lines)
    with open(str(csv_path).replace(".csv", "_preview.txt"), "w") as f:
        f.write(preview + "\n")
    return preview
