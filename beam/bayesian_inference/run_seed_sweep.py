r"""Noise-seed sweep. Edits nothing permanently in the pipeline.

The first sweep that re-runs Phase 1 per case: the seed in MainKratos_phase1.py is
patched so each run gets a different draw from the same N(0, sigma). One case sets
NOISE_FRACTION to 0.0 for a noise-free reference. Phase 2 is untouched apart from
output paths -- the assumed sigma stays pinned at the matched 2% for every run, and
the sampler seed stays fixed, so the only thing varying is the Phase 1 noise draw.

Both source files are restored afterwards, including on Ctrl-C or a crash.

Drop next to MainBayesian.py, run from bayesian_inference/:
    python run_seed_sweep.py
"""
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ARCHIVE = Path(r"D:\KratosProjects\MCMC\Analysis\Noise_Seed")

SIGMA_2PCT = 3.788097e-08
NOISE_FRACTION = 0.02

# (tag, seed, noise_fraction, expected z)  -- ordered by z
RUNS = [
    ("z-1.97", 51182162, NOISE_FRACTION, -1.973),
    ("z-0.95", 20260802, NOISE_FRACTION, -0.952),   # the frozen dataset all other sweeps used
    ("z-0.83", 47318869, NOISE_FRACTION, -0.834),
    ("z+0.00", 343805, NOISE_FRACTION, 0.000),      # draw is -6.4e-06 sigma: noise-free to 7 digits
    ("z+0.72", 75516750, NOISE_FRACTION, 0.723),
    ("z+1.76", 14415961, NOISE_FRACTION, 1.763),
    ("z+2.39", 3485256, NOISE_FRACTION, 2.385),
]

CONFIG = Path("BayesianParameters.json")
MAIN = Path("MainBayesian.py")
PHASE1_DIR = Path("../damaged_system")
PHASE1 = PHASE1_DIR / "MainKratos_phase1.py"
NOISE_MODEL = PHASE1_DIR / "noise_model.json"
MEASURED = PHASE1_DIR / "measured_data.csv"

SEED_RE = re.compile(r"^(SEED\s*=\s*)(-?\d+)", re.M)
FRAC_RE = re.compile(r"^(NOISE_FRACTION\s*=\s*)([0-9.eE+-]+)", re.M)


def patch_phase1(text, seed, fraction):
    text, n1 = SEED_RE.subn(lambda m: f"{m.group(1)}{seed}", text)
    text, n2 = FRAC_RE.subn(lambda m: f"{m.group(1)}{fraction}", text)
    if (n1, n2) != (1, 1):
        raise RuntimeError(f"expected one SEED and one NOISE_FRACTION line in {PHASE1.name}, "
                           f"found {n1} and {n2}")
    return text


def set_outputs(cfg, dest):
    for proc in cfg["output_processes"]:
        proc["Parameters"]["output_path"] = str(
            dest / "vtk_output" if "vtk" in proc["python_module"] else dest)


def check_phase1(seed, fraction, z_expected):
    nm = json.loads(NOISE_MODEL.read_text())
    if nm["seed"] != seed:
        raise RuntimeError(f"noise_model.json reports seed {nm['seed']}, expected {seed}")

    u_true = np.asarray(nm["u_true"], float)
    u_hat = np.asarray(nm["u_hat"], float)

    if fraction == 0.0:
        if nm["sigma"] != 0.0 or not np.array_equal(u_true, u_hat):
            raise RuntimeError("noise-free run still carries noise")
        return 0.0

    if abs(nm["sigma"] - SIGMA_2PCT) / SIGMA_2PCT > 1e-3:
        raise RuntimeError(f"Phase 1 sigma is {nm['sigma']:.6e}, expected {SIGMA_2PCT:.6e}")
    z = float(((u_hat - u_true) / nm["sigma"]).mean())
    if abs(z - z_expected) > 5e-3:
        raise RuntimeError(f"draw is z = {z:.4f}, expected {z_expected:.4f}")
    return z


def main():
    for required in (CONFIG, MAIN, PHASE1):
        if not required.exists():
            sys.exit(f"run this from bayesian_inference/ -- {required} not found")

    wanted = set(sys.argv[1:])
    if wanted:
        unknown = wanted - {tag for tag, *_ in RUNS}
        if unknown:
            sys.exit(f"unknown tag(s): {', '.join(sorted(unknown))}\n"
                     f"available: {', '.join(tag for tag, *_ in RUNS)}")
        print(f"running only: {', '.join(sorted(wanted))}")

    base_config = json.loads(CONFIG.read_text())
    sigma = base_config["likelihood"]["noise_model"]["sigma"]
    if abs(sigma - SIGMA_2PCT) / SIGMA_2PCT > 1e-3:
        sys.exit(f"Phase 2 sigma is {sigma:.6e}, expected the matched {SIGMA_2PCT:.6e}\n"
                 f"fix BayesianParameters.json before sweeping")

    phase1_source = PHASE1.read_text()
    patch_phase1(phase1_source, RUNS[0][1], RUNS[0][2])      # dry run before touching anything

    prior = base_config["parameters"][0]["prior"]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    cfg_backup = CONFIG.with_suffix(CONFIG.suffix + ".sweep_backup")
    p1_backup = PHASE1.with_suffix(PHASE1.suffix + ".sweep_backup")
    shutil.copy(CONFIG, cfg_backup)
    shutil.copy(PHASE1, p1_backup)

    results = []
    try:
        for i, (tag, seed, fraction, z_expected) in enumerate(RUNS, 1):
            if wanted and tag not in wanted:
                continue
            dest = ARCHIVE / f"output_{i}_{tag}"
            dest.mkdir(parents=True, exist_ok=True)
            print(f"\n{'=' * 66}\n  [{i}/{len(RUNS)}] seed {seed}   noise {100 * fraction:g}%   "
                  f"z = {z_expected:+.3f}  ->  {dest}\n{'=' * 66}", flush=True)
            started = datetime.now()

            PHASE1.write_text(patch_phase1(phase1_source, seed, fraction))
            code = subprocess.run([sys.executable, PHASE1.name], cwd=PHASE1_DIR).returncode
            if code != 0:
                raise RuntimeError(f"Phase 1 exited with {code}")
            z = check_phase1(seed, fraction, z_expected)

            cfg = json.loads(json.dumps(base_config))
            set_outputs(cfg, dest)
            cfg["problem_data"]["problem_name"] = f"bayesian_inference_beam_seed_{tag}"
            CONFIG.write_text(json.dumps(cfg, indent=4))

            code = subprocess.run([sys.executable, str(MAIN)]).returncode
            elapsed = (datetime.now() - started).total_seconds()
            if code != 0:
                print(f"\n  *** Phase 2 FAILED (exit {code}) for {tag} -- no summary.json will "
                      f"be written and this case will be missing from the table ***\n", flush=True)
            elif not (dest / "summary.json").exists():
                print(f"\n  *** Phase 2 exited cleanly but wrote no summary.json in {dest} ***\n",
                      flush=True)

            (dest / "run_info.json").write_text(json.dumps({
                "label": tag if fraction else "no noise",
                "seed": seed,
                "z": z,
                "noise_fraction_data": fraction,
                "sigma_assumed": sigma,
                "sigma_data": SIGMA_2PCT if fraction else 0.0,
                "e_ref": 206.9e9,
                "prior_type": prior["type"],
                "prior_parameters": prior["parameters"],
                "returncode": code,
                "wall_time_s": round(elapsed, 1),
                "started": started.isoformat(timespec="seconds"),
            }, indent=2))
            for src in (NOISE_MODEL, MEASURED):
                shutil.copy(src, dest / src.name)
            shutil.copy(CONFIG, dest / "BayesianParameters.json")

            results.append((tag, code, elapsed))
            print(f"  {'done' if code == 0 else 'FAILED (exit %d)' % code} in {elapsed:.0f} s")

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        shutil.move(str(cfg_backup), str(CONFIG))
        shutil.move(str(p1_backup), str(PHASE1))
        print(f"restored {CONFIG} and {PHASE1}")

    ok = sum(1 for _, code, _ in results if code == 0)
    print(f"\n{ok}/{len(RUNS)} runs completed")
    for tag, code, elapsed in results:
        print(f"  {tag:<10} {'ok' if code == 0 else 'FAILED':<7} {elapsed:>7.0f} s")
    print(f"archive: {ARCHIVE}")
    print("\nPhase 1 is back on its original source -- re-run it once to restore "
          "measured_data.csv for the other sweeps.")


if __name__ == "__main__":
    main()
