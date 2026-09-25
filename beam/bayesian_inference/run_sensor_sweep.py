r"""Sensor-count sweep. Edits nothing permanently in the pipeline.

Both phases read ../sensor_placement/sensor_data.json, so swapping that one file
drives Phase 1 (which sensors get measured and noised) and Phase 2 (which sensors
the forward model interpolates). No pipeline code changes.

Sensors sit at mid-depth y = 0.05, added tip first then inward. Five runs on a
widening spacing, chosen to trace the saturation curve rather than walk it:
    N=1  -> x = 1.0
    N=2  -> x = 1.0, 0.9
    N=4  -> x = 1.0 ... 0.7
    N=7  -> x = 1.0 ... 0.4
    N=10 -> x = 1.0 ... 0.1

Phase 1 carries two modes. RUN_SINGLE_DETERMINISTIC is flipped to True for the
sweep (one solve at the materials-file E) and restored afterwards -- the sampling
mode run_distribution() draws E from a distribution, which would confound sensor
count with stiffness spread. Sigma may be a NOISE_SIGMA literal or NOISE_FRACTION
* max|u_true|; both are accepted and checked against Phase 2. default_rng fills sequentially,
so sensor k keeps the same draw at every N, and sigma = 0.02 * max|u_true| stays at
the tip value because the tip is always in the list. The assumed sigma in Phase 2
stays pinned at the matched 2% for every run.

sensor_data.json and BayesianParameters.json are restored afterwards, including on
Ctrl-C or a crash, and Phase 1 is re-run once at the end so measured_data.csv goes
back to the frozen N=1 dataset the other sweeps depend on.

Drop next to MainBayesian.py, run from bayesian_inference/:
    python run_sensor_sweep.py
    python run_sensor_sweep.py N04 N07     # re-run single cases
"""
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ARCHIVE = Path(r"D:\KratosProjects\MCMC\Analysis\Sensors")

SIGMA_2PCT = 3.788097e-08
SEED = 20260802
NOISE_FRACTION = 0.02
Z_TIP = -0.952
E_REF = 206.9e9

Y_MID = 0.05
N_MAX = 10
STATIONS = [round(1.0 - 0.1 * i, 1) for i in range(N_MAX)]

# which sensor counts to run -- any subset of 1..N_MAX
SENSOR_COUNTS = [1, 2, 4, 7, 10]

CONFIG = Path("BayesianParameters.json")
MAIN = Path("MainBayesian.py")
PHASE1_DIR = Path("../damaged_system")
PHASE1 = PHASE1_DIR / "MainKratos_phase1.py"
NOISE_MODEL = PHASE1_DIR / "noise_model.json"
MEASURED = PHASE1_DIR / "measured_data.csv"
SENSOR_FILE = Path("../sensor_placement/sensor_data.json")

SEED_RE = re.compile(r"^SEED\s*=\s*(-?\d+)", re.M)
FRAC_RE = re.compile(r"^NOISE_FRACTION\s*=\s*([0-9.eE+-]+)", re.M)
SIGMA_RE = re.compile(r"^NOISE_SIGMA\s*=\s*([0-9.eE+-]+|None)", re.M)
MODE_RE = re.compile(r"^RUN_SINGLE_DETERMINISTIC\s*=\s*(True|False)", re.M)


def patch_phase1(text):
    """Some versions of Phase 1 carry two modes: run_distribution() samples E from a
    distribution, which would confound sensor count with stiffness spread. Where that
    switch exists it is forced to the single deterministic solve for the duration of
    the sweep and restored afterwards. A single-mode Phase 1 is left untouched."""
    if not MODE_RE.search(text):
        return text
    text, n = MODE_RE.subn("RUN_SINGLE_DETERMINISTIC = True", text)
    if n != 1:
        raise RuntimeError(f"expected one RUN_SINGLE_DETERMINISTIC line in {PHASE1.name}, "
                           f"found {n}")
    return text


def sensor_name(x):
    return "disp_y_tip" if x == 1.0 else f"disp_y_x{int(round(x * 1000)):04d}"


def sensor_block(k):
    return {"list_of_sensors": [
        {"type": "displacement_sensor",
         "name": sensor_name(x),
         "location": [x, Y_MID, 0.0],
         "direction": [0.0, 1.0, 0.0],
         "weight": 1.0}
        for x in STATIONS[:k]]}


def check_phase1_source():
    """Phase 1 may set sigma either way:
         NOISE_SIGMA    = <literal>   fixed instrument spec, independent of the sensor set
         NOISE_FRACTION = 0.02        sigma = 0.02 * max|u_true|, so the tip must be present
    Both are accepted; the literal form is preferred here because it cannot drift with N."""
    text = PHASE1.read_text()
    seed = SEED_RE.search(text)
    if not seed:
        raise RuntimeError(f"could not find a SEED line in {PHASE1.name}")
    if int(seed.group(1)) != SEED:
        raise RuntimeError(f"{PHASE1.name} has SEED = {seed.group(1)}, expected {SEED} "
                           f"-- a previous sweep may not have restored it")

    sig = SIGMA_RE.search(text)
    frac = FRAC_RE.search(text)
    if sig and sig.group(1).strip() != "None":
        value = float(sig.group(1))
        if abs(value - SIGMA_2PCT) / SIGMA_2PCT > 1e-3:
            raise RuntimeError(f"{PHASE1.name} has NOISE_SIGMA = {value:.6e}, expected "
                               f"{SIGMA_2PCT:.6e} to match Phase 2")
        return {"mode": "NOISE_SIGMA", "sigma": value, "noise_fraction": None}
    if frac:
        value = float(frac.group(1))
        if abs(value - NOISE_FRACTION) > 1e-12:
            raise RuntimeError(f"{PHASE1.name} has NOISE_FRACTION = {value}, "
                               f"expected {NOISE_FRACTION}")
        return {"mode": "NOISE_FRACTION", "sigma": None, "noise_fraction": value}
    raise RuntimeError(f"{PHASE1.name} sets neither NOISE_SIGMA nor NOISE_FRACTION")


def check_phase1_output(k):
    nm = json.loads(NOISE_MODEL.read_text())
    if nm["seed"] != SEED:
        raise RuntimeError(f"noise_model.json reports seed {nm['seed']}, expected {SEED}")

    u_true = np.asarray(nm["u_true"], float)
    u_hat = np.asarray(nm["u_hat"], float)
    if u_true.size != k:
        raise RuntimeError(f"Phase 1 wrote {u_true.size} sensors, expected {k}")

    if abs(nm["sigma"] - SIGMA_2PCT) / SIGMA_2PCT > 1e-3:
        raise RuntimeError(f"Phase 1 sigma is {nm['sigma']:.6e}, expected {SIGMA_2PCT:.6e} "
                           f"-- the tip should always set the scale")

    z = (u_hat - u_true) / nm["sigma"]
    if abs(z[0] - Z_TIP) > 5e-3:
        raise RuntimeError(f"tip draw is z = {z[0]:.4f}, expected {Z_TIP:.4f} "
                           f"-- the noise stream is not stable across N")
    return u_true.tolist(), z.tolist()


def set_outputs(cfg, dest):
    for proc in cfg["output_processes"]:
        proc["Parameters"]["output_path"] = str(
            dest / "vtk_output" if "vtk" in proc["python_module"] else dest)


def main():
    for required in (CONFIG, MAIN, PHASE1, SENSOR_FILE):
        if not required.exists():
            sys.exit(f"run this from bayesian_inference/ -- {required} not found")

    if not SENSOR_COUNTS or max(SENSOR_COUNTS) > N_MAX or min(SENSOR_COUNTS) < 1:
        sys.exit(f"SENSOR_COUNTS must be a non-empty subset of 1..{N_MAX}")
    tags = [f"N{k:02d}" for k in SENSOR_COUNTS]
    wanted = set(sys.argv[1:])
    if wanted:
        unknown = wanted - set(tags)
        if unknown:
            sys.exit(f"unknown tag(s): {', '.join(sorted(unknown))}\n"
                     f"available: {', '.join(tags)}")
        print(f"running only: {', '.join(sorted(wanted))}")

    phase1_noise = check_phase1_source()
    phase1_source = PHASE1.read_text()
    patch_phase1(phase1_source)          # dry run before touching anything
    print(f"Phase 1 noise: sigma from {phase1_noise['mode']}")

    base_config = json.loads(CONFIG.read_text())
    sigma = base_config["likelihood"]["noise_model"]["sigma"]
    if abs(sigma - SIGMA_2PCT) / SIGMA_2PCT > 1e-3:
        sys.exit(f"Phase 2 sigma is {sigma:.6e}, expected the matched {SIGMA_2PCT:.6e}\n"
                 f"fix BayesianParameters.json before sweeping")

    prior = base_config["parameters"][0]["prior"]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    cfg_backup = CONFIG.with_suffix(CONFIG.suffix + ".sweep_backup")
    sensor_backup = SENSOR_FILE.with_suffix(SENSOR_FILE.suffix + ".sweep_backup")
    p1_backup = PHASE1.with_suffix(PHASE1.suffix + ".sweep_backup")
    shutil.copy(CONFIG, cfg_backup)
    shutil.copy(SENSOR_FILE, sensor_backup)
    shutil.copy(PHASE1, p1_backup)
    PHASE1.write_text(patch_phase1(phase1_source))

    results = []
    try:
        for i, (k, tag) in enumerate(zip(SENSOR_COUNTS, tags), 1):
            if wanted and tag not in wanted:
                continue
            dest = ARCHIVE / f"output_{tag}"
            dest.mkdir(parents=True, exist_ok=True)
            stations = STATIONS[:k]
            print(f"\n{'=' * 70}\n  [{i}/{len(SENSOR_COUNTS)}] {k} sensor{'s' if k > 1 else ''}   "
                  f"x = {', '.join(format(x, '.1f') for x in stations)}\n"
                  f"  -> {dest}\n{'=' * 70}", flush=True)
            started = datetime.now()

            SENSOR_FILE.write_text(json.dumps(sensor_block(k), indent=4))
            code = subprocess.run([sys.executable, PHASE1.name], cwd=PHASE1_DIR).returncode
            if code != 0:
                raise RuntimeError(f"Phase 1 exited with {code}")
            u_true, z = check_phase1_output(k)

            cfg = json.loads(json.dumps(base_config))
            set_outputs(cfg, dest)
            cfg["problem_data"]["problem_name"] = f"bayesian_inference_beam_{tag}"
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
                "label": tag,
                "n_sensors": k,
                "stations": stations,
                "seed": SEED,
                "z": z,
                "u_true": u_true,
                "phase1_noise_mode": phase1_noise["mode"],
                "noise_fraction_data": phase1_noise["noise_fraction"],
                "sigma_assumed": sigma,
                "sigma_data": SIGMA_2PCT,
                "e_ref": E_REF,
                "prior_type": prior["type"],
                "prior_parameters": prior["parameters"],
                "returncode": code,
                "wall_time_s": round(elapsed, 1),
                "started": started.isoformat(timespec="seconds"),
            }, indent=2))
            for src in (NOISE_MODEL, MEASURED, SENSOR_FILE):
                shutil.copy(src, dest / src.name)
            shutil.copy(CONFIG, dest / "BayesianParameters.json")

            results.append((tag, code, elapsed))
            print(f"  {'done' if code == 0 else 'FAILED (exit %d)' % code} in {elapsed:.0f} s")

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        shutil.move(str(cfg_backup), str(CONFIG))
        shutil.move(str(sensor_backup), str(SENSOR_FILE))
        print(f"restored {CONFIG} and {SENSOR_FILE}")
        print("re-running Phase 1 to restore the frozen dataset ...", flush=True)
        code = subprocess.run([sys.executable, PHASE1.name], cwd=PHASE1_DIR).returncode
        print("measured_data.csv restored" if code == 0
              else f"*** Phase 1 restore run FAILED (exit {code}) -- re-run it by hand ***")
        shutil.move(str(p1_backup), str(PHASE1))
        print(f"restored {PHASE1}")

    ok = sum(1 for _, code, _ in results if code == 0)
    print(f"\n{ok}/{len(results)} runs completed")
    for tag, code, elapsed in results:
        print(f"  {tag:<6} {'ok' if code == 0 else 'FAILED':<7} {elapsed:>7.0f} s")
    print(f"archive: {ARCHIVE}")


if __name__ == "__main__":
    main()
