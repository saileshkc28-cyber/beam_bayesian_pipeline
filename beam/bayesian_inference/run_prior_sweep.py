r"""One-off prior sweep. Edits nothing in the pipeline.

Only BayesianParameters.json is patched (prior block + output paths);
PrimalParametersBayes.json is not touched. The original config is restored
afterwards, including on Ctrl-C or a crash.

Drop next to MainBayesian.py, run from bayesian_inference/:
    python run_prior_sweep.py
"""
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ARCHIVE = Path(r"D:\KratosProjects\MCMC\Analysis\Prior")

# (folder tag, ERADist type, PAR parameters)
PRIORS = [
    ("unif_0.2_2.0", "uniform", [0.2, 2.0]),      # baseline, current config
    ("unif_0.8_1.2", "uniform", [0.8, 1.2]),      # narrow, still contains the posterior
    ("unif_0.5_1.5", "uniform", [0.5, 1.5]),
    ("unif_0.1_3.0", "uniform", [0.1, 3.0]),      # wide
    ("norm_1.0_0.1", "normal", [1.0, 0.1]),       # informative, centred on truth
]

CONFIG = Path("BayesianParameters.json")      # opened by name inside MainBayesian.py
MAIN = Path("MainBayesian.py")


def set_prior(cfg, prior_type, params):
    entries = cfg["parameters"]
    if len(entries) != 1:
        raise RuntimeError("expected exactly one inferred parameter, found %d" % len(entries))
    entries[0]["prior"] = {"type": prior_type, "parameters": params}


def set_outputs(cfg, dest):
    for proc in cfg["output_processes"]:
        proc["Parameters"]["output_path"] = str(
            dest / "vtk_output" if "vtk" in proc["python_module"] else dest)


def main():
    for required in (CONFIG, MAIN):
        if not required.exists():
            sys.exit(f"run this from bayesian_inference/ -- {required} not found")

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    backup = CONFIG.with_suffix(CONFIG.suffix + ".sweep_backup")
    shutil.copy(CONFIG, backup)

    base_config = json.loads(CONFIG.read_text())
    sigma = base_config["likelihood"]["noise_model"]["sigma"]
    results = []

    try:
        for tag, prior_type, params in PRIORS:
            dest = ARCHIVE / f"output_{tag}"
            dest.mkdir(parents=True, exist_ok=True)

            cfg = json.loads(json.dumps(base_config))
            set_prior(cfg, prior_type, params)
            set_outputs(cfg, dest)
            cfg["problem_data"]["problem_name"] = f"bayesian_inference_beam_prior_{tag}"
            CONFIG.write_text(json.dumps(cfg, indent=4))

            print(f"\n{'=' * 60}\n  prior = {prior_type} {params}  ->  {dest}\n{'=' * 60}",
                  flush=True)
            started = datetime.now()
            code = subprocess.run([sys.executable, str(MAIN)]).returncode
            elapsed = (datetime.now() - started).total_seconds()

            (dest / "run_info.json").write_text(json.dumps({
                "label": f"{prior_type}{params}",
                "prior_type": prior_type,
                "prior_parameters": params,
                "sigma_assumed": sigma,
                "returncode": code,
                "wall_time_s": round(elapsed, 1),
                "started": started.isoformat(timespec="seconds"),
            }, indent=2))
            shutil.copy(CONFIG, dest / "BayesianParameters.json")

            results.append((tag, code, elapsed))
            print(f"  {'done' if code == 0 else 'FAILED (exit %d)' % code} in {elapsed:.0f} s")

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        shutil.move(str(backup), str(CONFIG))
        print(f"restored {CONFIG}")

    ok = sum(1 for _, code, _ in results if code == 0)
    print(f"\n{ok}/{len(PRIORS)} runs completed")
    for tag, code, elapsed in results:
        print(f"  {tag:<16} {'ok' if code == 0 else 'FAILED':<7} {elapsed:>7.0f} s")
    print(f"archive: {ARCHIVE}")


if __name__ == "__main__":
    main()
