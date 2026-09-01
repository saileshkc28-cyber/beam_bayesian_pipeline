r"""One-off assumed-noise sweep. Edits nothing in the pipeline.

Only BayesianParameters.json is patched (noise sigma + output paths);
PrimalParametersBayes.json and damaged_system/noise_model.json are not touched,
so the data keeps its true 2% noise and only the analyst's assumption varies.
The original config is restored afterwards, including on Ctrl-C or a crash.

Drop next to MainBayesian.py, run from bayesian_inference/:
    python run_noise_sweep.py
"""
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ARCHIVE = Path(r"D:\KratosProjects\MCMC\Analysis\Noise")

SIGMA_TRUE = 3.788097e-08     # 2% of |u_true|max, the value the data was generated with

# (folder tag, multiplier of the true sigma)
FACTORS = [
    ("sig_0.5x", 0.5),        # over-confident, half the true noise
    ("sig_0.9x", 0.9),
    ("sig_1.0x", 1.0),        # matched, baseline
    ("sig_1.1x", 1.1),
    ("sig_2.0x", 2.0),        # under-confident, double the true noise
]

CONFIG = Path("BayesianParameters.json")      # opened by name inside MainBayesian.py
MAIN = Path("MainBayesian.py")


def set_sigma(cfg, sigma):
    cfg["likelihood"]["noise_model"]["sigma"] = sigma


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
    base_sigma = base_config["likelihood"]["noise_model"]["sigma"]
    if abs(base_sigma - SIGMA_TRUE) / SIGMA_TRUE > 1e-3:
        sys.exit(f"config sigma is {base_sigma:.6e}, expected the matched "
                 f"{SIGMA_TRUE:.6e}\nfix BayesianParameters.json before sweeping")

    prior = base_config["parameters"][0]["prior"]
    results = []

    try:
        for tag, factor in FACTORS:
            sigma = factor * SIGMA_TRUE
            dest = ARCHIVE / f"output_{tag}"
            dest.mkdir(parents=True, exist_ok=True)

            cfg = json.loads(json.dumps(base_config))
            set_sigma(cfg, sigma)
            set_outputs(cfg, dest)
            cfg["problem_data"]["problem_name"] = f"bayesian_inference_beam_noise_{tag}"
            CONFIG.write_text(json.dumps(cfg, indent=4))

            print(f"\n{'=' * 60}\n  sigma = {sigma:.6e}  ({factor:g}x true, "
                  f"{2.0 * factor:.1f}% of |u|max)  ->  {dest}\n{'=' * 60}", flush=True)
            started = datetime.now()
            code = subprocess.run([sys.executable, str(MAIN)]).returncode
            elapsed = (datetime.now() - started).total_seconds()

            (dest / "run_info.json").write_text(json.dumps({
                "label": f"{2.0 * factor:.1f}%",
                "sigma_assumed": sigma,
                "sigma_true": SIGMA_TRUE,
                "sigma_factor": factor,
                "noise_fraction_assumed": 0.02 * factor,
                "prior_type": prior["type"],
                "prior_parameters": prior["parameters"],
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
    print(f"\n{ok}/{len(FACTORS)} runs completed")
    for tag, code, elapsed in results:
        print(f"  {tag:<16} {'ok' if code == 0 else 'FAILED':<7} {elapsed:>7.0f} s")
    print(f"archive: {ARCHIVE}")


if __name__ == "__main__":
    main()
