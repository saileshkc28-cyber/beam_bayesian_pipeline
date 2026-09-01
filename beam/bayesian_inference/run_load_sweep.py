r"""One-off load sweep. Edits nothing in the pipeline.

Both configs are patched in place, MainBayesian.py is run as-is, and the
originals are restored afterwards -- including on Ctrl-C or a crash.

Drop next to MainBayesian.py, run from bayesian_inference/:
    python run_load_sweep.py
"""
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ARCHIVE = Path(r"D:\KratosProjects\MCMC\Analysis\Load")
LOADS = [900.0, 990.0, 1000.0, 1010.0, 1100.0]

CONFIG = Path("BayesianParameters.json")      # opened by name inside MainBayesian.py
PRIMAL = Path("PrimalParametersBayes.json")
MAIN = Path("MainBayesian.py")


def set_load(primal, value):
    for entry in primal["processes"]["loads_process_list"]:
        params = entry["Parameters"]
        if params.get("variable_name") == "LINE_LOAD":
            params["modulus"] = value
            return
    raise RuntimeError("no LINE_LOAD process found in " + str(PRIMAL))


def set_outputs(cfg, dest):
    for proc in cfg["output_processes"]:
        proc["Parameters"]["output_path"] = str(
            dest / "vtk_output" if "vtk" in proc["python_module"] else dest)


def main():
    for required in (CONFIG, PRIMAL, MAIN):
        if not required.exists():
            sys.exit(f"run this from bayesian_inference/ -- {required} not found")

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    backups = {}
    for original in (CONFIG, PRIMAL):
        backup = original.with_suffix(original.suffix + ".sweep_backup")
        shutil.copy(original, backup)
        backups[original] = backup

    base_config = json.loads(CONFIG.read_text())
    base_primal = json.loads(PRIMAL.read_text())
    sigma = base_config["likelihood"]["noise_model"]["sigma"]
    results = []

    try:
        for load in LOADS:
            dest = ARCHIVE / f"output_{int(load)}"
            dest.mkdir(parents=True, exist_ok=True)

            primal = json.loads(json.dumps(base_primal))
            set_load(primal, load)
            PRIMAL.write_text(json.dumps(primal, indent=4))

            cfg = json.loads(json.dumps(base_config))
            set_outputs(cfg, dest)
            cfg["problem_data"]["problem_name"] = f"bayesian_inference_beam_load_{int(load)}"
            CONFIG.write_text(json.dumps(cfg, indent=4))

            print(f"\n{'=' * 60}\n  load = {load}  ->  {dest}\n{'=' * 60}", flush=True)
            started = datetime.now()
            code = subprocess.run([sys.executable, str(MAIN)]).returncode
            elapsed = (datetime.now() - started).total_seconds()

            (dest / "run_info.json").write_text(json.dumps({
                "load_modulus": load,
                "sigma_assumed": sigma,
                "returncode": code,
                "wall_time_s": round(elapsed, 1),
                "started": started.isoformat(timespec="seconds"),
            }, indent=2))
            shutil.copy(CONFIG, dest / "BayesianParameters.json")
            shutil.copy(PRIMAL, dest / "PrimalParametersBayes.json")

            results.append((load, code, elapsed))
            print(f"  {'done' if code == 0 else 'FAILED (exit %d)' % code} in {elapsed:.0f} s")

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        for original, backup in backups.items():
            shutil.move(str(backup), str(original))
            print(f"restored {original}")

    ok = sum(1 for _, code, _ in results if code == 0)
    print(f"\n{ok}/{len(LOADS)} runs completed")
    for load, code, elapsed in results:
        print(f"  {int(load):>5}  {'ok' if code == 0 else 'FAILED':<7} {elapsed:>7.0f} s")
    print(f"archive: {ARCHIVE}")


if __name__ == "__main__":
    main()
