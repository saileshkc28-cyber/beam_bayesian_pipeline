r"""One-off E_ref sweep. Edits nothing in the pipeline.

Only BayesianParameters.json is patched: an explicit "reference_value" is
added to the inferred parameter (kratos_forward_model.py already honours it,
overriding the value StructuralMaterialsBayes.json assigns), plus the output
paths. The original config is restored afterwards, including on Ctrl-C.

Drop next to MainBayesian.py, run from bayesian_inference/:
    python run_eref_sweep.py
"""
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ARCHIVE = Path(r"D:\KratosProjects\MCMC\Analysis\E_ref")

# (folder tag, reference Young's modulus in Pa)
E_REFS = [
    ("180GPa", 180.0e9),
    ("200GPa", 200.0e9),
   ("206.9GPa", 206.9e9),      # baseline, matches StructuralMaterialsBayes.json
    ("210GPa", 210.0e9),
    ("230GPa", 230.0e9),
]

CONFIG = Path("BayesianParameters.json")      # opened by name inside MainBayesian.py
MAIN = Path("MainBayesian.py")


def set_reference(cfg, e_ref):
    entries = cfg["parameters"]
    if len(entries) != 1:
        raise RuntimeError("expected exactly one inferred parameter, found %d" % len(entries))
    entries[0]["reference_value"] = e_ref


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
    prior = base_config["parameters"][0]["prior"]
    results = []

    try:
        for tag, e_ref in E_REFS:
            dest = ARCHIVE / f"output_{tag}"
            dest.mkdir(parents=True, exist_ok=True)

            cfg = json.loads(json.dumps(base_config))
            set_reference(cfg, e_ref)
            set_outputs(cfg, dest)
            cfg["problem_data"]["problem_name"] = f"bayesian_inference_beam_eref_{tag}"
            CONFIG.write_text(json.dumps(cfg, indent=4))

            print(f"\n{'=' * 60}\n  E_ref = {e_ref:.4e} Pa  ->  {dest}\n{'=' * 60}", flush=True)
            started = datetime.now()
            code = subprocess.run([sys.executable, str(MAIN)]).returncode
            elapsed = (datetime.now() - started).total_seconds()

            (dest / "run_info.json").write_text(json.dumps({
                "label": tag,
                "e_ref": e_ref,
                "sigma_assumed": sigma,
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
    print(f"\n{ok}/{len(E_REFS)} runs completed")
    for tag, code, elapsed in results:
        print(f"  {tag:<12} {'ok' if code == 0 else 'FAILED':<7} {elapsed:>7.0f} s")
    print(f"archive: {ARCHIVE}")


if __name__ == "__main__":
    main()
