r"""One-off E_ref sweep, materials-file version. Edits nothing permanently.

Patches StructuralMaterialsBayes.json (the assumed Young's modulus that
Phase 2 builds its model with, and which kratos_forward_model.py reads as the
reference) plus the output paths in BayesianParameters.json. Both originals
are restored afterwards, including on Ctrl-C.

Phase 1 is never touched: damaged_system/StructuralMaterials.json holds the
truth and is a different file in a different folder.

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

# (folder tag, assumed Young's modulus in Pa)
E_REFS = [
    ("180GPa", 180.0e9),
    ("200GPa", 200.0e9),
    ("206.9GPa", 206.9e9),      # baseline, the value currently in the file
    ("210GPa", 210.0e9),
    ("230GPa", 230.0e9),
]

CONFIG = Path("BayesianParameters.json")        # opened by name inside MainBayesian.py
MATERIALS = Path("StructuralMaterialsBayes.json")
MAIN = Path("MainBayesian.py")
TARGET_PART = "Structure.Beam_Auto1"


def set_modulus(materials, value):
    for block in materials["properties"]:
        if block["model_part_name"] == TARGET_PART:
            variables = block["Material"]["Variables"]
            if "YOUNG_MODULUS" not in variables:
                raise RuntimeError(f"{TARGET_PART} has no YOUNG_MODULUS")
            variables["YOUNG_MODULUS"] = value
            return
    raise RuntimeError(f"{TARGET_PART} not found in {MATERIALS}")


def set_outputs(cfg, dest):
    for proc in cfg["output_processes"]:
        proc["Parameters"]["output_path"] = str(
            dest / "vtk_output" if "vtk" in proc["python_module"] else dest)


def main():
    for required in (CONFIG, MATERIALS, MAIN):
        if not required.exists():
            sys.exit(f"run this from bayesian_inference/ -- {required} not found")

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    backups = {}
    for original in (CONFIG, MATERIALS):
        backup = original.with_suffix(original.suffix + ".sweep_backup")
        shutil.copy(original, backup)
        backups[original] = backup

    base_config = json.loads(CONFIG.read_text())
    base_materials = json.loads(MATERIALS.read_text())
    sigma = base_config["likelihood"]["noise_model"]["sigma"]
    prior = base_config["parameters"][0]["prior"]

    if "reference_value" in base_config["parameters"][0]:
        sys.exit("BayesianParameters.json still carries reference_value -- remove it, "
                 "or it will override the materials file and the sweep will do nothing")

    results = []
    try:
        for tag, e_ref in E_REFS:
            dest = ARCHIVE / f"output_{tag}"
            dest.mkdir(parents=True, exist_ok=True)

            materials = json.loads(json.dumps(base_materials))
            set_modulus(materials, e_ref)
            MATERIALS.write_text(json.dumps(materials, indent=4))

            cfg = json.loads(json.dumps(base_config))
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
                "e_ref_source": "StructuralMaterialsBayes.json",
                "sigma_assumed": sigma,
                "prior_type": prior["type"],
                "prior_parameters": prior["parameters"],
                "returncode": code,
                "wall_time_s": round(elapsed, 1),
                "started": started.isoformat(timespec="seconds"),
            }, indent=2))
            shutil.copy(CONFIG, dest / "BayesianParameters.json")
            shutil.copy(MATERIALS, dest / "StructuralMaterialsBayes.json")

            results.append((tag, code, elapsed))
            print(f"  {'done' if code == 0 else 'FAILED (exit %d)' % code} in {elapsed:.0f} s")

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        for original, backup in backups.items():
            shutil.move(str(backup), str(original))
            print(f"restored {original}")

    ok = sum(1 for _, code, _ in results if code == 0)
    print(f"\n{ok}/{len(E_REFS)} runs completed")
    for tag, code, elapsed in results:
        print(f"  {tag:<12} {'ok' if code == 0 else 'FAILED':<7} {elapsed:>7.0f} s")
    print(f"archive: {ARCHIVE}")


if __name__ == "__main__":
    main()
