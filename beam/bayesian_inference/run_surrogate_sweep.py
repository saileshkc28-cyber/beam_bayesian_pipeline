r"""Surrogate + hierarchical sensor-count sweep. Edits nothing in the repository.

For k in 1, 4, 10 (the first k sensors of sensor_data_10s.json, tip first, then
inward) a response surrogate u(E) is built through Kratos, and the population
parameters (mu_E, sd_E) are inferred from the 10-sensor Phase 1 specimens, read
through the same first k sensors. Every input a run needs is written into its own
output folder, and the run writes everything else there too:

    <archive>/N04/sensor_data.json               first 4 sensors of sensor_data_10s.json
    <archive>/N04/BayesianParameters.json        forward model for the surrogate: primal
                                                 file absolute, the sensor file above
    <archive>/N04/SurrogateParameters.json       base_config above, outputs in surrogate/,
                                                 accuracy-check sigma 3.788e-08, GP seed
    <archive>/N04/HierarchicalParameters.json    specimens, the surrogate below, sigma
                                                 3.788e-08, output in hierarchical/
    <archive>/N04/surrogate/                     training/validation solves, model, report,
                                                 surrogate_fit.png (tip sensor)
    <archive>/N04/hierarchical/                  summary.json, posterior.npz
    <archive>/N04/surrogate.log                  BuildSurrogate.py console output
    <archive>/N04/plot_surrogate.log             plot_surrogate.py console output
    <archive>/N04/hierarchical.log               MainHierarchical.py console output
    <archive>/N04/run_info.json                  k, sensors, sigma, both truths, specimen
                                                 count, surrogate facts, per-stage status;
                                                 merged, so stages run separately keep both

Stages:
    surrogate      BuildSurrogate.py, 25 training + 15 validation Kratos solves, run from
                   bayesian_inference/ (the primal file uses relative paths). A case whose
                   surrogate/ already holds a training CSV is skipped unless --resume.
                   The saved sensor_names must equal the first k sensors, in order.
    hierarchical   MainHierarchical.py, no Kratos solves, run from hierarchical/. Needs a
                   surrogate whose sensor_names were verified as above.

The repo's sensor_data_10s.json, specimen CSV, truth file and the three base configs
are only read. Nothing is ever deleted.

Noise model, option A: one instrument sigma = 3.788e-08 for every sensor, as in the
u_mean and weighted sweeps. The specimens are the 1000 valid ones of
phase1_distribution_runs_10s, each measured with that sigma on every sensor.

Truth, both recorded in run_info.json:
    prescribed   alpha ~ N(1.0, 0.1), the Phase 1 generator's distribution
    realised     population_truth of phase1_clean_summary.json (the drawn specimens)

Run from bayesian_inference/:
    python run_surrogate_sweep.py                          # N01, N04, N10, both stages
    python run_surrogate_sweep.py N04                      # re-run single cases
    python run_surrogate_sweep.py --stage surrogate        # Kratos part only
    python run_surrogate_sweep.py --stage hierarchical     # reuse the surrogates on file
    python run_surrogate_sweep.py --resume                 # continue a surrogate build
    python run_surrogate_sweep.py --prepare-only           # write inputs, run nothing
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(r"D:\KratosProjects\MCMC\Analysis\sensors_surrogate")
SENSORS_10 = HERE.parent / "sensor_placement" / "sensor_data_10s.json"
SPECIMENS = (HERE.parent / "damaged_system" / "phase1_distribution_runs_10s"
             / "phase1_samples_clean.csv")
TRUTH_FILE = SPECIMENS.parent / "phase1_clean_summary.json"

BASE_BAYES = HERE / "BayesianParameters.json"
BASE_SURROGATE = HERE / "SurrogateParameters.json"
BASE_HIERARCHICAL = HERE / "hierarchical" / "HierarchicalParameters.json"
BUILD = HERE / "BuildSurrogate.py"
PLOT = HERE / "plot_surrogate.py"
MAIN_HIERARCHICAL = HERE / "hierarchical" / "MainHierarchical.py"

SIGMA = 3.788e-08
SENSOR_COUNTS = [1, 4, 10]
GP_RANDOM_STATE = 20260911

# The Phase 1 generator's prescribed distribution (MainKratos_phase1.py:
# E_MEAN = 206.9e9, E_STD = 20.69e9), as alpha = E / E_ref.
ALPHA_MEAN = 1.0
ALPHA_SD = 0.1


def is_fresh(path, started):
    """Exists and was written by this run, not left over from an earlier one."""
    return path.exists() and path.stat().st_mtime >= started.timestamp() - 1.0


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_inputs(dest, k, tag, sensors, base):
    dest.mkdir(parents=True, exist_ok=True)
    sdir = dest / "surrogate"

    sensor_file = dest / "sensor_data.json"
    sensor_file.write_text(json.dumps({"list_of_sensors": sensors[:k]}, indent=4))

    bayes = json.loads(json.dumps(base["bayes"]))
    bayes["problem_data"]["problem_name"] = f"bayesian_inference_beam_surrogate_{tag}"
    fm = bayes["forward_model"]
    fm["primal_parameters_file"] = str(HERE / fm["primal_parameters_file"])
    fm["sensor_data_file"] = str(sensor_file)
    bayes["likelihood"]["noise_model"]["sigma"] = SIGMA
    for block in ("three_point_inference", "batch_inference"):
        if block in bayes:
            bayes[block]["enabled"] = False
    bayes_file = dest / "BayesianParameters.json"
    bayes_file.write_text(json.dumps(bayes, indent=4))

    surr = json.loads(json.dumps(base["surrogate"]))
    surr["base_config"] = str(bayes_file)
    out = surr["output"]
    out["dir"] = str(sdir)
    for key in ("training_csv", "validation_csv", "model_file", "report_file"):
        out[key] = str(sdir / Path(out[key]).name)
    surr["accuracy_gate"]["sigma_noise_m"] = [SIGMA]
    surr["gp"]["random_state"] = GP_RANDOM_STATE
    surr_file = dest / "SurrogateParameters.json"
    surr_file.write_text(json.dumps(surr, indent=4))

    hier = json.loads(json.dumps(base["hierarchical"]))
    hier["problem_data"]["problem_name"] = f"hierarchical_population_beam_{tag}"
    lk = hier["likelihood"]
    lk["observations_file"] = str(SPECIMENS)
    lk["surrogate_file"] = out["model_file"]
    lk["noise_model"]["sigma"] = SIGMA
    hier["output"]["output_path"] = str(dest / "hierarchical")
    hier_file = dest / "HierarchicalParameters.json"
    hier_file.write_text(json.dumps(hier, indent=4))

    return {"bayesian": str(bayes_file), "surrogate": str(surr_file),
            "hierarchical": str(hier_file), "sensor_data": str(sensor_file)}


def run_child(cmd, cwd, log_file):
    """One child process, output on screen and in log_file as it happens."""
    # unbuffered child so its prints reach the screen and the log as they happen
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log_file, "w") as log:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def verify_surrogate(dest, names):
    """(verified, reason, identity): verified only if the saved surrogate was built for
    exactly these sensors, in this order."""
    meta = dest / "surrogate" / "response_surrogate.joblib.json"
    if not (dest / "surrogate" / "response_surrogate.joblib").exists() or not meta.exists():
        return False, "no surrogate on file", None
    identity = read_json(meta)
    saved = identity.get("sensor_names")
    if saved is None:
        return False, "the surrogate records no sensor names (built before they were saved)", \
            identity
    if saved != names:
        return False, f"the surrogate's sensors {saved} are not the first {len(names)} " \
                      f"of {SENSORS_10.name} {names}", identity
    return True, "sensor names verified", identity


def surrogate_facts(dest, identity, verified, started):
    report_file = dest / "surrogate" / "validation_report.json"
    report = read_json(report_file) if report_file.exists() else {}
    n_tr, n_va = report.get("n_training"), report.get("n_validation")
    return {
        "n_train": n_tr,
        "n_validation": n_va,
        # one Kratos solve per training or validation point behind this surrogate
        "kratos_solves": None if n_tr is None or n_va is None else n_tr + n_va,
        # only known when this run wrote the report
        "kratos_solves_this_run": (report.get("solves_used_this_session")
                                   if is_fresh(report_file, started) else None),
        "random_state": None if identity is None else identity.get("random_state"),
        "sensor_names_verified": verified,
        "gp_passes_gate": report.get("gp_passes_gate"),
        "max_error_over_sigma": report.get("gp", {}).get("max_error_over_sigma"),
    }


def stage_surrogate(tag, dest, names, configs, resume):
    started = datetime.now()
    stages, training_csv = {}, dest / "surrogate" / "training_responses.csv"

    if training_csv.exists() and not resume:
        print(f"\n  {tag}: {training_csv} already exists -- surrogate stage skipped "
              f"(pass --resume to continue that build; nothing is deleted)", flush=True)
        stages["surrogate"] = {"state": "skipped: training CSV on file, no --resume",
                               "returncode": None, "wall_time_s": 0.0,
                               "started": started.isoformat(timespec="seconds")}
    else:
        code = run_child([sys.executable, BUILD, "--config", configs["surrogate"]], HERE,
                         dest / "surrogate.log")
        model_meta = dest / "surrogate" / "response_surrogate.joblib.json"
        if code != 0:
            state = f"FAILED (exit {code})"
        elif not is_fresh(model_meta, started):
            state = "FAILED (exit 0 but no fresh response_surrogate.joblib.json)"
        else:
            state = "ok"
        elapsed = (datetime.now() - started).total_seconds()
        stages["surrogate"] = {"state": state, "returncode": code,
                               "wall_time_s": round(elapsed, 1),
                               "started": started.isoformat(timespec="seconds"),
                               "log": str(dest / "surrogate.log")}

        if state == "ok":
            p_started = datetime.now()
            p_code = run_child([sys.executable, PLOT, configs["surrogate"]], HERE,
                               dest / "plot_surrogate.log")
            png = dest / "surrogate" / "surrogate_fit.png"
            stages["plot_surrogate"] = {
                "state": "ok" if p_code == 0 and is_fresh(png, p_started)
                else f"FAILED (exit {p_code})" if p_code else "FAILED (no fresh png)",
                "returncode": p_code,
                "started": p_started.isoformat(timespec="seconds"),
                "log": str(dest / "plot_surrogate.log")}

    verified, why, identity = verify_surrogate(dest, names)
    if stages["surrogate"]["state"] == "ok" and not verified:
        stages["surrogate"]["state"] = f"FAILED ({why})"
    print(f"\n  {tag}: surrogate {stages['surrogate']['state']}; {why}", flush=True)
    return stages, surrogate_facts(dest, identity, verified, started)


def stage_hierarchical(tag, dest, names, configs):
    started = datetime.now()
    verified, why, _ = verify_surrogate(dest, names)
    if not verified:
        print(f"\n  {tag}: hierarchical stage skipped -- {why}", flush=True)
        return {"state": f"skipped: {why}", "returncode": None, "wall_time_s": 0.0,
                "started": started.isoformat(timespec="seconds")}

    code = run_child([sys.executable, MAIN_HIERARCHICAL, configs["hierarchical"]],
                     HERE / "hierarchical", dest / "hierarchical.log")
    if code != 0:
        state = f"FAILED (exit {code})"
    elif not is_fresh(dest / "hierarchical" / "summary.json", started):
        state = "FAILED (exit 0 but no fresh hierarchical/summary.json)"
    else:
        state = "ok"
    print(f"\n  {tag}: hierarchical {state}", flush=True)
    return {"state": state, "returncode": code,
            "wall_time_s": round((datetime.now() - started).total_seconds(), 1),
            "started": started.isoformat(timespec="seconds"),
            "log": str(dest / "hierarchical.log")}


def main():
    ap = argparse.ArgumentParser(description="surrogate + hierarchical sensor-count sweep")
    ap.add_argument("tags", nargs="*", help="cases to run, e.g. N04 (default: all)")
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--prepare-only", action="store_true",
                    help="write the per-case inputs and run_info.json, run nothing")
    ap.add_argument("--stage", choices=["all", "surrogate", "hierarchical"], default="all")
    ap.add_argument("--resume", action="store_true",
                    help="continue a surrogate build whose training CSV is on file")
    args = ap.parse_args()

    cases = {f"N{k:02d}": k for k in SENSOR_COUNTS}
    unknown = set(args.tags) - set(cases)
    if unknown:
        sys.exit(f"unknown tag(s): {', '.join(sorted(unknown))}\n"
                 f"available: {', '.join(cases)}")
    wanted = args.tags or list(cases)

    sensors = read_json(SENSORS_10)["list_of_sensors"]
    if len(sensors) < max(SENSOR_COUNTS):
        sys.exit(f"{SENSORS_10} has {len(sensors)} sensors, need {max(SENSOR_COUNTS)}")
    names_10 = [s["name"] for s in sensors]
    base = {"bayes": read_json(BASE_BAYES), "surrogate": read_json(BASE_SURROGATE),
            "hierarchical": read_json(BASE_HIERARCHICAL)}
    truth = read_json(TRUTH_FILE)["population_truth"]

    e_ref = float(base["surrogate"]["domain"]["e_scale_Pa"])
    if abs(e_ref - float(truth["e_ref_Pa"])) > 1e-9 * e_ref:
        sys.exit(f"E_ref mismatch: {BASE_SURROGATE.name} domain.e_scale_Pa = {e_ref:.6e} Pa "
                 f"but {TRUTH_FILE.name} population_truth.e_ref_Pa = "
                 f"{float(truth['e_ref_Pa']):.6e} Pa")

    with open(SPECIMENS, newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing = [n for n in names_10 if f"u_hat_{n}" not in columns]
        if missing:
            sys.exit(f"{SPECIMENS} has no u_hat_ column for {', '.join(missing)}")
        if "valid" not in columns:
            sys.exit(f"{SPECIMENS} has no 'valid' column -- run clean_phase1.py first")
        n_valid = sum(1 for r in reader if r["valid"].strip() == "1")

    truth_prescribed = {
        "alpha_mean": ALPHA_MEAN, "alpha_sd": ALPHA_SD,
        "E_mean_GPa": ALPHA_MEAN * e_ref / 1e9, "E_sd_GPa": ALPHA_SD * e_ref / 1e9,
        "source": ("run_surrogate_sweep.py constants ALPHA_MEAN / ALPHA_SD = the Phase 1 "
                   "generator's prescribed distribution (MainKratos_phase1.py E_MEAN = "
                   "206.9e9, E_STD = 20.69e9); E_ref from SurrogateParameters.json "
                   "domain.e_scale_Pa"),
    }
    truth_realised = {key: truth[key] for key in
                      ("alpha_mean", "alpha_sd", "E_mean_GPa", "E_sd_GPa")}
    truth_realised["source"] = str(TRUTH_FILE)

    print(f"sensors      : {SENSORS_10}  (first k of {len(sensors)})")
    print(f"specimens    : {SPECIMENS}  ({n_valid} valid)")
    print(f"truth        : prescribed alpha ~ N({ALPHA_MEAN}, {ALPHA_SD})   realised "
          f"mean {truth['alpha_mean']:.6f} sd {truth['alpha_sd']:.6f}   "
          f"E_ref {e_ref / 1e9:g} GPa")
    print(f"sigma        : {SIGMA:.4e} for every sensor (option A)")
    print(f"stage        : {'prepare only' if args.prepare_only else args.stage}"
          f"{'   (resume)' if args.resume else ''}")
    print(f"archive      : {args.archive}")

    results = []
    for tag in wanted:
        k = cases[tag]
        dest = args.archive / tag
        names = names_10[:k]
        stations = [s["location"][0] for s in sensors[:k]]
        print(f"\n{'=' * 70}\n  {tag}: {k} sensor{'s' if k > 1 else ''}   "
              f"x = {', '.join(format(x, '.1f') for x in stations)}"
              f"   -> {dest}\n{'=' * 70}", flush=True)

        started = datetime.now()
        configs = write_inputs(dest, k, tag, sensors, base)
        info_file = dest / "run_info.json"
        info = read_json(info_file) if info_file.exists() else {}
        info.update({
            "label": tag,
            "n_sensors": k,
            "sensor_names": names,
            "stations": stations,
            "sigma_assumed": SIGMA,
            "sigma_mode": "option A: one instrument sigma for every sensor",
            "truth_prescribed": truth_prescribed,
            "truth_realised": truth_realised,
            "n_specimens_valid": n_valid,
            "specimens_file": str(SPECIMENS),
            "configs": configs,
        })
        stages = info.setdefault("stages", {})

        if not args.prepare_only:
            if args.stage in ("all", "surrogate"):
                surr_stages, facts = stage_surrogate(tag, dest, names, configs, args.resume)
                stages.update(surr_stages)
                info["surrogate"] = facts
            if args.stage in ("all", "hierarchical"):
                stages["hierarchical"] = stage_hierarchical(tag, dest, names, configs)
        if "surrogate" not in info:
            verified, _, identity = verify_surrogate(dest, names)
            info["surrogate"] = surrogate_facts(dest, identity, verified, started)

        elapsed = (datetime.now() - started).total_seconds()
        info["last_run"] = {"started": started.isoformat(timespec="seconds"),
                            "wall_time_s": round(elapsed, 1),
                            "stage": "prepare-only" if args.prepare_only else args.stage}
        info_file.write_text(json.dumps(info, indent=2))
        results.append((tag, stages, elapsed))

    print(f"\n{'prepared' if args.prepare_only else 'ran'} {len(results)} case(s)")
    for tag, stages, elapsed in results:
        state = ("prepared" if args.prepare_only else "   ".join(
            f"{name} {stages[name]['state']}" for name in ("surrogate", "hierarchical")
            if name in stages))
        print(f"  {tag:<6} {elapsed:>7.0f} s   {state}")
    print(f"archive: {args.archive}")


if __name__ == "__main__":
    main()
