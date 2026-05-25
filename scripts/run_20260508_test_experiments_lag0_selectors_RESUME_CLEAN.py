"""
Run the updated no-publication-lag nowcasting experiment grid for 2026-05-08.
Includes argparse-safe handling for negative vintage labels and skips already
completed experiment labels by default.

It runs:
1) ElasticNet:
   - datasets: 5 thesis datasets, including GT v1 and GT lt variants
   - fills: locf, vertical_realignment, autoarima, tactis2
   - selectors: pca, corr_top_n, lasso
   - fixed alpha values: 1e-8, 1e-5, 1e-4, 1e-3
   - l1_ratio = 0.25
   - publication lags = 0

2) DFM:
   - DFM common-frequency
   - DFM_MF mixed-frequency

3) MIDAS:
   - MIDAS ridge
   - MIDAS linear
   - MIDASML elasticnet

Results:
- run_local_vintage_nowcasts.py still writes forecast CSVs to data/forecasts.
- This wrapper copies newly written forecast/metric CSVs and plots into:
      data/forecasts/2026.0508_test_lag0_selectors/
- Checkpoints are written directly into:
      data/forecasts/2026.0508_test_lag0_selectors/checkpoints/

Run from the project root:
    py scripts/run_20260508_test_experiments_lag0_selectors.py

Useful options:
    py scripts/run_20260508_test_experiments_lag0_selectors.py --dry-run
    py scripts/run_20260508_test_experiments_lag0_selectors.py --only elasticnet
    py scripts/run_20260508_test_experiments_lag0_selectors.py --only dfm
    py scripts/run_20260508_test_experiments_lag0_selectors.py --only midas
    py scripts/run_20260508_test_experiments_lag0_selectors.py --smoke-test
    py scripts/run_20260508_test_experiments_lag0_selectors.py --force-rerun
    py scripts/run_20260508_test_experiments_lag0_selectors.py --no-copy-outputs
    py scripts/run_20260508_test_experiments_lag0_selectors.py --skip-tactis2
    py scripts/run_20260508_test_experiments_lag0_selectors.py --reverse-order
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------

DATASETS = [
    "final_thesis_baseline_common",
    "final_thesis_common_plus_gt_v1",
    "final_thesis_gt_only_v1",
    "final_thesis_common_plus_gt_lt",
    "final_thesis_gt_only_lt",
]

COMMON_ARGS = {
    "--start-date": "2019-03-31",
    "--end-date": "2026-06-30",
    "--train-start": "2000-01-01",
    "--rolling-window-quarters": "76",
    "--vintages": "-2,-1,0,1,2",
    "--monthly-feature-release-lag-months": "0",
    "--quarterly-feature-release-lag-months": "0",
    "--gt-release-lag-months": "0",
    "--seed": "2234",
    "--checkpoint-every": "1",
}

ELASTICNET_FILLS = [
    "locf",
    "vertical_realignment",
    "autoarima",
    "tactis2",
]

ELASTICNET_SELECTORS = [
    "pca",
    "corr_top_n",
    "lasso",
]

PCA_COMPONENTS = "50"
TOP_N = "50"
SELECTOR_ALPHA = "0.1"

# Keep 1e-5 because it is the main Hopp-style low-penalty specification.
# 1e-8 is diagnostic / almost unregularized.
# 1e-4 and 1e-3 test stronger regularization.
ELASTICNET_ALPHAS = [
    "1e-8",
    "1e-5",
    "1e-4",
    "1e-3",
]

ELASTICNET_L1_RATIO = "0.25"
ELASTICNET_MAX_ITER = "20000"

# ---------------------------------------------------------------------
# DFM / DFM_MF tuning grid
# ---------------------------------------------------------------------
DFM_K_FACTORS_GRID = [1, 2, 3]
DFM_FACTOR_ORDER_GRID = [1, 2, 3]

DFM_SELECTORS = ["pca", "corr_top_n"]
DFM_PCA_COMPONENTS_GRID = [10, 20]
DFM_TOP_N_GRID = [20, 50]

DFM_MF_SELECTORS = ["corr_top_n"]
DFM_MF_TOP_N_GRID = [20, 50]

DFM_MAXITER = "50"
DFM_TOLERANCE = "1e-4"

# ---------------------------------------------------------------------
# MIDAS / MIDASML tuning grid
# ---------------------------------------------------------------------
# For MIDAS, this is not internal CV inside the model.
# It is an outer pseudo-real-time grid/backtest over lags and regression type.
MIDAS_N_LAGS_GRID = [3, 4, 6, 9, 12]
MIDAS_REGRESSION_MODELS = ["ridge", "linear"]
MIDAS_INTERNAL_FILL = "ffill_then_zero"

# MIDASML has real internal CV through --midasml-cv.
MIDASML_N_LAGS_GRID = [3, 4, 6, 9, 12]
MIDASML_CV = "5"
MIDASML_L1_RATIO_GRID = [0.1, 0.5, 0.9]
MIDASML_MAX_ITER = "5000"


@dataclass
class Experiment:
    label: str
    args: list[str]


def sanitize_part(s: str) -> str:
    return (
        str(s)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace("=", "")
        .replace(",", "_")
        .replace(":", "-")
        .replace(".", "p")
        .replace("+", "plus")
    )


def add_kv_args(cmd: list[str], kv: dict[str, str | int | float | None]) -> None:
    for key, value in kv.items():
        if value is None:
            continue

        value_str = str(value)

        # Important for argparse:
        # values like "-2,-1,0,1,2" can be interpreted as another option
        # if passed as ["--vintages", "-2,-1,0,1,2"].
        # Passing "--vintages=-2,-1,0,1,2" avoids returncode=2.
        if value_str.startswith("-"):
            cmd.append(f"{key}={value_str}")
        else:
            cmd.extend([key, value_str])


def base_command(project_root: Path, run_dir: Path, label: str) -> list[str]:
    checkpoint_dir = run_dir / "checkpoints" / label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_local_vintage_nowcasts.py"),
    ]
    add_kv_args(cmd, COMMON_ARGS)
    cmd.extend([
        "--datasets", ",".join(DATASETS),
        "--checkpoint-dir", str(checkpoint_dir),
        "--resume",
    ])
    return cmd


def build_elasticnet_experiments(project_root: Path, run_dir: Path, include_tactis2: bool = True) -> list[Experiment]:
    experiments: list[Experiment] = []

    fills = list(ELASTICNET_FILLS)
    if not include_tactis2:
        fills = [f for f in fills if f != "tactis2"]

    for fill in fills:
        for selector in ELASTICNET_SELECTORS:
            for alpha in ELASTICNET_ALPHAS:
                label = sanitize_part(
                    f"elasticnet_{fill}_{selector}_a{alpha}_l1{ELASTICNET_L1_RATIO}_lag0"
                )
                cmd = base_command(project_root, run_dir, label)
                cmd.extend([
                    "--model", "ElasticNet",
                    "--ragged-fill-method", fill,
                    "--selector", selector,
                    "--elasticnet-no-cv",
                    "--elasticnet-alpha", alpha,
                    "--elasticnet-l1-ratio", ELASTICNET_L1_RATIO,
                    "--elasticnet-max-iter", ELASTICNET_MAX_ITER,
                ])

                if selector == "pca":
                    cmd.extend(["--pca-components", PCA_COMPONENTS])
                elif selector == "corr_top_n":
                    cmd.extend(["--top-n", TOP_N])
                elif selector == "lasso":
                    cmd.extend(["--selector-alpha", SELECTOR_ALPHA])

                # AutoARIMA is expensive; use the fast configuration and a separate cache for this run.
                if fill == "autoarima":
                    arima_cache = run_dir / "cache" / "arima_vintage_cache.sqlite"
                    arima_cache.parent.mkdir(parents=True, exist_ok=True)
                    cmd.extend([
                        "--arima-fast",
                        "--arima-cache-path", str(arima_cache),
                    ])

                # TACTiS2 is very expensive. These are intentionally lightweight defaults.
                # Remove or modify these flags if you want the heavy author-style configuration.
                if fill == "tactis2":
                    tactis_cache = run_dir / "cache" / "tactis2"
                    tactis_cache.mkdir(parents=True, exist_ok=True)
                    cmd.extend([
                        "--tactis2-author-config",
                        "--tactis2-epochs-phase-1", "20",
                        "--tactis2-epochs-phase-2", "20",
                        "--tactis2-batch-size", "128",
                        "--tactis2-num-batches-per-epoch", "128",
                        "--tactis2-learning-rate", "1e-3",
                        "--tactis2-weight-decay", "1e-4",
                        "--tactis2-maximum-learning-rate", "1e-3",
                        "--tactis2-clip-gradient", "1000",
                        "--tactis2-bagging-size", "20",
                        "--tactis2-skip-copula", "false",
                        "--tactis2-context-length", "120",
                        "--tactis2-num-samples", "100",
                        "--tactis2-device", "auto",
                    ])

                experiments.append(Experiment(label=label, args=cmd))

    return experiments

def build_dfm_experiments(project_root: Path, run_dir: Path) -> list[Experiment]:
    experiments: list[Experiment] = []

    # ------------------------------------------------------------
    # Common-frequency DFM
    # Grid:
    #   k_factors: 1, 2, 3
    #   factor_order: 1, 2, 3
    #   selector: pca, corr_top_n
    # ------------------------------------------------------------
    for k in DFM_K_FACTORS_GRID:
        for order in DFM_FACTOR_ORDER_GRID:
            for selector in DFM_SELECTORS:

                if selector == "pca":
                    for pca_components in DFM_PCA_COMPONENTS_GRID:
                        label = sanitize_part(
                            f"dfm_aggmean_k{k}_p{order}_selpca_pca{pca_components}"
                        )
                        cmd = base_command(project_root, run_dir, label)
                        cmd.extend([
                            "--model", "DFM",
                            "--ragged-fill-method", "none",
                            "--quarterly-aggregation", "mean",
                            "--dfm-k-factors", str(k),
                            "--dfm-factor-order", str(order),
                            "--dfm-selector", "pca",
                            "--dfm-pca-components", str(pca_components),
                            "--dfm-maxiter", DFM_MAXITER,
                            "--dfm-tolerance", DFM_TOLERANCE,
                        ])
                        experiments.append(Experiment(label=label, args=cmd))

                elif selector == "corr_top_n":
                    for top_n in DFM_TOP_N_GRID:
                        label = sanitize_part(
                            f"dfm_aggmean_k{k}_p{order}_selcorr_top_n_top{top_n}"
                        )
                        cmd = base_command(project_root, run_dir, label)
                        cmd.extend([
                            "--model", "DFM",
                            "--ragged-fill-method", "none",
                            "--quarterly-aggregation", "mean",
                            "--dfm-k-factors", str(k),
                            "--dfm-factor-order", str(order),
                            "--dfm-selector", "corr_top_n",
                            "--top-n", str(top_n),
                            "--dfm-maxiter", DFM_MAXITER,
                            "--dfm-tolerance", DFM_TOLERANCE,
                        ])
                        experiments.append(Experiment(label=label, args=cmd))

    # ------------------------------------------------------------
    # Mixed-frequency DFM_MF
    # Same k/order grid as DFM.
    #
    # Important:
    # run_local_vintage_nowcasts.py currently allows DFM_MF selector
    # only in ["none", "corr_top_n"], so we use corr_top_n here.
    # PCA for DFM_MF would require extending run_local_vintage_nowcasts.py.
    # ------------------------------------------------------------
    for k in DFM_K_FACTORS_GRID:
        for order in DFM_FACTOR_ORDER_GRID:
            for top_n in DFM_MF_TOP_N_GRID:
                label = sanitize_part(
                    f"dfm_mf_k{k}_p{order}_mfselcorr_top_n_top{top_n}"
                )
                cmd = base_command(project_root, run_dir, label)
                cmd.extend([
                    "--model", "DFM_MF",
                    "--ragged-fill-method", "none",
                    "--dfm-k-factors", str(k),
                    "--dfm-factor-order", str(order),
                    "--dfm-mf-selector", "corr_top_n",
                    "--dfm-mf-top-n", str(top_n),
                    "--dfm-maxiter", DFM_MAXITER,
                    "--dfm-tolerance", DFM_TOLERANCE,
                ])
                experiments.append(Experiment(label=label, args=cmd))

    return experiments


def build_midas_experiments(project_root: Path, run_dir: Path) -> list[Experiment]:
    experiments: list[Experiment] = []

    # ------------------------------------------------------------
    #MIDAS
    # This is an outer pseudo-real-time CV/grid over lags and backend.
    # No internal --midas-cv exists in run_local_vintage_nowcasts.py.
    # ------------------------------------------------------------
    for n_lags in MIDAS_N_LAGS_GRID:
        for reg in MIDAS_REGRESSION_MODELS:
            label = sanitize_part(
                f"midas_{reg}_lags{n_lags}_fill{MIDAS_INTERNAL_FILL}"
            )
            cmd = base_command(project_root, run_dir, label)
            cmd.extend([
                "--model", "MIDAS",
                "--ragged-fill-method", "none",
                "--midas-n-lags", str(n_lags),
                "--midas-regression-model", reg,
                "--midas-internal-fill-strategy", MIDAS_INTERNAL_FILL,
            ])
            experiments.append(Experiment(label=label, args=cmd))

    # ------------------------------------------------------------
    # MIDASML
    # Internal CV = 5 through --midasml-cv.
    # Grid over lags and ElasticNet l1_ratio.
    # ------------------------------------------------------------
    for n_lags in MIDASML_N_LAGS_GRID:
        for l1_ratio in MIDASML_L1_RATIO_GRID:
            label = sanitize_part(
                f"midasml_elasticnet_lags{n_lags}_cv{MIDASML_CV}_l1{l1_ratio}_fill{MIDAS_INTERNAL_FILL}"
            )
            cmd = base_command(project_root, run_dir, label)
            cmd.extend([
                "--model", "MIDASML",
                "--ragged-fill-method", "none",
                "--midas-n-lags", str(n_lags),
                "--midas-internal-fill-strategy", MIDAS_INTERNAL_FILL,
                "--midasml-regression-model", "elasticnet",
                "--midasml-cv", MIDASML_CV,
                "--midasml-l1-ratio", str(l1_ratio),
                "--midasml-max-iter", MIDASML_MAX_ITER,
            ])
            experiments.append(Experiment(label=label, args=cmd))

    return experiments


def iter_candidate_outputs(forecasts_dir: Path) -> Iterable[Path]:
    # Forecast and metric CSVs written by run_local_vintage_nowcasts.py
    for p in forecasts_dir.glob("*.csv"):
        if p.is_file():
            yield p

    # Plots, if the underlying script creates or updates them
    plots_dir = forecasts_dir / "plots"
    if plots_dir.exists():
        for p in plots_dir.glob("*.png"):
            if p.is_file():
                yield p


def copy_recent_outputs(
    forecasts_dir: Path,
    run_dir: Path,
    since_ts: float,
    label: str,
) -> list[str]:
    copied: list[str] = []

    result_dir = run_dir / "results"
    plot_dir = run_dir / "plots"
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for src in iter_candidate_outputs(forecasts_dir):
        try:
            if src.stat().st_mtime < since_ts - 1.0:
                continue
        except FileNotFoundError:
            continue

        if src.suffix.lower() == ".png":
            dst = plot_dir / src.name
        else:
            dst = result_dir / src.name

        # Avoid accidentally copying from inside run_dir to itself
        try:
            if src.resolve() == dst.resolve():
                continue
        except FileNotFoundError:
            pass

        # Do not rewrite already copied files. This keeps repeated wrapper runs
        # from touching old results/plots or creating duplicate-looking outputs.
        if dst.exists():
            copied.append(str(dst.relative_to(run_dir)) + " [already_exists]")
            continue

        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(run_dir)))

    # Per-experiment output list
    out_list = run_dir / "logs" / f"{label}_copied_outputs.txt"
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("\n".join(copied), encoding="utf-8")

    return copied


def write_manifest(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "experiment_manifest.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = run_dir / "experiment_manifest.csv"
    fieldnames = [
        "label",
        "status",
        "returncode",
        "runtime_sec",
        "log_file",
        "n_copied_outputs",
        "command",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})



def load_completed_labels(run_dir: Path) -> set[str]:
    """
    Labels that were completed successfully in a previous run.
    Used to avoid re-running completed experiments and regenerating duplicate plots/results.
    """
    completed: set[str] = set()

    manifest_csv = run_dir / "experiment_manifest.csv"
    if manifest_csv.exists():
        try:
            with manifest_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") == "ok" and row.get("label"):
                        completed.add(row["label"])
        except Exception as e:
            print(f"WARNING: failed to read previous manifest {manifest_csv}: {e}")

    done_dir = run_dir / "done"
    if done_dir.exists():
        for p in done_dir.glob("*.done"):
            completed.add(p.stem)

    return completed


def mark_done(run_dir: Path, label: str, row: dict) -> None:
    done_dir = run_dir / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    done_path = done_dir / f"{label}.done"
    done_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiment(
    exp: Experiment,
    project_root: Path,
    forecasts_dir: Path,
    run_dir: Path,
    dry_run: bool,
    smoke_test: bool,
    force_rerun: bool,
    copy_outputs: bool,
    completed_labels: set[str],
) -> dict:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{exp.label}.log"

    if (not force_rerun) and exp.label in completed_labels:
        print("\n" + "=" * 90)
        print(f"Skipping already completed: {exp.label}")
        print("Use --force-rerun if you want to run it again and regenerate outputs.")
        print("=" * 90)
        return {
            "label": exp.label,
            "status": "skipped_existing_ok",
            "returncode": 0,
            "runtime_sec": 0,
            "log_file": str(log_path),
            "n_copied_outputs": 0,
            "command": "",
        }

    cmd = list(exp.args)
    if smoke_test:
        cmd.append("--smoke-test")

    cmd_str = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    start_ts = time.time()

    print("\n" + "=" * 90)
    print(f"Running: {exp.label}")
    print(cmd_str)
    print("=" * 90)

    if dry_run:
        log_path.write_text(cmd_str + "\n", encoding="utf-8")
        return {
            "label": exp.label,
            "status": "dry_run",
            "returncode": "",
            "runtime_sec": 0,
            "log_file": str(log_path),
            "n_copied_outputs": 0,
            "command": cmd_str,
        }

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(cmd_str + "\n\n")
        log_file.flush()

        proc = subprocess.run(
            cmd,
            cwd=project_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if copy_outputs:
        copied = copy_recent_outputs(
            forecasts_dir=forecasts_dir,
            run_dir=run_dir,
            since_ts=start_ts,
            label=exp.label,
        )
    else:
        copied = []
        out_list = run_dir / "logs" / f"{exp.label}_copied_outputs.txt"
        out_list.parent.mkdir(parents=True, exist_ok=True)
        out_list.write_text("copy_outputs disabled\n", encoding="utf-8")

    runtime = time.time() - start_ts
    status = "ok" if proc.returncode == 0 else "failed"

    print(f"Finished {exp.label}: {status}, returncode={proc.returncode}, runtime={runtime:.1f}s")
    print(f"Copied outputs: {len(copied)}")
    print(f"Log: {log_path}")

    row = {
        "label": exp.label,
        "status": status,
        "returncode": proc.returncode,
        "runtime_sec": round(runtime, 2),
        "log_file": str(log_path),
        "n_copied_outputs": len(copied),
        "command": cmd_str,
    }

    if status == "ok":
        mark_done(run_dir, exp.label, row)
        completed_labels.add(exp.label)

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 2026-05-08 nowcasting experiment grid.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="data/forecasts/2026.0508_test_lag0_selectors",
        help="Directory where logs, checkpoints, copied results, and manifest will be stored.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "elasticnet", "dfm", "midas"],
        default="all",
        help="Run only one experiment family.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print/write commands but do not execute them.")
    parser.add_argument("--smoke-test", action="store_true", help="Pass --smoke-test to each experiment.")
    parser.add_argument("--skip-tactis2", action="store_true", help="Skip ElasticNet + tactis2 fill. TACTiS2 is included by default and can be very slow.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if one experiment fails.")
    parser.add_argument("--force-rerun", action="store_true", help="Run even experiment labels that already completed successfully.")
    parser.add_argument("--no-copy-outputs", action="store_true", help="Do not copy freshly generated CSV/PNG outputs into the run_dir results/plots folders.")
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="Run experiments in reverse order. Useful when another machine is running the same grid from the beginning.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    forecasts_dir = project_root / "data" / "forecasts"
    run_dir = project_root / args.run_dir

    forecasts_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "cache").mkdir(parents=True, exist_ok=True)
    (run_dir / "done").mkdir(parents=True, exist_ok=True)

    completed_labels = load_completed_labels(run_dir)
    if completed_labels and not args.force_rerun:
        print(f"Found {len(completed_labels)} previously completed experiment labels; they will be skipped.")

    experiments: list[Experiment] = []
    if args.only in ["all", "elasticnet"]:
        experiments.extend(build_elasticnet_experiments(project_root, run_dir, include_tactis2=not args.skip_tactis2))
    if args.only in ["all", "dfm"]:
        experiments.extend(build_dfm_experiments(project_root, run_dir))
    if args.only in ["all", "midas"]:
        experiments.extend(build_midas_experiments(project_root, run_dir))

    if args.reverse_order:
        experiments = list(reversed(experiments))
        print("Running experiments in reverse order.")

    print(f"Project root: {project_root}")
    print(f"Run dir:      {run_dir}")
    print(f"Experiments:  {len(experiments)}")
    print(f"Datasets:     {', '.join(DATASETS)}")

    manifest_rows: list[dict] = []

    for exp in experiments:
        row = run_experiment(
            exp=exp,
            project_root=project_root,
            forecasts_dir=forecasts_dir,
            run_dir=run_dir,
            dry_run=args.dry_run,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
            copy_outputs=not args.no_copy_outputs,
            completed_labels=completed_labels,
        )
        manifest_rows.append(row)
        write_manifest(run_dir, manifest_rows)

        if row["status"] == "failed" and not args.continue_on_error:
            print("Stopping because an experiment failed. Use --continue-on-error to keep going.")
            return int(row["returncode"] or 1)

    write_manifest(run_dir, manifest_rows)

    print("\nDone.")
    print(f"Manifest: {run_dir / 'experiment_manifest.csv'}")
    print(f"Results:  {run_dir / 'results'}")
    print(f"Logs:     {run_dir / 'logs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
