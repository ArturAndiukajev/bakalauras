from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

PROJECT = Path(r"C:\Users\artur\bvp_prognozavimo_db")
RUN_DIR = PROJECT / r"data\forecasts\2026.0512_midas_dfm_tuning"

SEARCH_DIRS = [
    RUN_DIR / "results",
    RUN_DIR / "checkpoints",
]

OUT_DIR = RUN_DIR / "gt_rmse_reduction_tables_dfm_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_DATASET = "final_thesis_baseline_common"

GT_DATASETS = {
    "GT angl.": "final_thesis_common_plus_gt_v1",
    "GT liet.": "final_thesis_common_plus_gt_lt",
}

SEED = "2234"
UNSTABLE_RMSE_THRESHOLD = 1.0


# ============================================================
# HELPERS
# ============================================================

def rmse(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def mae(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(np.abs(x)))


def fmt4(x):
    if pd.isna(x):
        return "--"
    return f"{float(x):.4f}".replace(".", ",")


def fmt_pct(x):
    if pd.isna(x):
        return "--"
    return f"{float(x):.2f}".replace(".", ",")


def latex_escape(s):
    s = "" if pd.isna(s) else str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("_", r"\_")
         .replace("%", r"\%")
         .replace("&", r"\&")
         .replace("#", r"\#")
         .replace("$", r"\$")
         .replace("{", r"\{")
         .replace("}", r"\}")
    )


def detect_dataset(filename: str):
    for ds in [BASELINE_DATASET] + list(GT_DATASETS.values()):
        if ds in filename:
            return ds
    return None


def has_forecast_cols(path: Path):
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False

    return {"target_quarter", "vintage_label", "prediction", "actual"}.issubset(cols)


def clean_config_key(path: Path, dataset: str):
    name = path.stem

    for prefix in [
        "vintage_nowcasts_metrics_",
        "vintage_nowcasts_",
        "checkpoint_",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    name = name.replace(f"_{dataset}_s{SEED}", "")
    name = name.replace(f"_{dataset}", "")

    return name


def parse_model_from_config(config_key: str):
    s = config_key.lower().replace("-", "_")

    if "dfm_mf" in s:
        return "DFM-MF"

    if "dfm" in s:
        return "DFM"

    return "other"


def simplify_config_label(config_key: str):
    s = config_key
    s = s.replace("native_ragged_", "")
    s = s.replace("DFM_MF", "DFM-MF")
    return s


def collect_forecast_files():
    rows = []

    for root in SEARCH_DIRS:
        if not root.exists():
            continue

        for path in root.rglob("*.csv"):
            name = path.name

            if ".corrupt" in name.lower():
                continue
            if "summary" in name.lower() or "status" in name.lower() or "dm_" in name.lower():
                continue
            if "metrics" in name.lower():
                continue

            dataset = detect_dataset(name)
            if dataset is None:
                continue

            config_key = clean_config_key(path, dataset)
            model = parse_model_from_config(config_key)

            if model not in {"DFM", "DFM-MF"}:
                continue

            if not has_forecast_cols(path):
                continue

            rows.append({
                "path": path,
                "source": "results" if root.name == "results" else "checkpoints",
                "dataset": dataset,
                "config_key": config_key,
                "config_label": simplify_config_label(config_key),
                "model": model,
            })

    grouped = {}
    for r in rows:
        key = (r["dataset"], r["config_key"])
        grouped.setdefault(key, []).append(r)

    selected = []
    duplicate_rows = []

    for key, items in grouped.items():
        if len(items) > 1:
            duplicate_rows.extend(items)

        def score(r):
            source_score = 2 if r["source"] == "results" else 1
            mtime = r["path"].stat().st_mtime
            return (source_score, mtime)

        selected.append(sorted(items, key=score, reverse=True)[0])

    pd.DataFrame([
        {**r, "path": str(r["path"])}
        for r in duplicate_rows
    ]).to_csv(
        OUT_DIR / "duplicate_candidates_dfm_only.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return selected


def read_forecast(row):
    df = pd.read_csv(row["path"])
    df = df.dropna(subset=["prediction", "actual"]).copy()

    df["target_quarter"] = df["target_quarter"].astype(str)
    df["vintage_label"] = df["vintage_label"].astype(str).str.replace("+", "", regex=False)
    df["prediction"] = df["prediction"].astype(float)
    df["actual"] = df["actual"].astype(float)
    df["error"] = df["prediction"] - df["actual"]
    df["abs_error"] = df["error"].abs()

    df["dataset"] = row["dataset"]
    df["config_key"] = row["config_key"]
    df["config_label"] = row["config_label"]
    df["model"] = row["model"]
    df["source_file"] = str(row["path"])

    return df


# ============================================================
# LOAD
# ============================================================

files = collect_forecast_files()

if not files:
    raise RuntimeError("No DFM / DFM-MF forecast files found.")

print("Selected forecast files:", len(files))
print("Files by model:")
print(pd.Series([r["model"] for r in files]).value_counts().to_string())

all_forecasts = pd.concat([read_forecast(r) for r in files], ignore_index=True)

all_forecasts.to_csv(
    OUT_DIR / "forecast_rows_used_dfm_only.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nRows:", len(all_forecasts))
print("Models:")
print(all_forecasts["model"].value_counts().to_string())


# ============================================================
# CALCULATE RMSE REDUCTION
# ============================================================

comparison_rows = []
vintage_rows = []

configs = (
    all_forecasts[["model", "config_key", "config_label"]]
    .drop_duplicates()
    .sort_values(["model", "config_key"])
)

for _, cfg in configs.iterrows():
    config_key = cfg["config_key"]

    base = all_forecasts[
        (all_forecasts["config_key"] == config_key)
        & (all_forecasts["dataset"] == BASELINE_DATASET)
    ].copy()

    if base.empty:
        continue

    for gt_label, gt_dataset in GT_DATASETS.items():
        gt = all_forecasts[
            (all_forecasts["config_key"] == config_key)
            & (all_forecasts["dataset"] == gt_dataset)
        ].copy()

        if gt.empty:
            continue

        paired = pd.merge(
            base[["target_quarter", "vintage_label", "error", "abs_error"]],
            gt[["target_quarter", "vintage_label", "error", "abs_error"]],
            on=["target_quarter", "vintage_label"],
            suffixes=("_base", "_gt"),
            how="inner",
        )

        if paired.empty:
            continue

        rmse_base = rmse(paired["error_base"])
        rmse_gt = rmse(paired["error_gt"])
        mae_base = mae(paired["error_base"])
        mae_gt = mae(paired["error_gt"])

        rmse_reduction_pct = (
            (rmse_base - rmse_gt) / rmse_base * 100
            if rmse_base != 0 else np.nan
        )

        mae_reduction_pct = (
            (mae_base - mae_gt) / mae_base * 100
            if mae_base != 0 else np.nan
        )

        unstable = bool(
            (rmse_base > UNSTABLE_RMSE_THRESHOLD)
            or (rmse_gt > UNSTABLE_RMSE_THRESHOLD)
        )

        comparison_rows.append({
            "model": cfg["model"],
            "gt_rinkinys": gt_label,
            "config_key": config_key,
            "config_label": cfg["config_label"],
            "n": len(paired),
            "rmse_be_gt": rmse_base,
            "rmse_su_gt": rmse_gt,
            "rmse_sumazejimas_pct": rmse_reduction_pct,
            "mae_be_gt": mae_base,
            "mae_su_gt": mae_gt,
            "mae_sumazejimas_pct": mae_reduction_pct,
            "unstable": unstable,
        })

        for vintage, g in paired.groupby("vintage_label"):
            rmse_base_v = rmse(g["error_base"])
            rmse_gt_v = rmse(g["error_gt"])

            vintage_rows.append({
                "model": cfg["model"],
                "gt_rinkinys": gt_label,
                "config_key": config_key,
                "config_label": cfg["config_label"],
                "vintage_label": vintage,
                "n": len(g),
                "rmse_be_gt": rmse_base_v,
                "rmse_su_gt": rmse_gt_v,
                "rmse_sumazejimas_pct": (
                    (rmse_base_v - rmse_gt_v) / rmse_base_v * 100
                    if rmse_base_v != 0 else np.nan
                ),
                "unstable": bool(
                    (rmse_base_v > UNSTABLE_RMSE_THRESHOLD)
                    or (rmse_gt_v > UNSTABLE_RMSE_THRESHOLD)
                ),
            })

overall_config = pd.DataFrame(comparison_rows)
by_vintage_config = pd.DataFrame(vintage_rows)

overall_config.to_csv(
    OUT_DIR / "gt_rmse_reduction_by_config_dfm_only.csv",
    index=False,
    encoding="utf-8-sig",
)

by_vintage_config.to_csv(
    OUT_DIR / "gt_rmse_reduction_by_config_and_vintage_dfm_only.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# STABLE SUMMARIES
# ============================================================

stable_overall = overall_config[~overall_config["unstable"]].copy()
stable_by_vintage = by_vintage_config[~by_vintage_config["unstable"]].copy()

summary_model_gt = (
    stable_overall
    .groupby(["model", "gt_rinkinys"], dropna=False)
    .agg(
        configs=("config_key", "nunique"),
        mean_rmse_reduction_pct=("rmse_sumazejimas_pct", "mean"),
        median_rmse_reduction_pct=("rmse_sumazejimas_pct", "median"),
        positive_configs=("rmse_sumazejimas_pct", lambda x: int((x > 0).sum())),
        negative_configs=("rmse_sumazejimas_pct", lambda x: int((x < 0).sum())),
        best_rmse_reduction_pct=("rmse_sumazejimas_pct", "max"),
        worst_rmse_reduction_pct=("rmse_sumazejimas_pct", "min"),
    )
    .reset_index()
)

summary_model_gt["positive_configs_pct"] = (
    summary_model_gt["positive_configs"] / summary_model_gt["configs"] * 100
)

summary_by_vintage = (
    stable_by_vintage
    .groupby(["model", "gt_rinkinys", "vintage_label"], dropna=False)
    .agg(
        configs=("config_key", "nunique"),
        mean_rmse_reduction_pct=("rmse_sumazejimas_pct", "mean"),
        median_rmse_reduction_pct=("rmse_sumazejimas_pct", "median"),
        positive_configs=("rmse_sumazejimas_pct", lambda x: int((x > 0).sum())),
        negative_configs=("rmse_sumazejimas_pct", lambda x: int((x < 0).sum())),
    )
    .reset_index()
)

summary_model_gt.to_csv(
    OUT_DIR / "gt_rmse_reduction_summary_by_model_dfm_only.csv",
    index=False,
    encoding="utf-8-sig",
)

summary_by_vintage.to_csv(
    OUT_DIR / "gt_rmse_reduction_summary_by_model_vintage_dfm_only.csv",
    index=False,
    encoding="utf-8-sig",
)

unstable_summary = (
    overall_config[overall_config["unstable"]]
    .groupby(["model", "gt_rinkinys"], dropna=False)
    .size()
    .reset_index(name="unstable_config_comparisons")
)

unstable_summary.to_csv(
    OUT_DIR / "gt_rmse_reduction_unstable_summary_dfm_only.csv",
    index=False,
    encoding="utf-8-sig",
)

best_improvements = (
    stable_overall
    .sort_values("rmse_sumazejimas_pct", ascending=False)
    .head(30)
    .copy()
)

best_improvements.to_csv(
    OUT_DIR / "gt_rmse_reduction_best_30_configs_dfm_only.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# LATEX
# ============================================================

def latex_summary_model(df):
    d = df.sort_values(["model", "gt_rinkinys"]).copy()

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Vidutinis RMSE pokytis pridėjus Google Trends duomenis DFM ir DFM-MF modeliams}",
        r"\label{tab:gt_rmse_reduction_summary_dfm}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Modelis & GT rinkinys & Konfig. sk. & Vid. RMSE sumaž., \% & Mediana, \% & Pagerėjo & Pagerėjo, \% & Blogiausias pokytis, \% \\",
        r"\midrule",
    ]

    for _, r in d.iterrows():
        lines.append(
            f"{r['model']} & "
            f"{r['gt_rinkinys']} & "
            f"{int(r['configs'])} & "
            f"{fmt_pct(r['mean_rmse_reduction_pct'])} & "
            f"{fmt_pct(r['median_rmse_reduction_pct'])} & "
            f"{int(r['positive_configs'])} & "
            f"{fmt_pct(r['positive_configs_pct'])} & "
            f"{fmt_pct(r['worst_rmse_reduction_pct'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def latex_best(df):
    d = df.copy()

    lines = [
        r"\begin{scriptsize}",
        r"\begin{longtable}{llp{5.0cm}rrrr}",
        r"\caption{Didžiausi RMSE sumažėjimai pridėjus Google Trends duomenis DFM ir DFM-MF modeliams}",
        r"\label{tab:gt_rmse_reduction_best_dfm}\\",
        r"\toprule",
        r"Modelis & GT rinkinys & Specifikacija & $n$ & RMSE be GT & RMSE su GT & Sumaž., \% \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Modelis & GT rinkinys & Specifikacija & $n$ & RMSE be GT & RMSE su GT & Sumaž., \% \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{Tęsinys kitame puslapyje}\\",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for _, r in d.iterrows():
        lines.append(
            f"{r['model']} & "
            f"{r['gt_rinkinys']} & "
            f"{latex_escape(r['config_label'])} & "
            f"{int(r['n'])} & "
            f"{fmt4(r['rmse_be_gt'])} & "
            f"{fmt4(r['rmse_su_gt'])} & "
            f"{fmt_pct(r['rmse_sumazejimas_pct'])} \\\\"
        )

    lines += [
        r"\end{longtable}",
        r"\end{scriptsize}",
    ]

    return "\n".join(lines)


tex1 = latex_summary_model(summary_model_gt)
tex2 = latex_best(best_improvements)

(OUT_DIR / "tab_gt_rmse_reduction_summary_dfm_only.tex").write_text(tex1, encoding="utf-8")
(OUT_DIR / "tab_gt_rmse_reduction_best_dfm_only.tex").write_text(tex2, encoding="utf-8")
(OUT_DIR / "gt_rmse_reduction_combined_tables_dfm_only.tex").write_text(tex1 + "\n\n" + tex2, encoding="utf-8")


# ============================================================
# PRINT
# ============================================================

print("\nSaved to:", OUT_DIR)

print("\nSUMMARY BY MODEL:")
print(summary_model_gt.to_string(index=False))

print("\nUNSTABLE SUMMARY:")
print(unstable_summary.to_string(index=False) if not unstable_summary.empty else "No unstable comparisons.")

print("\nBEST 20 IMPROVEMENTS:")
print(best_improvements[
    [
        "model", "gt_rinkinys", "config_label",
        "n", "rmse_be_gt", "rmse_su_gt", "rmse_sumazejimas_pct",
    ]
].head(20).to_string(index=False))