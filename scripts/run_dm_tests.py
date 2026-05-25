"""
Diebold-Mariano tests for nowcasting forecast CSV files.
Reads forecast CSVs, aligns forecasts by target_quarter, and computes DM tests
for selected, baseline, or pairwise comparisons.
"""

import argparse
import glob
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def dm_test(actual, pred_a, pred_b, loss="squared", horizon=1, hac_lags="auto", small_sample_correction=True):
    """
    Diebold-Mariano test for equal predictive accuracy.
    d_t = L(e_A,t) - L(e_B,t).
    mean_loss_diff < 0 means model A has lower average loss.
    mean_loss_diff > 0 means model B has lower average loss.
    """
    actual = np.asarray(actual, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred_a) & np.isfinite(pred_b)
    actual, pred_a, pred_b = actual[mask], pred_a[mask], pred_b[mask]
    n = len(actual)
    if n == 0:
        return np.nan, np.nan, np.nan, 0, 0, small_sample_correction

    if loss == "squared":
        d = (actual - pred_a) ** 2 - (actual - pred_b) ** 2
    elif loss == "absolute":
        d = np.abs(actual - pred_a) - np.abs(actual - pred_b)
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    d_bar = float(np.mean(d))
    h = max(0, int(horizon) - 1) if hac_lags == "auto" else max(0, int(hac_lags))

    centered = d - d_bar
    gamma = np.zeros(h + 1, dtype=float)
    for k in range(h + 1):
        if k == 0:
            gamma[k] = np.sum(centered ** 2) / n
        else:
            gamma[k] = np.sum(centered[k:] * centered[:-k]) / n

    v_hat = gamma[0] + 2.0 * np.sum(gamma[1:])
    if not np.isfinite(v_hat) or v_hat <= 1e-12:
        return 0.0, 1.0, d_bar, n, h, small_sample_correction

    dm_stat = d_bar / np.sqrt(v_hat / n)
    if small_sample_correction:
        correction_num = n + 1 - 2 * horizon + (horizon / n) * (horizon - 1)
        correction = np.sqrt(max(correction_num / n, 0.0))
        dm_stat *= correction
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value), d_bar, n, h, small_sample_correction


def pretty_dataset_name(dataset_name: str) -> str:
    mapping = {
        "baseline_common": "be GT",
        "final_thesis_baseline_common": "be GT",
        "common_plus_gt": "su GT",
        "final_thesis_common_plus_gt": "su GT",
        "common_plus_gt_v1": "su GT (angl.)",
        "final_thesis_common_plus_gt_v1": "su GT (angl.)",
        "common_plus_gt_lt": "su GT (liet.)",
        "final_thesis_common_plus_gt_lt": "su GT (liet.)",
        "gt_only": "tik GT",
        "final_thesis_gt_only": "tik GT",
        "gt_only_v1": "tik GT (angl.)",
        "final_thesis_gt_only_v1": "tik GT (angl.)",
        "gt_only_lt": "tik GT (liet.)",
        "final_thesis_gt_only_lt": "tik GT (liet.)",
    }
    return mapping.get(str(dataset_name), str(dataset_name))


def pretty_vintage(vintage: str) -> str:
    try:
        v = int(str(vintage).replace("+", ""))
        return f"+{v}" if v > 0 else str(v)
    except Exception:
        return str(vintage)


def result_interpretation_lt(p_value: float, better_model: str, model_a: str, model_b: str) -> str:
    if not np.isfinite(p_value) or p_value >= 0.05:
        return "Nėra reikšmingo skirtumo"
    if better_model == model_a:
        return "A reikšmingai geresnis"
    if better_model == model_b:
        return "B reikšmingai geresnis"
    return "Nėra reikšmingo skirtumo"


def _not_missing(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value) not in {"", "nan", "None", "unknown"}


def _fmt_num(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def _first_existing(row: pd.Series, names: Iterable[str]) -> Optional[Any]:
    for name in names:
        if name in row and _not_missing(row.get(name)):
            return row.get(name)
    return None


def _append_if_present(parts: List[str], row: pd.Series, names: Iterable[str], prefix: str):
    value = _first_existing(row, names)
    if value is not None:
        parts.append(f"{prefix}{_fmt_num(value)}")


def normalize_vintage_label(v) -> str:
    if pd.isna(v):
        return "unknown_vintage"
    s = str(v).replace("+", "").strip()
    try:
        return str(int(float(s)))
    except Exception:
        return s


def parse_metadata_from_filename(filename: str, df: pd.DataFrame) -> pd.DataFrame:
    basename = os.path.basename(filename)
    if "dataset_type" not in df.columns or df["dataset_type"].isna().all():
        known_datasets = [
            "final_thesis_common_plus_gt_lt", "final_thesis_common_plus_gt_v1",
            "final_thesis_common_plus_gt", "final_thesis_baseline_common",
            "final_thesis_gt_only_lt", "final_thesis_gt_only_v1", "final_thesis_gt_only",
            "common_plus_gt_lt", "common_plus_gt_v1", "common_plus_gt",
            "baseline_common", "gt_only_lt", "gt_only_v1", "gt_only",
        ]
        df["dataset_type"] = next((d for d in known_datasets if d in basename), "unknown_dataset")

    if "model" not in df.columns or df["model"].isna().all():
        if "DFM_MF" in basename:
            df["model"] = "DFM_MF"
        elif "MIDASML" in basename:
            df["model"] = "MIDASML"
        elif "ElasticNet" in basename:
            df["model"] = "ElasticNet"
        elif "MIDAS" in basename:
            df["model"] = "MIDAS"
        elif "DFM" in basename:
            df["model"] = "DFM"
        else:
            df["model"] = "unknown_model"

    if "fill_method" not in df.columns or df["fill_method"].isna().all():
        known_fills = ["vertical_realignment", "autoarima", "rolling_mean", "tactis2", "locf", "native_ragged", "none"]
        df["fill_method"] = next((f for f in known_fills if f in basename), "unknown_fill")

    if "vintage_label" not in df.columns:
        df["vintage_label"] = "unknown_vintage"
    return df


def parse_config_from_filename(source_file: str) -> Dict[str, str]:
    name = os.path.basename(source_file)
    parsed: Dict[str, str] = {}
    m = re.search(r"fixed_a([^_]+)_l1([^_]+)", name)
    if m:
        parsed["elasticnet_alpha"] = m.group(1)
        parsed["elasticnet_l1_ratio"] = m.group(2)
    m = re.search(r"DFM_agg([^_]+)_k(\d+)_p(\d+)_sel([^_]+)", name)
    if m:
        parsed.update({"quarterly_aggregation": m.group(1), "dfm_k_factors": m.group(2), "dfm_factor_order": m.group(3), "dfm_selector": m.group(4)})
    m = re.search(r"DFM_MF_k(\d+)_p(\d+)_mfsel(.+?)_top(\d+)", name)
    if m:
        parsed.update({"dfm_k_factors": m.group(1), "dfm_factor_order": m.group(2), "dfm_mf_selector": m.group(3), "dfm_mf_top_n": m.group(4)})
    m = re.search(r"MIDAS_([^_]+)_lags(\d+)_fill(.+?)_final_thesis", name)
    if m:
        parsed.update({"midas_regression_model": m.group(1), "midas_n_lags": m.group(2), "midas_internal_fill_strategy": m.group(3)})
    m = re.search(r"MIDASML_([^_]+)_lags(\d+)_cv(\d+)_l1([^_]+)_fill(.+?)_final_thesis", name)
    if m:
        parsed.update({"midasml_regression_model": m.group(1), "midas_n_lags": m.group(2), "midasml_cv": m.group(3), "midasml_l1_ratio": m.group(4), "midas_internal_fill_strategy": m.group(5)})
    return parsed


def determine_model_config(row: pd.Series, source_file: Optional[str] = None) -> str:
    filename_parts = parse_config_from_filename(source_file or "")
    model = str(row.get("model", "UnknownModel"))
    fill = str(row.get("fill_method", ""))
    parts: List[str] = [model]

    if model == "ElasticNet":
        if fill and fill != "unknown_fill":
            parts.append(fill)
        vr_mode = _first_existing(row, ["vertical_realignment_mode", "vr_mode"])
        if fill == "vertical_realignment" and vr_mode is not None:
            parts.append(str(vr_mode))
        alpha = _first_existing(row, ["elasticnet_alpha", "alpha"]) or filename_parts.get("elasticnet_alpha")
        l1 = _first_existing(row, ["elasticnet_l1_ratio", "l1_ratio"]) or filename_parts.get("elasticnet_l1_ratio")
        if alpha is not None:
            parts.append(f"alpha{_fmt_num(alpha)}")
        if l1 is not None:
            parts.append(f"l1{_fmt_num(l1)}")
        selector = _first_existing(row, ["selector", "feature_selector", "selector_method"])
        if selector is not None and str(selector) != "none":
            parts.append(f"sel{selector}")
        _append_if_present(parts, row, ["pca_components", "dfm_pca_components"], "pca")
        _append_if_present(parts, row, ["top_n", "corr_top_n", "dfm_mf_top_n"], "top")

    elif model == "DFM":
        k = _first_existing(row, ["dfm_k_factors", "k_factors"]) or filename_parts.get("dfm_k_factors")
        p = _first_existing(row, ["dfm_factor_order", "factor_order"]) or filename_parts.get("dfm_factor_order")
        selector = _first_existing(row, ["dfm_selector", "selector"]) or filename_parts.get("dfm_selector")
        pca = _first_existing(row, ["dfm_pca_components", "pca_components"])
        agg = _first_existing(row, ["quarterly_aggregation"]) or filename_parts.get("quarterly_aggregation")
        if k is not None: parts.append(f"k{_fmt_num(k)}")
        if p is not None: parts.append(f"p{_fmt_num(p)}")
        if selector is not None: parts.append(str(selector))
        if pca is not None: parts.append(f"pca{_fmt_num(pca)}")
        if agg is not None: parts.append(f"agg{agg}")

    elif model == "DFM_MF":
        k = _first_existing(row, ["dfm_k_factors", "k_factors"]) or filename_parts.get("dfm_k_factors")
        p = _first_existing(row, ["dfm_factor_order", "factor_order"]) or filename_parts.get("dfm_factor_order")
        selector = _first_existing(row, ["dfm_mf_selector"]) or filename_parts.get("dfm_mf_selector")
        top_n = _first_existing(row, ["dfm_mf_top_n"]) or filename_parts.get("dfm_mf_top_n")
        if k is not None: parts.append(f"k{_fmt_num(k)}")
        if p is not None: parts.append(f"p{_fmt_num(p)}")
        if selector is not None: parts.append(str(selector).replace("_", ""))
        if top_n is not None: parts.append(f"top{_fmt_num(top_n)}")

    elif model == "MIDAS":
        reg = _first_existing(row, ["midas_regression_model"]) or filename_parts.get("midas_regression_model")
        lags = _first_existing(row, ["midas_n_lags"]) or filename_parts.get("midas_n_lags")
        fill_strategy = _first_existing(row, ["midas_internal_fill_strategy"]) or filename_parts.get("midas_internal_fill_strategy")
        if reg is not None: parts.append(str(reg))
        if lags is not None: parts.append(f"lags{_fmt_num(lags)}")
        if fill_strategy is not None: parts.append(str(fill_strategy))

    elif model == "MIDASML":
        reg = _first_existing(row, ["midasml_regression_model"]) or filename_parts.get("midasml_regression_model")
        lags = _first_existing(row, ["midas_n_lags"]) or filename_parts.get("midas_n_lags")
        cv = _first_existing(row, ["midasml_cv"]) or filename_parts.get("midasml_cv")
        l1 = _first_existing(row, ["midasml_l1_ratio"]) or filename_parts.get("midasml_l1_ratio")
        fill_strategy = _first_existing(row, ["midas_internal_fill_strategy"]) or filename_parts.get("midas_internal_fill_strategy")
        if reg is not None: parts.append(str(reg))
        if lags is not None: parts.append(f"lags{_fmt_num(lags)}")
        if cv is not None: parts.append(f"cv{_fmt_num(cv)}")
        if l1 is not None: parts.append(f"l1{_fmt_num(l1)}")
        if fill_strategy is not None: parts.append(str(fill_strategy))
    else:
        if fill and fill != "unknown_fill":
            parts.append(fill)

    clean_parts = []
    for p in parts:
        s = str(p).strip()
        if s and s not in {"nan", "None", "unknown"}:
            clean_parts.append(s)
    return "_".join(clean_parts).replace("__", "_")


def load_forecast_files(input_glob: str) -> Tuple[pd.DataFrame, int, int]:
    files = glob.glob(input_glob)
    if not files:
        logger.error("No files matched by glob: %s", input_glob)
        return pd.DataFrame(), 0, 0
    logger.info("Found %d files to process.", len(files))
    all_data, skipped = [], 0
    for file in sorted(files):
        try:
            df = pd.read_csv(file)
            if df.empty:
                skipped += 1
                continue
            missing = {"prediction", "actual", "target_quarter"} - set(df.columns)
            if missing:
                logger.warning("Skipping %s, missing required columns: %s", file, sorted(missing))
                skipped += 1
                continue
            df = parse_metadata_from_filename(file, df)
            df["source_file"] = os.path.basename(file)
            df["source_path"] = str(file)
            df["vintage_label"] = df["vintage_label"].apply(normalize_vintage_label)
            df = df.dropna(subset=["prediction", "actual", "target_quarter"])
            if df.empty:
                skipped += 1
                continue
            df["model_config"] = df.apply(lambda row: determine_model_config(row, source_file=file), axis=1)
            all_data.append(df)
        except Exception as exc:
            logger.exception("Failed to process %s: %s", file, exc)
            skipped += 1
    if not all_data:
        return pd.DataFrame(), len(files), skipped
    combined = pd.concat(all_data, ignore_index=True)
    logger.info("Loaded %d valid forecast rows from %d valid files.", len(combined), len(all_data))
    return combined, len(files), skipped


def deduplicate_forecasts(df: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, int]:
    keys = ["dataset_type", "vintage_label", "target_quarter", "model_config"]
    dup_counts = df.groupby(keys, dropna=False).size().reset_index(name="duplicate_count").query("duplicate_count > 1")
    dup_counts.to_csv(out_dir / "dm_duplicate_diagnostics.csv", index=False)
    if dup_counts.empty:
        logger.info("No duplicated forecast keys detected.")
        return df.copy(), 0
    logger.warning("Detected %d duplicated forecast keys. Aggregating duplicates by mean.", len(dup_counts))
    agg = df.groupby(keys, dropna=False).agg(
        prediction=("prediction", "mean"),
        actual=("actual", "mean"),
        source_file=("source_file", lambda x: ";".join(sorted(set(map(str, x)))[:3])),
        source_path=("source_path", lambda x: ";".join(sorted(set(map(str, x)))[:3])),
        model=("model", "first"),
        fill_method=("fill_method", "first"),
    ).reset_index()
    return agg, int(len(dup_counts))


def build_series_table(df: pd.DataFrame, actual_tolerance: float = 1e-10) -> pd.DataFrame:
    actual_checks = df.groupby(["dataset_type", "vintage_label", "target_quarter"], dropna=False)["actual"].agg(["min", "max", "count"]).reset_index()
    inconsistent = actual_checks[(actual_checks["max"] - actual_checks["min"]).abs() > actual_tolerance]
    if not inconsistent.empty:
        logger.warning("Actual values differ across models for %d dataset/vintage/quarter groups.", len(inconsistent))
    rows = []
    for (dataset, vintage, model_config), g in df.groupby(["dataset_type", "vintage_label", "model_config"], dropna=False):
        g = g.sort_values("target_quarter")
        rows.append({
            "dataset_type": dataset,
            "vintage_label": vintage,
            "model_config": model_config,
            "source_file": ";".join(sorted(set(map(str, g.get("source_file", []))))[:3]),
            "prediction_series": g.set_index("target_quarter")["prediction"],
            "actual_series": g.set_index("target_quarter")["actual"],
        })
    return pd.DataFrame(rows)


def safe_regex_match(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, str(text)) is not None
    except re.error as exc:
        logger.warning("Invalid regex %r: %s", pattern, exc)
        return False


def align_for_comparison(row_a: pd.Series, row_b: pd.Series, actual_tolerance: float = 1e-8):
    pred_a, pred_b = row_a["prediction_series"], row_b["prediction_series"]
    actual_a, actual_b = row_a["actual_series"], row_b["actual_series"]
    idx = pred_a.dropna().index.intersection(pred_b.dropna().index)
    idx = idx.intersection(actual_a.dropna().index).intersection(actual_b.dropna().index)
    if len(idx) == 0:
        return np.array([]), np.array([]), np.array([]), []
    a_actual = actual_a.loc[idx].astype(float)
    b_actual = actual_b.loc[idx].astype(float)
    valid_idx = (a_actual - b_actual).abs().loc[lambda s: s <= actual_tolerance].index
    actual = ((a_actual.loc[valid_idx] + b_actual.loc[valid_idx]) / 2.0).values
    return actual, pred_a.loc[valid_idx].astype(float).values, pred_b.loc[valid_idx].astype(float).values, list(valid_idx)


def build_pairwise_comparisons(series_df: pd.DataFrame):
    comps = []
    for (_, _), g in series_df.groupby(["dataset_type", "vintage_label"], dropna=False):
        rows = list(g.sort_values("model_config").iterrows())
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                comps.append(("pairwise", rows[i][1], rows[j][1]))
    return comps


def build_baseline_comparisons(series_df: pd.DataFrame, baseline_pattern: str):
    comps = []
    for (dataset, vintage), g in series_df.groupby(["dataset_type", "vintage_label"], dropna=False):
        matched = g[g["model_config"].apply(lambda x: safe_regex_match(baseline_pattern, x))].sort_values("model_config")
        if matched.empty:
            logger.warning("No baseline matched %r in dataset=%s vintage=%s", baseline_pattern, dataset, vintage)
            continue
        baseline = matched.iloc[0]
        if len(matched) > 1:
            logger.info("Multiple baselines matched in %s %s, using %s", dataset, vintage, baseline["model_config"])
        for _, row in g.sort_values("model_config").iterrows():
            if row["model_config"] != baseline["model_config"]:
                comps.append(("baseline", row, baseline))
    return comps


def write_comparison_template(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame([
        {"comparison_name": "GT_LT_vs_no_GT", "dataset_pattern": r"final_thesis_(baseline_common|common_plus_gt_lt)$", "vintage_pattern": r".*", "model_a_pattern": r"ElasticNet_autoarima_alpha0\.001_l10\.25", "model_b_pattern": r"ElasticNet_autoarima_alpha0\.001_l10\.25"},
        {"comparison_name": "GT_EN_vs_no_GT", "dataset_pattern": r"final_thesis_(baseline_common|common_plus_gt_v1)$", "vintage_pattern": r".*", "model_a_pattern": r"ElasticNet_autoarima_alpha0\.001_l10\.25", "model_b_pattern": r"ElasticNet_autoarima_alpha0\.001_l10\.25"},
        {"comparison_name": "AutoARIMA_vs_LOCF", "dataset_pattern": r".*", "vintage_pattern": r".*", "model_a_pattern": r"ElasticNet_autoarima_alpha0\.001_l10\.25", "model_b_pattern": r"ElasticNet_locf_alpha0\.001_l10\.25"},
        {"comparison_name": "VR_vs_AutoARIMA", "dataset_pattern": r".*", "vintage_pattern": r".*", "model_a_pattern": r"ElasticNet_vertical_realignment.*alpha0\.001_l10\.25", "model_b_pattern": r"ElasticNet_autoarima_alpha0\.001_l10\.25"},
        {"comparison_name": "MIDAS_vs_ElasticNet", "dataset_pattern": r".*", "vintage_pattern": r".*", "model_a_pattern": r"MIDAS_ridge_lags4_ffill_then_zero", "model_b_pattern": r"ElasticNet_autoarima_alpha0\.001_l10\.25"},
        {"comparison_name": "DFM_vs_ElasticNet", "dataset_pattern": r".*", "vintage_pattern": r".*", "model_a_pattern": r"DFM_k3_p2.*aggmean", "model_b_pattern": r"ElasticNet_autoarima_alpha0\.001_l10\.25"},
    ])
    template.to_csv(path, index=False)
    logger.info("Wrote template to %s", path)


def build_selected_comparisons(series_df: pd.DataFrame, comparisons_csv: Path):
    spec = pd.read_csv(comparisons_csv)
    required = {"dataset_pattern", "vintage_pattern", "model_a_pattern", "model_b_pattern", "comparison_name"}
    missing = required - set(spec.columns)
    if missing:
        raise ValueError(f"Comparisons CSV missing columns: {sorted(missing)}")
    comps = []
    for _, rule in spec.iterrows():
        name, ds_pat, v_pat = str(rule["comparison_name"]), str(rule["dataset_pattern"]), str(rule["vintage_pattern"])
        a_pat, b_pat = str(rule["model_a_pattern"]), str(rule["model_b_pattern"])
        subset = series_df[
            series_df["dataset_type"].apply(lambda x: safe_regex_match(ds_pat, x)) &
            series_df["vintage_label"].apply(lambda x: safe_regex_match(v_pat, x))
        ]
        for vintage, g_v in subset.groupby("vintage_label", dropna=False):
            a_rows = g_v[g_v["model_config"].apply(lambda x: safe_regex_match(a_pat, x))]
            b_rows = g_v[g_v["model_config"].apply(lambda x: safe_regex_match(b_pat, x))]
            for _, a in a_rows.sort_values(["dataset_type", "model_config"]).iterrows():
                for _, b in b_rows.sort_values(["dataset_type", "model_config"]).iterrows():
                    if a["dataset_type"] == b["dataset_type"] and a["model_config"] == b["model_config"] and a["source_file"] == b["source_file"]:
                        continue
                    comps.append((name, a, b))
    logger.info("Built %d selected comparisons from %s", len(comps), comparisons_csv)
    return comps


def run_comparisons(comparisons, loss, min_obs, horizon, hac_lags, small_sample_correction):
    results, skipped = [], 0
    for comparison_name, row_a, row_b in comparisons:
        actual, pred_a, pred_b, used_quarters = align_for_comparison(row_a, row_b)
        if len(actual) < min_obs:
            skipped += 1
            continue
        dm_stat, p_value, mean_loss_diff, n_obs, h_lags, correction_used = dm_test(actual, pred_a, pred_b, loss, horizon, hac_lags, small_sample_correction)
        model_a, model_b = row_a["model_config"], row_b["model_config"]
        better = model_a if mean_loss_diff < 0 else (model_b if mean_loss_diff > 0 else "Tie")
        dataset_a, dataset_b = str(row_a["dataset_type"]), str(row_b["dataset_type"])
        dataset = dataset_a if dataset_a == dataset_b else f"{dataset_a} vs {dataset_b}"
        dataset_label_a, dataset_label_b = pretty_dataset_name(dataset_a), pretty_dataset_name(dataset_b)
        dataset_label = dataset_label_a if dataset_label_a == dataset_label_b else f"{dataset_label_a} vs {dataset_label_b}"
        vintage_a, vintage_b = str(row_a["vintage_label"]), str(row_b["vintage_label"])
        vintage = vintage_a if vintage_a == vintage_b else f"{vintage_a} vs {vintage_b}"
        results.append({
            "comparison_name": comparison_name,
            "dataset": dataset,
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "dataset_label": dataset_label,
            "vintage": vintage,
            "vintage_a": vintage_a,
            "vintage_b": vintage_b,
            "vintage_label_pretty": pretty_vintage(vintage),
            "model_a": model_a,
            "model_b": model_b,
            "model_a_label": model_a,
            "model_b_label": model_b,
            "file_a": row_a.get("source_file", ""),
            "file_b": row_b.get("source_file", ""),
            "n_obs": n_obs,
            "first_target_quarter": min(used_quarters) if used_quarters else "",
            "last_target_quarter": max(used_quarters) if used_quarters else "",
            "loss": loss,
            "mean_loss_diff": mean_loss_diff,
            "dm_stat": dm_stat,
            "p_value": p_value,
            "significant_10pct": bool(p_value < 0.10),
            "significant_5pct": bool(p_value < 0.05),
            "significant_1pct": bool(p_value < 0.01),
            "better_model": better,
            "result_interpretation_lt": result_interpretation_lt(p_value, better, model_a, model_b),
            "hac_lags": h_lags,
            "correction_used": correction_used,
        })
    return pd.DataFrame(results), skipped


def create_summary_by_model(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, vintage), g in results_df.groupby(["dataset", "vintage"], dropna=False):
        models = sorted(set(g["model_a"]).union(set(g["model_b"])))
        for model in models:
            as_a, as_b = g[g["model_a"] == model], g[g["model_b"] == model]
            wins = int(((as_a["mean_loss_diff"] < 0) & as_a["significant_5pct"]).sum() + ((as_b["mean_loss_diff"] > 0) & as_b["significant_5pct"]).sum())
            losses = int(((as_a["mean_loss_diff"] > 0) & as_a["significant_5pct"]).sum() + ((as_b["mean_loss_diff"] < 0) & as_b["significant_5pct"]).sum())
            diffs = list(as_a["mean_loss_diff"]) + list(-as_b["mean_loss_diff"])
            pvals = list(as_a["p_value"]) + list(as_b["p_value"])
            rows.append({"dataset": dataset, "vintage": vintage, "model": model, "num_comparisons": len(diffs), "significant_wins_5pct": wins, "significant_losses_5pct": losses, "average_mean_loss_diff_from_model_perspective": float(np.mean(diffs)) if diffs else np.nan, "average_p_value": float(np.mean(pvals)) if pvals else np.nan})
    return pd.DataFrame(rows)


def create_summary_by_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    rows = []
    for (comparison_name, dataset, vintage), g in results_df.groupby(["comparison_name", "dataset", "vintage"], dropna=False):
        significant = g[g["significant_5pct"]]
        best = ""
        if not significant.empty:
            counts = significant["better_model"].value_counts()
            if len(counts) == 1 or counts.iloc[0] > counts.iloc[1]:
                best = counts.index[0]
        rows.append({"comparison_name": comparison_name, "dataset": dataset, "vintage": vintage, "number_of_tests": len(g), "number_significant_5pct": int(g["significant_5pct"].sum()), "best_model_if_dominates": best, "average_p_value": float(g["p_value"].mean()), "average_abs_mean_loss_diff": float(g["mean_loss_diff"].abs().mean())})
    return pd.DataFrame(rows)


def write_report(path: Path, args, n_files, skipped_files, n_rows, n_model_configs, n_groups, n_comparisons_built, n_comparisons_run, n_skipped_min_obs, n_duplicates, results_df):
    command = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Diebold-Mariano Test Report ===\n\n")
        f.write(f"Command: {command}\nInput glob: {args.input_glob}\nMode: {args.mode}\nLoss: {args.loss}\nMin obs: {args.min_obs}\nHorizon: {args.horizon}\nHAC lags: {args.hac_lags}\nSmall sample correction: {args.small_sample_correction}\n\n")
        f.write("Run statistics:\n")
        f.write(f"- Files matched: {n_files}\n- Files skipped: {skipped_files}\n- Valid forecast rows: {n_rows}\n- Unique model configurations: {n_model_configs}\n- Dataset/vintage groups: {n_groups}\n- Comparisons built: {n_comparisons_built}\n- Comparisons run: {n_comparisons_run}\n- Comparisons skipped due to min_obs: {n_skipped_min_obs}\n- Duplicate forecast keys detected: {n_duplicates}\n\n")
        f.write("Methodological notes:\n")
        f.write("- The DM test compares predictive accuracy of two competing forecasts.\n")
        f.write("- H0: equal predictive accuracy.\n")
        f.write("- d_t = L_A,t - L_B,t. Negative mean loss differential means model A has lower average loss.\n")
        f.write("- Kadangi pseudo-realaus laiko vertinimo imtis yra santykinai maža, Diebold--Mariano testo rezultatai turėtų būti interpretuojami kaip papildomas, o ne vienintelis modelių palyginimo kriterijus.\n\n")
        f.write("Top significant results at 5% level:\n")
        if results_df.empty or not results_df["significant_5pct"].any():
            f.write("No significant differences found at the 5% level.\n")
        else:
            top = results_df[results_df["significant_5pct"]].sort_values("p_value").head(20)
            for _, row in top.iterrows():
                worse = row["model_b"] if row["better_model"] == row["model_a"] else row["model_a"]
                f.write(f"- {row['dataset']} | Vintage {row['vintage']}: {row['better_model']} beats {worse} (p={row['p_value']:.4f}, mean loss diff={row['mean_loss_diff']:.6g}, n={row['n_obs']})\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Diebold-Mariano forecast comparison tests.")
    parser.add_argument("--input-glob", required=False, help="Glob pattern for forecast CSVs.")
    parser.add_argument("--run-name", default="default_run")
    parser.add_argument("--mode", choices=["pairwise", "baseline", "selected"], default="selected")
    parser.add_argument("--loss", choices=["squared", "absolute"], default="squared")
    parser.add_argument("--min-obs", type=int, default=8)
    parser.add_argument("--baseline-pattern", type=str, default=None)
    parser.add_argument("--comparisons-csv", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hac-lags", default="auto")
    parser.add_argument("--small-sample-correction", action="store_true", default=True)
    parser.add_argument("--no-small-sample-correction", dest="small_sample_correction", action="store_false")
    parser.add_argument("--out-root", default="data/forecasts/dm_tests")
    parser.add_argument("--write-template", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    if args.write_template:
        write_comparison_template(out_root / "comparisons_template.csv")
        if not args.input_glob:
            return
    if not args.input_glob:
        raise SystemExit("--input-glob is required unless only --write-template is used.")

    out_dir = out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df, n_files, skipped_files = load_forecast_files(args.input_glob)
    if df.empty:
        logger.error("No valid data loaded.")
        return

    df, duplicate_count = deduplicate_forecasts(df, out_dir)
    series_df = build_series_table(df)
    n_model_configs = int(series_df["model_config"].nunique())
    n_groups = int(series_df.groupby(["dataset_type", "vintage_label"], dropna=False).ngroups)
    logger.info("Unique model_config values found: %d", n_model_configs)
    logger.info("Dataset/vintage groups found: %d", n_groups)

    if args.mode == "pairwise":
        comparisons = build_pairwise_comparisons(series_df)
    elif args.mode == "baseline":
        if not args.baseline_pattern:
            raise SystemExit("--baseline-pattern is required when --mode baseline")
        comparisons = build_baseline_comparisons(series_df, args.baseline_pattern)
    else:
        if not args.comparisons_csv:
            raise SystemExit("--comparisons-csv is required when --mode selected")
        comparisons = build_selected_comparisons(series_df, Path(args.comparisons_csv))

    results_df, skipped_min_obs = run_comparisons(comparisons, args.loss, args.min_obs, args.horizon, args.hac_lags, args.small_sample_correction)
    results_df.to_csv(out_dir / "dm_results.csv", index=False)
    create_summary_by_model(results_df).to_csv(out_dir / "dm_summary_by_model.csv", index=False)
    create_summary_by_comparison(results_df).to_csv(out_dir / "dm_summary_by_comparison.csv", index=False)
    write_report(out_dir / "dm_report.txt", args, n_files, skipped_files, len(df), n_model_configs, n_groups, len(comparisons), len(results_df), skipped_min_obs, duplicate_count, results_df)
    logger.info("Saved outputs to %s", out_dir)
    logger.info("Comparisons run: %d", len(results_df))
    logger.info("Comparisons skipped due to min_obs: %d", skipped_min_obs)


if __name__ == "__main__":
    main()
