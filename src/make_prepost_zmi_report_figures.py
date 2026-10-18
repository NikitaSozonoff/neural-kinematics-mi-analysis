"""Create a zMI report with absolute surrogate-normalized MI values.

The report is intentionally descriptive: it shows absolute ``zMI`` separately
for PreNFB and PostNFB.  It does not compute or plot Post-minus-Pre contrasts.

Input data are existing ``summary.csv`` files from the MI surrogate pipeline.
The script is read-only with respect to existing run folders and writes all
outputs to a new timestamped report directory.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from make_invasive_movement_comparison_figures import (
    DEFAULT_RESULTS_ROOT,
    parse_run_name,
)

try:
    import mne
except Exception:  # pragma: no cover - handled at runtime with a clear warning
    mne = None


DEFAULT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT.parent / "zMI_report"
REQUIRED_COLUMNS = {"group", "channel", "band_fmin", "band_fmax", "MI_z"}
REPORT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
}
TOPO_BANDS_MAIN = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
}
BETA_RANGE = (13.0, 30.0)
CONDITION_COLORS = {"PreNFB": "#2878b5", "PostNFB": "#d1495b"}
INVASIVE_BETA_BUBBLE_COLUMNS = [
    ("Clenching", "LeftHand", "PreNFB", "Cl-L Pre"),
    ("Clenching", "LeftHand", "PostNFB", "Cl-L Post"),
    ("Clenching", "RightHand", "PreNFB", "Cl-R Pre"),
    ("Clenching", "RightHand", "PostNFB", "Cl-R Post"),
    ("Rotation", "LeftHand", "PreNFB", "Rot-L Pre"),
    ("Rotation", "LeftHand", "PostNFB", "Rot-L Post"),
    ("Rotation", "RightHand", "PreNFB", "Rot-R Pre"),
    ("Rotation", "RightHand", "PostNFB", "Rot-R Post"),
    ("Tapping", "LeftHand", "PreNFB", "Tap-L Pre"),
    ("Tapping", "LeftHand", "PostNFB", "Tap-L Post"),
    ("Tapping", "RightHand", "PreNFB", "Tap-R Pre"),
    ("Tapping", "RightHand", "PostNFB", "Tap-R Post"),
]
TARGET_ALIASES = {
    "Angle": {"angle", "ang", "mocapang"},
    "Velocity": {"velocity", "vel", "mocapvel"},
}
GROUP_ALIASES = {
    "invasive": {"invasive", "inv"},
    "noninvasive": {"noninvasive", "eeg", "noninvasiveeeg", "scalp"},
}
METADATA_COLUMNS = [
    "run_id",
    "run_folder",
    "patient",
    "day",
    "movement",
    "hand",
    "condition",
    "target",
    "target_label",
    "n_surrogates",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a descriptive zMI report from existing MI surrogate summary.csv "
            "files. This report plots absolute zMI only, not Post-minus-Pre."
        )
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--patient", default="Patient1")
    parser.add_argument("--day", default="Day2")
    parser.add_argument(
        "--movements", nargs="+", default=["Clenching", "Rotation", "Tapping"]
    )
    parser.add_argument("--hands", nargs="+", default=["LeftHand", "RightHand"])
    parser.add_argument("--targets", nargs="+", default=["Angle", "Velocity"])
    parser.add_argument("--conditions", nargs="+", default=["PreNFB", "PostNFB"])
    parser.add_argument("--groups", nargs="+", default=["noninvasive", "invasive"])
    parser.add_argument("--n-surrogates", type=int, default=200)
    parser.add_argument(
        "--topomap-vlim",
        choices=["fixed", "robust", "auto"],
        default="robust",
        help="Color scale for topomaps. fixed uses 0..5; robust uses a high percentile.",
    )
    parser.add_argument("--topomap-cmap", default="magma")
    parser.add_argument("--heatmap-cmap", default="magma")
    parser.add_argument(
        "--plot-noninvasive-rect-heatmaps",
        action="store_true",
        help="Also save rectangular noninvasive channel x frequency zMI heatmaps.",
    )
    parser.add_argument("--save-svg", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def warn(message: str, warnings: list[str] | None = None) -> None:
    print(f"WARNING: {message}", file=sys.stderr)
    if warnings is not None:
        warnings.append(message)


def canonical_hand(value: str) -> str:
    norm = normalize_token(value)
    if norm in {"left", "lefthand", "lh"}:
        return "LeftHand"
    if norm in {"right", "righthand", "rh"}:
        return "RightHand"
    return value


def canonical_target(value: str) -> str:
    norm = normalize_token(value)
    for label, aliases in TARGET_ALIASES.items():
        if norm in aliases:
            return label
    return str(value)


def canonical_group(value: str) -> str:
    norm = normalize_token(value)
    for label, aliases in GROUP_ALIASES.items():
        if norm in aliases:
            return label
    return str(value)


def target_matches_any(found: str, requested: list[str]) -> bool:
    found_label = canonical_target(found)
    found_norm = normalize_token(found)
    for value in requested:
        requested_label = canonical_target(value)
        requested_norm = normalize_token(value)
        if requested_label in TARGET_ALIASES:
            if found_label == requested_label:
                return True
        elif found_norm == requested_norm:
            return True
    return False


def unique_output_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = output_root / f"zMI_report_{stamp}"
    candidate = base
    counter = 2
    while candidate.exists():
        candidate = Path(f"{base}_{counter}")
        counter += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def assign_report_band(center: pd.Series) -> pd.Series:
    choices = list(REPORT_BANDS)
    conditions = [
        center.ge(low) & center.lt(high) for low, high in REPORT_BANDS.values()
    ]
    return pd.Series(np.select(conditions, choices, default="other"), index=center.index)


def robust_upper(values: pd.Series | np.ndarray, percentile: float = 0.98, fallback: float = 1.0) -> float:
    array = pd.Series(np.asarray(values, dtype=float).ravel()).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if array.empty:
        return fallback
    vmax = float(array.quantile(percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        return fallback
    return vmax


def annotation_color_for_value(
    value: float,
    cmap_name: str,
    vmin: float,
    vmax: float,
    *,
    luminance_threshold: float = 0.55,
) -> str:
    if not np.isfinite(value) or not np.isfinite(vmax) or vmax <= vmin:
        return "0.35"
    rgba = plt.get_cmap(cmap_name)(Normalize(vmin=vmin, vmax=vmax, clip=True)(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance >= luminance_threshold else "white"


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    args: argparse.Namespace,
    manifest: list[dict[str, object]],
    figure_type: str,
    group: str | None,
    target: str | None,
    status: str = "generated",
    message: str = "",
) -> None:
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    if args.save_svg:
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    manifest.append(
        {
            "filename": output_path.name,
            "figure_type": figure_type,
            "group": group or "",
            "target_label": target or "",
            "status": status,
            "message": message,
        }
    )


def load_matching_summaries(
    summary_files: list[Path], args: argparse.Namespace, warnings: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, object]] = []
    requested_movements = {value.casefold() for value in args.movements}
    requested_hands = {canonical_hand(value).casefold() for value in args.hands}
    requested_conditions = {value.casefold() for value in args.conditions}
    requested_groups = {canonical_group(value).casefold() for value in args.groups}

    for summary_path in summary_files:
        metadata = parse_run_name(summary_path.parent.name)
        if metadata is None:
            continue
        if not (
            str(metadata["patient"]).casefold() == args.patient.casefold()
            and str(metadata["day"]).casefold() == args.day.casefold()
            and str(metadata["movement"]).casefold() in requested_movements
            and str(metadata["hand"]).casefold() in requested_hands
            and str(metadata["condition"]).casefold() in requested_conditions
            and int(metadata["n_surrogates"]) == args.n_surrogates
            and target_matches_any(str(metadata["target"]), args.targets)
        ):
            continue

        try:
            frame = pd.read_csv(summary_path)
        except Exception as exc:
            warn(f"Could not read {summary_path}: {exc}", warnings)
            continue
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            warn(
                f"Skipped {summary_path}: missing columns {', '.join(sorted(missing))}",
                warnings,
            )
            continue

        frame = frame.copy()
        frame["group"] = frame["group"].astype(str).map(canonical_group)
        frame = frame.loc[frame["group"].str.casefold().isin(requested_groups)].copy()
        if frame.empty:
            continue

        for column in ("band_fmin", "band_fmax", "MI_z"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[["band_fmin", "band_fmax", "MI_z"]].isna().any(axis=1)
        if invalid.any():
            warn(f"Dropped {int(invalid.sum())} non-numeric rows from {summary_path}", warnings)
            frame = frame.loc[~invalid].copy()
        if frame.empty:
            continue

        frame["band_center"] = (frame["band_fmin"] + frame["band_fmax"]) / 2.0
        frame["analysis_band"] = assign_report_band(frame["band_center"])
        run_id = summary_path.parent.name
        row = {
            "run_id": run_id,
            "run_folder": str(summary_path.parent.resolve()),
            **metadata,
            "target_label": canonical_target(str(metadata["target"])),
        }
        for key, value in row.items():
            frame[key] = value
        frames.append(frame)

        for group, group_frame in frame.groupby("group", dropna=False):
            run_rows.append(
                {
                    **row,
                    "group": group,
                    "n_rows": len(group_frame),
                    "n_channels": group_frame["channel"].nunique(),
                    "n_bands": group_frame[["band_fmin", "band_fmax"]]
                    .drop_duplicates()
                    .shape[0],
                }
            )

    if not frames:
        return pd.DataFrame(), pd.DataFrame(run_rows)
    data = pd.concat(frames, ignore_index=True)

    # Duplicate rows can appear after migrations or repeated summaries.  Average
    # them before plotting so every channel x frequency cell is represented once.
    grouping = METADATA_COLUMNS + ["group", "channel", "band_fmin", "band_fmax", "band_center", "analysis_band"]
    data = data.groupby(grouping, as_index=False, dropna=False)["MI_z"].mean()
    return data, pd.DataFrame(run_rows)


def report_expected_runs(
    args: argparse.Namespace, runs: pd.DataFrame, warnings: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    found = set()
    if not runs.empty:
        found = {
            (
                str(row.movement),
                str(row.hand),
                str(row.target_label),
                str(row.condition),
                str(row.group),
            )
            for row in runs.itertuples(index=False)
        }
    print("Expected run summaries:")
    for movement, hand_raw, target_raw, condition, group_raw in product(
        args.movements,
        args.hands,
        args.targets,
        args.conditions,
        args.groups,
    ):
        hand = canonical_hand(hand_raw)
        target = canonical_target(target_raw)
        group = canonical_group(group_raw)
        status = "FOUND" if (movement, hand, target, condition, group) in found else "MISSING"
        label = f"{movement} / {hand} / {target} / {condition} / {group}"
        print(f"  {status:<7} {label}")
        if status == "MISSING":
            warn(f"Missing requested summary rows: {label}", warnings)
        rows.append(
            {
                "movement": movement,
                "hand": hand,
                "target_label": target,
                "condition": condition,
                "group": group,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def condition_band_summary(data: pd.DataFrame) -> pd.DataFrame:
    band_data = data.loc[data["analysis_band"].isin(REPORT_BANDS)].copy()
    grouping = [
        "patient",
        "day",
        "movement",
        "hand",
        "condition",
        "target_label",
        "group",
        "analysis_band",
    ]
    return (
        band_data.groupby(grouping, as_index=False, dropna=False)
        .agg(
            mean_zMI=("MI_z", "mean"),
            median_zMI=("MI_z", "median"),
            n_bins=("MI_z", "size"),
            n_channels=("channel", "nunique"),
            fraction_zMI_gt_2=("MI_z", lambda values: float(values.gt(2).mean())),
            fraction_zMI_gt_3=("MI_z", lambda values: float(values.gt(3).mean())),
        )
        .sort_values(grouping)
    )


def beta_channel_summary(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    beta = data.loc[
        data["band_center"].ge(BETA_RANGE[0]) & data["band_center"].lt(BETA_RANGE[1])
    ].copy()
    grouping = ["patient", "day", "movement", "hand", "target_label", "group", "channel"]
    condition_summary = (
        beta.groupby(grouping + ["condition"], as_index=False, dropna=False)
        .agg(
            mean_beta_zMI=("MI_z", "mean"),
            max_beta_zMI=("MI_z", "max"),
            n_beta_bins=("MI_z", "size"),
        )
    )

    rows: list[dict[str, object]] = []
    for keys, frame in condition_summary.groupby(grouping, dropna=False):
        row = dict(zip(grouping, keys))
        means = []
        for condition in args.conditions:
            selected = frame.loc[frame["condition"].eq(condition)]
            suffix = condition.replace("NFB", "").lower()
            if selected.empty:
                row[f"mean_beta_zMI_{suffix}"] = np.nan
                row[f"max_beta_zMI_{suffix}"] = np.nan
                row[f"n_beta_bins_{suffix}"] = 0
            else:
                mean_value = float(selected["mean_beta_zMI"].iloc[0])
                row[f"mean_beta_zMI_{suffix}"] = mean_value
                row[f"max_beta_zMI_{suffix}"] = float(selected["max_beta_zMI"].iloc[0])
                row[f"n_beta_bins_{suffix}"] = int(selected["n_beta_bins"].iloc[0])
                means.append(mean_value)
        row["mean_beta_zMI_avg"] = float(np.nanmean(means)) if means else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(grouping) if rows else pd.DataFrame()


def subset(
    frame: pd.DataFrame,
    movement: str | None = None,
    hand: str | None = None,
    target: str | None = None,
    condition: str | None = None,
    group: str | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    if movement is not None:
        mask &= frame["movement"].eq(movement)
    if hand is not None:
        mask &= frame["hand"].eq(hand)
    if target is not None:
        mask &= frame["target_label"].eq(target)
    if condition is not None:
        mask &= frame["condition"].eq(condition)
    if group is not None:
        mask &= frame["group"].eq(group)
    return frame.loc[mask]


def make_eeg_info_for_topomap(channels: list[str]):
    rename_for_montage = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }
    plot_channels = [rename_for_montage.get(str(ch), str(ch)) for ch in channels]
    info = mne.create_info(ch_names=plot_channels, sfreq=250.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="raise")
    return info


def topomap_vmax(values: pd.Series, args: argparse.Namespace) -> float:
    if args.topomap_vlim == "fixed":
        return 5.0
    if args.topomap_vlim == "auto":
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        return max(float(finite.max()), 1.0) if not finite.empty else 1.0
    return robust_upper(values, percentile=0.98, fallback=1.0)


def plot_topomap_cell(
    ax: plt.Axes,
    panel: pd.DataFrame,
    band_name: str,
    vmin: float,
    vmax: float,
    cmap: str,
    warnings: list[str],
):
    low, high = TOPO_BANDS_MAIN[band_name]
    band_panel = panel.loc[
        panel["band_center"].ge(low) & panel["band_center"].lt(high)
    ].copy()
    if band_panel.empty:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None
    by_channel = band_panel.groupby("channel", dropna=False)["MI_z"].mean()
    by_channel = by_channel.replace([np.inf, -np.inf], np.nan).dropna()
    if by_channel.shape[0] < 3:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None
    if mne is None:
        ax.text(0.5, 0.5, "MNE missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None
    try:
        info = make_eeg_info_for_topomap(by_channel.index.tolist())
        try:
            im, _ = mne.viz.plot_topomap(
                by_channel.to_numpy(dtype=float),
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap=cmap,
                vlim=(vmin, vmax),
            )
        except TypeError:
            im, _ = mne.viz.plot_topomap(
                by_channel.to_numpy(dtype=float),
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        return im
    except Exception as exc:
        warn(f"Could not draw topomap for {band_name}: {exc}", warnings)
        ax.text(0.5, 0.5, "topomap\nfailed", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None


def plot_noninvasive_topomap_panel(
    data: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    manifest: list[dict[str, object]],
    warnings: list[str],
) -> None:
    if "noninvasive" not in {canonical_group(group) for group in args.groups}:
        warn("Skipped topomap panels because group=noninvasive was not requested.", warnings)
        return
    if mne is None:
        warn("MNE is not available; noninvasive topomap panels were skipped.", warnings)
        return

    row_specs = [(movement, condition) for movement in args.movements for condition in args.conditions]
    col_specs = [
        (canonical_hand(hand), band)
        for hand in args.hands
        for band in TOPO_BANDS_MAIN
    ]
    for target in [canonical_target(value) for value in args.targets]:
        target_data = subset(data, target=target, group="noninvasive")
        if target_data.empty:
            warn(f"Skipped noninvasive topomap panel for {target}: no data.", warnings)
            continue
        relevant = target_data.loc[target_data["analysis_band"].isin(TOPO_BANDS_MAIN)]
        vmax = topomap_vmax(relevant["MI_z"], args)
        fig, axes = plt.subplots(
            len(row_specs),
            len(col_specs),
            figsize=(2.25 * len(col_specs), 2.1 * len(row_specs)),
            squeeze=False,
        )
        last_im = None
        for row_index, (movement, condition) in enumerate(row_specs):
            for col_index, (hand, band) in enumerate(col_specs):
                ax = axes[row_index, col_index]
                panel = subset(
                    target_data,
                    movement=movement,
                    hand=hand,
                    condition=condition,
                    group="noninvasive",
                )
                im = plot_topomap_cell(
                    ax,
                    panel,
                    band,
                    vmin=0.0,
                    vmax=vmax,
                    cmap=args.topomap_cmap,
                    warnings=warnings,
                )
                if im is not None:
                    last_im = im
                if row_index == 0:
                    hand_label = hand.replace("Hand", "")
                    ax.set_title(f"{hand_label} {band.replace('_', ' ')}", fontsize=8)
                if col_index == 0:
                    ax.text(
                        -0.20,
                        0.5,
                        f"{movement}\n{condition}",
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=9,
                    )
        fig.suptitle(f"Noninvasive EEG zMI topomaps | target = {target}", fontsize=15, y=0.995)
        fig.subplots_adjust(top=0.93, bottom=0.04, left=0.08, right=0.90, hspace=0.25, wspace=0.08)
        if last_im is not None:
            cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
            fig.colorbar(last_im, cax=cax, label="zMI")
        out_path = output_dir / f"fig1_noninvasive_topomap_panel_{target.lower()}.png"
        save_figure(fig, out_path, args, manifest, "noninvasive_topomap_panel", "noninvasive", target)


def plot_band_summary_heatmaps(
    summary: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    manifest: list[dict[str, object]],
    warnings: list[str],
) -> None:
    if summary.empty:
        warn("Skipped band summary heatmaps: no summary data.", warnings)
        return
    row_specs = [
        (movement, canonical_hand(hand), condition)
        for movement in args.movements
        for hand in args.hands
        for condition in args.conditions
    ]
    bands = list(REPORT_BANDS)
    for group in [canonical_group(value) for value in args.groups]:
        for target in [canonical_target(value) for value in args.targets]:
            panel_summary = subset(summary, group=group, target=target)
            if panel_summary.empty:
                warn(f"Skipped band summary for {group}/{target}: no data.", warnings)
                continue
            values = []
            labels = []
            for movement, hand, condition in row_specs:
                labels.append(f"{movement} {hand.replace('Hand', '')} {condition}")
                selected = panel_summary.loc[
                    panel_summary["movement"].eq(movement)
                    & panel_summary["hand"].eq(hand)
                    & panel_summary["condition"].eq(condition)
                ]
                by_band = selected.set_index("analysis_band")["mean_zMI"]
                values.append([by_band.get(band, np.nan) for band in bands])
            matrix = np.asarray(values, dtype=float)
            vmax = robust_upper(matrix, percentile=0.98, fallback=1.0)
            fig, ax = plt.subplots(figsize=(8.0, 7.4))
            image = ax.imshow(
                np.ma.masked_invalid(matrix),
                aspect="auto",
                cmap=args.heatmap_cmap,
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_xticks(range(len(bands)), [band.replace("_", " ") for band in bands], rotation=25, ha="right")
            ax.set_yticks(range(len(labels)), labels)
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = matrix[row, col]
                    text = "NA" if not np.isfinite(value) else f"{value:.2f}"
                    color = annotation_color_for_value(
                        value,
                        args.heatmap_cmap,
                        vmin=0.0,
                        vmax=vmax,
                    )
                    ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)
            fig.colorbar(image, ax=ax, label="Mean zMI")
            ax.set_title(f"Band-level mean zMI | {group} | target = {target}")
            fig.tight_layout()
            out_path = output_dir / f"fig2_band_summary_mean_zmi_{group}_{target.lower()}.png"
            save_figure(fig, out_path, args, manifest, "band_summary_mean_zmi", group, target)


def plot_rect_heatmaps(
    data: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    manifest: list[dict[str, object]],
    warnings: list[str],
    group: str,
    figure_prefix: str,
    max_frequency: float | None = None,
) -> None:
    row_specs = [(movement, condition) for movement in args.movements for condition in args.conditions]
    hands = [canonical_hand(hand) for hand in args.hands]
    for target in [canonical_target(value) for value in args.targets]:
        target_data = subset(data, group=group, target=target)
        if max_frequency is not None:
            target_data = target_data.loc[target_data["band_center"].le(max_frequency)].copy()
        if target_data.empty:
            warn(f"Skipped rectangular heatmaps for {group}/{target}: no data.", warnings)
            continue
        vmax = robust_upper(target_data["MI_z"], percentile=0.98, fallback=1.0)
        fig, axes = plt.subplots(
            len(row_specs),
            len(hands),
            figsize=(6.4 * len(hands), 2.55 * len(row_specs)),
            squeeze=False,
        )
        image = None
        for row_index, (movement, condition) in enumerate(row_specs):
            for col_index, hand in enumerate(hands):
                ax = axes[row_index, col_index]
                panel = subset(
                    target_data,
                    movement=movement,
                    hand=hand,
                    condition=condition,
                    group=group,
                )
                if panel.empty:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                pivot = panel.pivot_table(
                    index="channel",
                    columns="band_center",
                    values="MI_z",
                    aggfunc="mean",
                ).sort_index(axis=1)
                centers = pivot.columns.to_numpy(dtype=float)
                image = ax.imshow(
                    pivot.to_numpy(),
                    aspect="auto",
                    origin="upper",
                    cmap=args.heatmap_cmap,
                    vmin=0.0,
                    vmax=vmax,
                    extent=[centers.min(), centers.max(), len(pivot), 0],
                    interpolation="nearest",
                )
                ax.set_yticks(np.arange(len(pivot)) + 0.5)
                ax.set_yticklabels(pivot.index, fontsize=7)
                for boundary in (4, 8, 13, 20, 30, 60, 90):
                    if centers.min() <= boundary <= centers.max():
                        ax.axvline(boundary, color="white", linestyle="--", linewidth=0.8, alpha=0.9)
                        ax.axvline(boundary, color="black", linestyle="--", linewidth=0.45, alpha=0.6)
                ax.set_title(f"{movement} {condition} — {hand}")
                ax.set_xlabel("Frequency band center (Hz)")
                ax.set_ylabel("Channel")
        fig.suptitle(f"{group.capitalize()} channel × frequency zMI heatmaps | target = {target}", fontsize=15, y=0.995)
        fig.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.88, hspace=0.48, wspace=0.28)
        if image is not None:
            cax = fig.add_axes([0.91, 0.13, 0.018, 0.74])
            fig.colorbar(image, cax=cax, label="zMI")
        out_path = output_dir / f"{figure_prefix}_{target.lower()}.png"
        save_figure(fig, out_path, args, manifest, f"{group}_rect_heatmaps", group, target)


def plot_invasive_rect_heatmaps(
    data: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    manifest: list[dict[str, object]],
    warnings: list[str],
) -> None:
    if "invasive" not in {canonical_group(group) for group in args.groups}:
        warn("Skipped invasive rectangular heatmaps because group=invasive was not requested.", warnings)
        return
    plot_rect_heatmaps(
        data,
        args,
        output_dir,
        manifest,
        warnings,
        group="invasive",
        figure_prefix="fig3_invasive_rect_heatmaps",
        max_frequency=None,
    )


def invasive_channel_order(target_data: pd.DataFrame) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for folder in target_data["run_folder"].dropna().astype(str).drop_duplicates():
        channel_file = Path(folder) / "invasive_channels.txt"
        if not channel_file.is_file():
            continue
        try:
            channels = channel_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for channel in channels:
            channel = channel.strip()
            if channel and channel not in seen:
                order.append(channel)
                seen.add(channel)
    for channel in target_data["channel"].dropna().astype(str).drop_duplicates():
        if channel not in seen:
            order.append(channel)
            seen.add(channel)
    return order


def dot_size_from_mean(
    value: float,
    size_vmax: float,
    min_dot_size: float = 12.0,
    max_dot_size: float = 180.0,
) -> float:
    if not np.isfinite(value):
        return np.nan
    clipped = float(np.clip(value, 0.0, size_vmax))
    return min_dot_size + (max_dot_size - min_dot_size) * clipped / size_vmax


def beta_bubble_size_vmax(values: pd.Series) -> float:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 1.0
    vmax = float(finite.quantile(0.95))
    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    return vmax


def beta_bubble_legend_values(size_vmax: float) -> list[float]:
    preferred = [1.0, 2.0, 3.0]
    values = [value for value in preferred if value <= size_vmax]
    if values:
        return values
    return [size_vmax / 3.0, 2.0 * size_vmax / 3.0, size_vmax]


def side_label(channel: str) -> str:
    token = str(channel).strip().upper()
    if token.startswith("L"):
        return "L"
    if token.startswith("R"):
        return "R"
    return ""


def plot_invasive_beta_bubble_summary(
    data: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    manifest: list[dict[str, object]],
    warnings: list[str],
) -> None:
    if "invasive" not in {canonical_group(group) for group in args.groups}:
        warn("Skipped invasive beta bubble summary because group=invasive was not requested.", warnings)
        return

    beta = data.loc[
        data["group"].eq("invasive")
        & data["band_center"].ge(BETA_RANGE[0])
        & data["band_center"].lt(BETA_RANGE[1])
    ].copy()
    if beta.empty:
        warn("Skipped invasive beta bubble summary: no invasive beta data.", warnings)
        return

    grouping = [
        "patient",
        "day",
        "target_label",
        "movement",
        "hand",
        "condition",
        "channel",
    ]
    summary = (
        beta.groupby(grouping, as_index=False, dropna=False)
        .agg(
            mean_beta_zMI=("MI_z", "mean"),
            coverage_beta_zMI_gt_2=("MI_z", lambda values: float(values.gt(2).mean())),
            n_beta_bins=("MI_z", "size"),
        )
    )

    for target in [canonical_target(value) for value in args.targets]:
        target_beta = beta.loc[beta["target_label"].eq(target)].copy()
        target_summary = summary.loc[summary["target_label"].eq(target)].copy()
        if target_summary.empty:
            warn(f"Skipped invasive beta bubble summary for {target}: no data.", warnings)
            continue

        channels = invasive_channel_order(target_beta)
        if not channels:
            warn(f"Skipped invasive beta bubble summary for {target}: no channel order.", warnings)
            continue

        size_vmax = beta_bubble_size_vmax(target_summary["mean_beta_zMI"])
        fig_width = 13.2
        fig_height = max(8.0, 0.34 * len(channels) + 2.8)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        cmap = plt.get_cmap(args.heatmap_cmap)
        norm = Normalize(vmin=0.0, vmax=1.0)

        keyed = target_summary.set_index(["channel", "movement", "hand", "condition"])
        for x_index, (movement, hand, condition, _label) in enumerate(INVASIVE_BETA_BUBBLE_COLUMNS):
            for y_index, channel in enumerate(channels):
                key = (channel, movement, hand, condition)
                if key not in keyed.index:
                    continue
                row = keyed.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.mean(numeric_only=True)
                mean_value = float(row["mean_beta_zMI"])
                coverage = float(row["coverage_beta_zMI_gt_2"])
                if not np.isfinite(mean_value) or not np.isfinite(coverage):
                    continue
                ax.scatter(
                    x_index,
                    y_index,
                    s=dot_size_from_mean(mean_value, size_vmax),
                    c=[cmap(norm(np.clip(coverage, 0.0, 1.0)))],
                    edgecolors="#2b2b2b",
                    linewidths=0.3,
                )

        ax.set_xlim(-0.6, len(INVASIVE_BETA_BUBBLE_COLUMNS) - 0.4)
        ax.set_ylim(len(channels) - 0.5, -0.5)
        ax.set_xticks(
            range(len(INVASIVE_BETA_BUBBLE_COLUMNS)),
            [spec[3] for spec in INVASIVE_BETA_BUBBLE_COLUMNS],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(channels)), channels)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("Invasive channel")
        ax.set_xlabel("Movement / hand / condition")
        ax.grid(axis="y", color="0.88", linewidth=0.45)
        for boundary in (1.5, 3.5, 5.5, 7.5, 9.5):
            ax.axvline(boundary, color="0.25", linewidth=0.9, alpha=0.65)
        for index in range(1, len(channels)):
            if side_label(channels[index]) != side_label(channels[index - 1]):
                ax.axhline(index - 0.5, color="0.20", linewidth=1.0, alpha=0.75)

        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        colorbar = fig.colorbar(scalar_map, ax=ax, fraction=0.025, pad=0.02)
        colorbar.set_label("Beta coverage (fraction of bins with zMI > 2)")

        legend_values = beta_bubble_legend_values(size_vmax)
        legend_handles = [
            ax.scatter(
                [],
                [],
                s=dot_size_from_mean(value, size_vmax),
                facecolors="none",
                edgecolors="#333333",
                linewidths=0.8,
            )
            for value in legend_values
        ]
        legend_labels = [f"{value:.2g}" for value in legend_values]
        ax.legend(
            legend_handles,
            legend_labels,
            title="Mean beta zMI",
            loc="upper left",
            bbox_to_anchor=(1.03, 0.98),
            frameon=False,
            borderaxespad=0.0,
        )

        fig.suptitle(f"Invasive beta zMI bubble summary | target = {target}", fontsize=15, y=0.992)
        fig.text(
            0.5,
            0.958,
            "Dot size = mean beta zMI (13-30 Hz); color = fraction of beta bins with zMI > 2.",
            ha="center",
            va="top",
            fontsize=10,
        )
        fig.text(
            0.5,
            0.025,
            "coverage = fraction of beta sliding bands with zMI > 2; this is a descriptive surrogate-normalized threshold, not a corrected significance test.",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        fig.subplots_adjust(top=0.91, bottom=0.16, left=0.12, right=0.82)

        out_path = output_dir / f"fig4_invasive_beta_bubble_summary_{target.lower()}.png"
        save_figure(
            fig,
            out_path,
            args,
            manifest,
            "invasive_beta_bubble_summary",
            "invasive",
            target,
        )


def plot_top_beta_channels_prepost(
    channels: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    manifest: list[dict[str, object]],
    warnings: list[str],
) -> None:
    if channels.empty:
        warn("Skipped top beta channel figures: no beta channel summary data.", warnings)
        return
    top_n_beta_channels = getattr(args, "top_n_beta_channels", 5)
    groups = [canonical_group(value) for value in args.groups]
    hands = [canonical_hand(value) for value in args.hands]
    pre_col = "mean_beta_zMI_pre"
    post_col = "mean_beta_zMI_post"
    for group in groups:
        for target in [canonical_target(value) for value in args.targets]:
            target_channels = subset(channels, group=group, target=target)
            if target_channels.empty:
                warn(f"Skipped top beta channels for {group}/{target}: no data.", warnings)
                continue
            fig, axes = plt.subplots(
                len(args.movements),
                len(hands),
                figsize=(6.2 * len(hands), 3.3 * len(args.movements)),
                squeeze=False,
            )
            for row_index, movement in enumerate(args.movements):
                for col_index, hand in enumerate(hands):
                    ax = axes[row_index, col_index]
                    panel = subset(target_channels, movement=movement, hand=hand, group=group)
                    panel = panel.loc[panel["mean_beta_zMI_avg"].notna()].copy()
                    if panel.empty:
                        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                        ax.set_axis_off()
                        continue
                    top = panel.sort_values("mean_beta_zMI_avg", ascending=False).head(
                        top_n_beta_channels
                    )
                    top = top.iloc[::-1]
                    y = np.arange(len(top), dtype=float)
                    height = 0.34
                    ax.barh(
                        y - height / 2,
                        top[pre_col],
                        height=height,
                        color=CONDITION_COLORS.get("PreNFB"),
                        label="PreNFB",
                    )
                    ax.barh(
                        y + height / 2,
                        top[post_col],
                        height=height,
                        color=CONDITION_COLORS.get("PostNFB"),
                        label="PostNFB",
                    )
                    for index, row in enumerate(top.itertuples(index=False)):
                        if pd.isna(getattr(row, pre_col)):
                            ax.text(0.02, y[index] - height / 2, "Pre missing", fontsize=7, va="center")
                        if pd.isna(getattr(row, post_col)):
                            ax.text(0.02, y[index] + height / 2, "Post missing", fontsize=7, va="center")
                    ax.set_yticks(y, top["channel"].tolist())
                    ax.set_xlabel("Mean beta zMI")
                    ax.set_title(f"{movement} — {hand}")
                    ax.grid(axis="x", alpha=0.25)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.suptitle(f"Top {top_n_beta_channels} beta channels | {group} | target = {target}", fontsize=15, y=0.995)
            if handles:
                unique = dict(zip(labels, handles))
                fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2)
                rect = (0, 0, 1, 0.90)
            else:
                rect = (0, 0, 1, 0.94)
            fig.tight_layout(rect=rect)
            out_path = output_dir / f"fig4_top{top_n_beta_channels}_beta_channels_{group}_{target.lower()}.png"
            save_figure(fig, out_path, args, manifest, "top_beta_channels_prepost", group, target)


def save_report_readme(
    output_dir: Path,
    args: argparse.Namespace,
    runs: pd.DataFrame,
    expected: pd.DataFrame,
    manifest: list[dict[str, object]],
    warnings: list[str],
) -> None:
    missing = expected.loc[expected["status"].eq("MISSING")] if not expected.empty else pd.DataFrame()
    lines = [
        "# zMI report",
        "",
        "This report shows absolute surrogate-normalized mutual information values (`zMI`) separately for PreNFB and PostNFB.",
        "",
        "No Post-minus-Pre zMI contrast figures are generated in this report. Simple subtraction of zMI values is descriptive only and is not treated here as a formal Pre/Post statistical contrast.",
        "",
        "The invasive beta bubble summary is a descriptive beta 13-30 Hz overview without Pre/Post delta.",
        "",
        "The bubble summary does not select top channels: it shows all invasive channels found in the source invasive channel lists.",
        "",
        "For the bubble summary, dot size is mean beta zMI and dot color is beta coverage, defined as the fraction of beta sliding bands with zMI > 2. This coverage threshold is descriptive and surrogate-normalized; it is not a corrected significance test.",
        "",
        "## Inputs",
        "",
        f"- results root: `{args.results_root}`",
        f"- patient: `{args.patient}`",
        f"- day: `{args.day}`",
        f"- movements: `{', '.join(args.movements)}`",
        f"- hands: `{', '.join(args.hands)}`",
        f"- targets: `{', '.join(args.targets)}`",
        f"- conditions: `{', '.join(args.conditions)}`",
        f"- groups: `{', '.join(args.groups)}`",
        f"- n_surrogates: `{args.n_surrogates}`",
        "",
        "## Loaded data",
        "",
        f"- loaded group-level run summaries: `{len(runs)}`",
        f"- missing requested group-level runs: `{len(missing)}`",
        "",
        "## Figures",
        "",
        "- `fig1_noninvasive_topomap_panel_*` - noninvasive EEG topomap panels by movement, condition, hand, and canonical bands.",
        "- `fig2_band_summary_mean_zmi_*` - band-level mean zMI heatmaps with contrast-aware cell labels.",
        "- `fig3_invasive_rect_heatmaps_*` - invasive channel x frequency zMI heatmaps.",
        "- `fig4_invasive_beta_bubble_summary_*` - all-channel invasive beta 13-30 Hz bubble maps; no top-channel selection and no Pre/Post delta.",
        "",
        "## Generated Figures",
        "",
    ]
    for item in manifest:
        lines.append(f"- `{item['filename']}` — {item['figure_type']} ({item['group']}, {item['target_label']})")
    lines.extend(
        [
            "",
            "## Missing runs",
            "",
        ]
    )
    if missing.empty:
        lines.append("No requested group-level runs were missing.")
    else:
        for row in missing.itertuples(index=False):
            lines.append(
                f"- {row.movement} / {row.hand} / {row.target_label} / {row.condition} / {row.group}"
            )
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for message in warnings:
            lines.append(f"- {message}")
    else:
        lines.append("No warnings.")
    (output_dir / "report_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.results_root = args.results_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.hands = [canonical_hand(value) for value in args.hands]
    args.targets = [canonical_target(value) for value in args.targets]
    args.groups = [canonical_group(value) for value in args.groups]
    warnings: list[str] = []
    manifest: list[dict[str, object]] = []

    if not args.results_root.is_dir():
        print(f"ERROR: Results root does not exist: {args.results_root}", file=sys.stderr)
        return 2

    output_dir = unique_output_dir(args.output_root)
    summary_files = sorted(args.results_root.rglob("summary.csv"))
    print(f"Run folders containing summary.csv found: {len(summary_files)}")
    data, runs = load_matching_summaries(summary_files, args, warnings)
    print(f"Group-level run summaries matching filters: {len(runs)}")
    expected = report_expected_runs(args, runs, warnings)

    runs.to_csv(output_dir / "loaded_run_summaries.csv", index=False)
    expected.to_csv(output_dir / "expected_run_status.csv", index=False)

    if data.empty:
        warn("No matching usable zMI data; no figures were generated.", warnings)
        pd.DataFrame(manifest).to_csv(output_dir / "figure_manifest.csv", index=False)
        save_report_readme(output_dir, args, runs, expected, manifest, warnings)
        print(f"Outputs saved to: {output_dir}")
        return 0

    long_columns = METADATA_COLUMNS + [
        "group",
        "channel",
        "band_fmin",
        "band_fmax",
        "band_center",
        "analysis_band",
        "MI_z",
    ]
    data[long_columns].to_csv(output_dir / "matched_input_zmi_long.csv", index=False)
    band_summary = condition_band_summary(data)
    beta_summary = beta_channel_summary(data, args)
    band_summary.to_csv(output_dir / "band_condition_summary.csv", index=False)
    beta_summary.to_csv(output_dir / "beta_channel_summary.csv", index=False)

    plot_noninvasive_topomap_panel(data, args, output_dir, manifest, warnings)
    plot_band_summary_heatmaps(band_summary, args, output_dir, manifest, warnings)
    plot_invasive_rect_heatmaps(data, args, output_dir, manifest, warnings)
    plot_invasive_beta_bubble_summary(data, args, output_dir, manifest, warnings)
    if args.plot_noninvasive_rect_heatmaps:
        plot_rect_heatmaps(
            data,
            args,
            output_dir,
            manifest,
            warnings,
            group="noninvasive",
            figure_prefix="fig5_noninvasive_rect_heatmaps",
            max_frequency=30.0,
        )

    figure_manifest = pd.DataFrame(manifest)
    figure_manifest.to_csv(output_dir / "figure_manifest.csv", index=False)
    save_report_readme(output_dir, args, runs, expected, manifest, warnings)

    print("Generated figures:")
    for item in manifest:
        print(f"  {item['filename']}")
    print("Generated CSV/MD files:")
    for name in [
        "loaded_run_summaries.csv",
        "expected_run_status.csv",
        "matched_input_zmi_long.csv",
        "band_condition_summary.csv",
        "beta_channel_summary.csv",
        "figure_manifest.csv",
        "report_readme.md",
    ]:
        print(f"  {name}")
    print(f"Outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
