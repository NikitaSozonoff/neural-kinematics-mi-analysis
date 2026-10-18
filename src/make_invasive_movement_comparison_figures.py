"""Build invasive-channel comparison tables and figures from MI surrogate summaries.

This helper is intentionally read-only with respect to existing result runs.  Every
invocation writes to a new timestamped directory beneath ``--results-root``.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = Path("outputs/MI_surrogates")
REQUIRED_COLUMNS = {
    "group",
    "channel",
    "band_fmin",
    "band_fmax",
    "MI_z",
}
BAND_RANGES = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "beta": (13.0, 30.0),
}
RUN_METADATA_COLUMNS = [
    "run_id",
    "run_folder",
    "patient",
    "day",
    "movement",
    "hand",
    "condition",
    "target",
    "n_surrogates",
]
CONDITION_COLORS = {"PreNFB": "#2878b5", "PostNFB": "#d1495b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare invasive MI surrogate summaries across movements."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--patient", default="Patient1")
    parser.add_argument("--day", default="Day2")
    parser.add_argument("--target", default="MoCap_Vel")
    parser.add_argument("--movements", nargs="+", default=["Clenching", "Rotation"])
    parser.add_argument("--hands", nargs="+", default=["LeftHand", "RightHand"])
    parser.add_argument("--conditions", nargs="+", default=["PreNFB", "PostNFB"])
    parser.add_argument("--n-surrogates", type=int, default=200)
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def normalized_target(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def target_matches(found: str, requested: str) -> bool:
    found_norm = normalized_target(found)
    requested_norm = normalized_target(requested)
    velocity_aliases = {"mocapvel", "velocity", "vel"}
    if requested_norm in velocity_aliases:
        return found_norm in velocity_aliases
    return found_norm == requested_norm


def parse_run_name(name: str) -> dict[str, object] | None:
    prefix = re.match(
        r"^(?P<patient>[^_]+)_(?P<day>[^_]+)_"
        r"(?P<condition>PreNFB|PostNFB)_"
        r"(?P<hand>LeftHand|RightHand)_"
        r"(?P<movement>Clenching|Rotation|Tapping)(?:__|_)",
        name,
        flags=re.IGNORECASE,
    )
    target = re.search(r"(?:^|__)target-(?P<target>.+?)(?:__|$)", name)
    n_surrogates = re.search(r"(?:^|__)N(?P<n>\d+)(?:__|$)", name, re.IGNORECASE)
    if not (prefix and target and n_surrogates):
        return None
    metadata: dict[str, object] = prefix.groupdict()
    metadata["target"] = target.group("target")
    metadata["n_surrogates"] = int(n_surrogates.group("n"))
    return metadata


def unique_output_dir(results_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = results_root / f"_invasive_movement_comparison_{stamp}"
    candidate = base
    counter = 2
    while candidate.exists():
        candidate = Path(f"{base}_{counter}")
        counter += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def assign_band_name(center: pd.Series) -> pd.Series:
    choices = ["delta", "theta", "alpha", "low_beta", "high_beta"]
    conditions = [
        center.ge(1) & center.lt(4),
        center.ge(4) & center.lt(8),
        center.ge(8) & center.lt(13),
        center.ge(13) & center.lt(20),
        center.ge(20) & center.lt(30),
    ]
    return pd.Series(np.select(conditions, choices, default="other"), index=center.index)


def load_matching_runs(
    summary_files: list[Path], args: argparse.Namespace
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    matched_metadata: list[dict[str, object]] = []

    for summary_path in summary_files:
        metadata = parse_run_name(summary_path.parent.name)
        if metadata is None:
            warn(f"Could not parse run folder name; skipped: {summary_path.parent.name}")
            continue
        if not (
            str(metadata["patient"]).casefold() == args.patient.casefold()
            and str(metadata["day"]).casefold() == args.day.casefold()
            and str(metadata["movement"]).casefold()
            in {value.casefold() for value in args.movements}
            and str(metadata["hand"]).casefold()
            in {value.casefold() for value in args.hands}
            and str(metadata["condition"]).casefold()
            in {value.casefold() for value in args.conditions}
            and target_matches(str(metadata["target"]), args.target)
            and metadata["n_surrogates"] == args.n_surrogates
        ):
            continue

        try:
            frame = pd.read_csv(summary_path)
        except Exception as exc:  # keep processing independent runs
            warn(f"Could not read {summary_path}: {exc}")
            continue
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            warn(
                f"Skipped {summary_path}: missing columns "
                f"{', '.join(sorted(missing_columns))}"
            )
            continue

        frame = frame.loc[frame["group"].astype(str).str.casefold() == "invasive"].copy()
        if frame.empty:
            warn(f"No invasive rows in {summary_path}")
            continue
        for column in ("band_fmin", "band_fmax", "MI_z"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[["band_fmin", "band_fmax", "MI_z"]].isna().any(axis=1)
        if invalid.any():
            warn(f"Dropped {int(invalid.sum())} non-numeric rows from {summary_path}")
            frame = frame.loc[~invalid].copy()
        if frame.empty:
            warn(f"No usable invasive rows in {summary_path}")
            continue

        frame["band_center"] = (frame["band_fmin"] + frame["band_fmax"]) / 2.0
        frame["band_name"] = assign_band_name(frame["band_center"])
        run_id = summary_path.parent.name
        metadata = {
            "run_id": run_id,
            "run_folder": str(summary_path.parent.resolve()),
            **metadata,
        }
        for key, value in metadata.items():
            frame[key] = value
        frames.append(frame)
        matched_metadata.append(metadata)

    if not frames:
        return pd.DataFrame(), matched_metadata
    return pd.concat(frames, ignore_index=True), matched_metadata


def describe_expected(args: argparse.Namespace, metadata: list[dict[str, object]]) -> None:
    expected = list(product(args.movements, args.hands, args.conditions))
    found = {
        (str(item["movement"]), str(item["hand"]), str(item["condition"]))
        for item in metadata
    }
    print("Expected movement/hand/condition combinations:")
    for combination in expected:
        label = " / ".join(combination)
        if combination in found:
            print(f"  FOUND   {label}")
        else:
            print(f"  MISSING {label}")
            warn(f"Expected run is missing: {label}")


def percentile95(values: pd.Series) -> float:
    return float(values.quantile(0.95)) if not values.empty else np.nan


def band_values(frame: pd.DataFrame, band: str) -> pd.Series:
    low, high = BAND_RANGES[band]
    return frame.loc[frame["band_center"].ge(low) & frame["band_center"].lt(high), "MI_z"]


def run_level_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, frame in data.groupby("run_id", sort=False):
        z = frame["MI_z"]
        top = frame.loc[z.idxmax()]
        row = {key: frame.iloc[0][key] for key in RUN_METADATA_COLUMNS}
        row.update(
            {
                "mean_zMI": z.mean(),
                "median_zMI": z.median(),
                "std_zMI": z.std(ddof=1),
                "max_zMI": z.max(),
                "p95_zMI": z.quantile(0.95),
                "p99_zMI": z.quantile(0.99),
                "n_bins_zMI_gt_2": int(z.gt(2).sum()),
                "fraction_bins_zMI_gt_2": z.gt(2).mean(),
                "n_bins_zMI_gt_3": int(z.gt(3).sum()),
                "fraction_bins_zMI_gt_3": z.gt(3).mean(),
                "top_channel": top["channel"],
                "top_band_fmin": top["band_fmin"],
                "top_band_fmax": top["band_fmax"],
                "top_zMI": top["MI_z"],
            }
        )
        for band in ("theta", "alpha", "beta", "low_beta", "high_beta"):
            values = band_values(frame, band)
            row[f"mean_{band}_zMI"] = values.mean()
            row[f"p95_{band}_zMI"] = percentile95(values)
            row[f"fraction_{band}_zMI_gt_2"] = values.gt(2).mean() if len(values) else np.nan
        beta = band_values(frame, "beta")
        row["max_beta_zMI"] = beta.max() if len(beta) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def frequency_profile(data: pd.DataFrame) -> pd.DataFrame:
    grouping = RUN_METADATA_COLUMNS + ["band_center"]
    return (
        data.groupby(grouping, as_index=False, dropna=False)["MI_z"]
        .agg(
            mean_zMI="mean",
            p95_zMI=lambda values: values.quantile(0.95),
            max_zMI="max",
            fraction_channels_zMI_gt_2=lambda values: values.gt(2).mean(),
        )
        .sort_values(["movement", "hand", "condition", "band_center"])
    )


def channel_summary(data: pd.DataFrame) -> pd.DataFrame:
    grouping = RUN_METADATA_COLUMNS + ["channel"]
    overall = (
        data.groupby(grouping, as_index=False, dropna=False)["MI_z"]
        .agg(
            mean_zMI="mean",
            p95_zMI=lambda values: values.quantile(0.95),
            max_zMI="max",
            fraction_zMI_gt_2=lambda values: values.gt(2).mean(),
        )
    )
    beta_data = data.loc[data["band_center"].ge(13) & data["band_center"].lt(30)]
    beta = (
        beta_data.groupby(grouping, as_index=False, dropna=False)["MI_z"]
        .agg(
            mean_beta_zMI="mean",
            p95_beta_zMI=lambda values: values.quantile(0.95),
            fraction_beta_zMI_gt_2=lambda values: values.gt(2).mean(),
        )
    )
    return overall.merge(beta, on=grouping, how="left").sort_values(
        ["movement", "hand", "condition", "channel"]
    )


def subset_for_panel(
    data: pd.DataFrame, movement: str, hand: str, condition: str | None = None
) -> pd.DataFrame:
    mask = data["movement"].eq(movement) & data["hand"].eq(hand)
    if condition is not None:
        mask &= data["condition"].eq(condition)
    return data.loc[mask]


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(data: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    row_specs = list(product(args.movements, args.hands))
    fig, axes = plt.subplots(
        len(row_specs), len(args.conditions), figsize=(14, 3.2 * len(row_specs)), squeeze=False
    )
    finite = data["MI_z"].replace([np.inf, -np.inf], np.nan).dropna()
    limit = max(float(finite.abs().max()), 1e-9)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = None
    for row_index, (movement, hand) in enumerate(row_specs):
        for column_index, condition in enumerate(args.conditions):
            ax = axes[row_index, column_index]
            panel = subset_for_panel(data, movement, hand, condition)
            if panel.empty:
                ax.text(0.5, 0.5, "Missing run", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            pivot = panel.pivot_table(
                index="channel", columns="band_center", values="MI_z", aggfunc="mean"
            ).sort_index(axis=1)
            centers = pivot.columns.to_numpy(dtype=float)
            image = ax.imshow(
                pivot.to_numpy(), aspect="auto", origin="upper", cmap="RdBu_r", norm=norm,
                extent=[centers.min(), centers.max(), len(pivot), 0], interpolation="nearest"
            )
            ax.set_yticks(np.arange(len(pivot)) + 0.5)
            ax.set_yticklabels(pivot.index, fontsize=7)
            for boundary in (13, 20, 30):
                ax.axvline(boundary, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(f"{movement} {hand} — {condition}")
            ax.set_xlabel("Frequency band center (Hz)")
            ax.set_ylabel("Invasive channel")
    fig.suptitle("Invasive MI heatmaps: velocity", fontsize=15)
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.07, right=0.87, hspace=0.42, wspace=0.32)
    if image is not None:
        colorbar_axis = fig.add_axes([0.90, 0.12, 0.018, 0.76])
        fig.colorbar(image, cax=colorbar_axis, label="zMI")
    save_figure(fig, output_dir / "fig1_invasive_heatmaps_clenching_rotation_velocity.png")


def plot_spectral_profiles(
    profile: pd.DataFrame, args: argparse.Namespace, output_dir: Path
) -> None:
    fig, axes = plt.subplots(
        len(args.movements), len(args.hands), figsize=(13, 8), squeeze=False, sharey=True
    )
    y_values = profile[["mean_zMI", "p95_zMI"]].to_numpy(dtype=float)
    finite = y_values[np.isfinite(y_values)]
    y_limits = None
    if finite.size:
        margin = max((finite.max() - finite.min()) * 0.08, 0.1)
        y_limits = (finite.min() - margin, finite.max() + margin)
    for row_index, movement in enumerate(args.movements):
        for column_index, hand in enumerate(args.hands):
            ax = axes[row_index, column_index]
            ax.axvspan(13, 20, color="#f2c14e", alpha=0.14, label="Low beta")
            ax.axvspan(20, 30, color="#f78154", alpha=0.12, label="High beta")
            panel = subset_for_panel(profile, movement, hand)
            if panel.empty:
                ax.text(0.5, 0.5, "Missing run", ha="center", va="center", transform=ax.transAxes)
            for condition in args.conditions:
                condition_data = panel.loc[panel["condition"].eq(condition)]
                if condition_data.empty:
                    continue
                aggregated = condition_data.groupby("band_center", as_index=False).agg(
                    mean_zMI=("mean_zMI", "mean"), p95_zMI=("p95_zMI", "mean")
                )
                color = CONDITION_COLORS.get(condition)
                ax.plot(
                    aggregated["band_center"], aggregated["mean_zMI"], marker="o",
                    markersize=2.5, linewidth=1.8, color=color, label=f"{condition} mean"
                )
                ax.plot(
                    aggregated["band_center"], aggregated["p95_zMI"], linestyle="--",
                    linewidth=1.2, color=color, alpha=0.8, label=f"{condition} p95"
                )
            ax.axhline(0, color="0.3", linewidth=0.7)
            ax.set_title(f"{movement} — {hand}")
            ax.set_xlabel("Frequency band center (Hz)")
            ax.set_ylabel("zMI across invasive channels")
            ax.grid(alpha=0.2)
            if y_limits:
                ax.set_ylim(*y_limits)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.suptitle("Invasive spectral profiles: velocity", fontsize=15, y=0.995)
    fig.legend(
        unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=4, fontsize=9
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, output_dir / "fig2_invasive_spectral_profiles_clenching_rotation_velocity.png")


def plot_band_summary(run_summary: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    band_columns = ["theta", "alpha", "low_beta", "high_beta", "beta"]
    rows: list[list[float]] = []
    labels: list[str] = []
    for movement, hand, condition in product(args.movements, args.hands, args.conditions):
        selected = run_summary.loc[
            run_summary["movement"].eq(movement)
            & run_summary["hand"].eq(hand)
            & run_summary["condition"].eq(condition)
        ]
        labels.append(f"{movement} {hand.replace('Hand', '')} {condition.replace('NFB', '')}")
        rows.append(
            [selected[f"mean_{band}_zMI"].mean() if not selected.empty else np.nan for band in band_columns]
        )
    values = np.asarray(rows, dtype=float)
    finite = values[np.isfinite(values)]
    limit = max(float(np.abs(finite).max()), 1e-9) if finite.size else 1.0
    masked = np.ma.masked_invalid(values)
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(masked, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(range(len(band_columns)), labels=[name.replace("_", " ") for name in band_columns])
    ax.set_yticks(range(len(labels)), labels=labels)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                color = "white" if abs(values[row, column]) > limit * 0.55 else "black"
                ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", color=color)
            else:
                ax.text(column, row, "NA", ha="center", va="center", color="0.4")
    fig.colorbar(image, ax=ax, label="Mean zMI")
    ax.set_title("Invasive band-level mean zMI")
    fig.tight_layout()
    save_figure(fig, output_dir / "fig3_invasive_band_mean_summary.png")


def plot_beta_metrics(run_summary: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    metrics = [
        ("p95_beta_zMI", "Beta p95 zMI"),
        ("fraction_beta_zMI_gt_2", "Fraction beta zMI > 2"),
        ("max_beta_zMI", "Maximum beta zMI"),
    ]
    categories = list(product(args.movements, args.hands))
    x = np.arange(len(categories), dtype=float)
    offsets = np.linspace(-0.16, 0.16, max(len(args.conditions), 1))
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5.2), squeeze=False)
    for metric_index, (metric, title) in enumerate(metrics):
        ax = axes[0, metric_index]
        for condition_index, condition in enumerate(args.conditions):
            values = []
            for movement, hand in categories:
                selected = run_summary.loc[
                    run_summary["movement"].eq(movement)
                    & run_summary["hand"].eq(hand)
                    & run_summary["condition"].eq(condition), metric
                ]
                values.append(selected.mean() if not selected.empty else np.nan)
            ax.scatter(
                x + offsets[condition_index], values, s=65,
                color=CONDITION_COLORS.get(condition), label=condition, zorder=3
            )
            ax.plot(x + offsets[condition_index], values, color=CONDITION_COLORS.get(condition), alpha=0.35)
        ax.set_xticks(x, [f"{m}\n{h.replace('Hand', '')}" for m, h in categories])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if metric == "fraction_beta_zMI_gt_2":
            ax.set_ylim(0, 1)
        if metric_index == 0:
            ax.legend()
    fig.suptitle("Invasive beta extent and strength", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir / "fig4_invasive_beta_metrics_summary.png")


def plot_top_channels(
    channels: pd.DataFrame, args: argparse.Namespace, output_dir: Path, top_n: int = 10
) -> None:
    categories = list(product(args.movements, args.hands))
    fig, axes = plt.subplots(
        len(args.movements), len(args.hands), figsize=(14, 10), squeeze=False
    )
    for index, (movement, hand) in enumerate(categories):
        row_index = index // len(args.hands)
        column_index = index % len(args.hands)
        ax = axes[row_index, column_index]
        panel = subset_for_panel(channels, movement, hand)
        if panel.empty:
            ax.text(0.5, 0.5, "Missing run", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ranking = (
            panel.groupby("channel")["p95_beta_zMI"].max().sort_values(ascending=False).head(top_n)
        )
        ordered_channels = ranking.index.tolist()[::-1]
        y = np.arange(len(ordered_channels), dtype=float)
        offsets = np.linspace(-0.14, 0.14, max(len(args.conditions), 1))
        for condition_index, condition in enumerate(args.conditions):
            condition_values = (
                panel.loc[panel["condition"].eq(condition)]
                .groupby("channel")["p95_beta_zMI"].mean()
                .reindex(ordered_channels)
            )
            ax.scatter(
                condition_values, y + offsets[condition_index], s=48,
                color=CONDITION_COLORS.get(condition), label=condition
            )
        ax.axvline(2, color="0.35", linestyle="--", linewidth=1)
        ax.set_yticks(y, ordered_channels)
        ax.set_xlabel("Beta p95 zMI")
        ax.set_title(f"{movement} — {hand}")
        ax.grid(axis="x", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Top {top_n} invasive channels by beta p95", fontsize=15, y=0.995)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96),
        ncol=max(len(labels), 1)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output_dir / "fig5_invasive_top_channels_beta_p95.png")


def plot_distributions(data: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    row_specs = list(product(args.movements, args.hands))
    fig, axes = plt.subplots(
        len(row_specs), len(args.conditions), figsize=(13, 3 * len(row_specs)), squeeze=False,
        sharex=True, sharey=True
    )
    for row_index, (movement, hand) in enumerate(row_specs):
        for column_index, condition in enumerate(args.conditions):
            ax = axes[row_index, column_index]
            values = subset_for_panel(data, movement, hand, condition)["MI_z"].dropna()
            if values.empty:
                ax.text(0.5, 0.5, "Missing run", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.hist(values, bins=45, density=True, alpha=0.75, color=CONDITION_COLORS.get(condition))
            ax.axvline(2, color="#e07a1f", linestyle="--", linewidth=1.2, label="zMI = 2")
            ax.axvline(3, color="#a11d33", linestyle=":", linewidth=1.4, label="zMI = 3")
            ax.set_title(f"{movement} {hand} — {condition}")
            ax.set_xlabel("zMI")
            ax.set_ylabel("Density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Invasive zMI distributions", fontsize=15, y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, output_dir / "fig6_invasive_zMI_distributions.png")


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    if not results_root.is_dir():
        print(f"ERROR: Results root does not exist: {results_root}", file=sys.stderr)
        return 2

    # Output directories also contain CSV files, so only accept a summary whose
    # immediate parent parses as an original run folder.
    summary_files = sorted(results_root.rglob("summary.csv"))
    print(f"Run folders containing summary.csv found: {len(summary_files)}")
    data, metadata = load_matching_runs(summary_files, args)
    print(f"Run folders matching filters with usable invasive data: {len(metadata)}")
    describe_expected(args, metadata)

    output_dir = unique_output_dir(results_root)
    if data.empty:
        warn("No matching usable data; no tables or figures were generated.")
        print(f"Outputs saved to: {output_dir}")
        return 0

    runs = run_level_summary(data)
    profile = frequency_profile(data)
    channels = channel_summary(data)
    runs.to_csv(output_dir / "invasive_run_level_summary.csv", index=False)
    profile.to_csv(output_dir / "invasive_frequency_profile.csv", index=False)
    channels.to_csv(output_dir / "invasive_channel_summary.csv", index=False)

    plot_heatmaps(data, args, output_dir)
    plot_spectral_profiles(profile, args, output_dir)
    plot_band_summary(runs, args, output_dir)
    plot_beta_metrics(runs, args, output_dir)
    plot_top_channels(channels, args, output_dir)
    plot_distributions(data, args, output_dir)

    print(f"Generated 3 CSV tables and 6 figures.")
    print(f"Outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
