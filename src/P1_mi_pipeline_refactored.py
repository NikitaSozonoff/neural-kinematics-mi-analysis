import argparse
import fnmatch
import re
import json
import csv
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")  # важно для Windows + joblib, не использовать Tkinter backend
import matplotlib.pyplot as plt

import mne
from scipy.signal import hilbert
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # ---------- paths ----------
    "file_path": "data/example_recording.fif",
    "out_root": "outputs/MI_surrogates",
    # Если хочешь использовать конкретную уже существующую папку для plot_only,
    # вставь сюда полный путь. Если пусто, папка будет собрана автоматически по параметрам.
    "existing_run_dir": "",

    # ---------- run mode ----------
    # "compute"   = считать MI заново
    # "plot_only" = не считать MI, а только строить графики из уже сохранённых .npy
    "run_mode": "plot_only",

    # Что делать, если папка запуска уже существует:
    # "ask"       = спросить в терминале
    # "overwrite" = удалить папку и считать заново
    # "append"    = использовать существующую папку, ничего не удалять
    # "copy"      = создать копию папки с timestamp
    # "cancel"    = остановить запуск
    "existing_run_policy": "ask",

    # Что делать, если внутри папки уже есть файл, который код хочет сохранить:
    # "ask", "overwrite", "copy", "skip", "cancel"
    "file_exists_policy": "ask",

    # ---------- target ----------
    # "Angle" или "Velocity"
    "target_channel": "Angle",

    # ---------- frequency sliding windows ----------
    "band_width": 3,
    "band_step": 1,

    "inv_fmin": 1,
    "inv_fmax": 100,

    "noninv_fmin": 1,
    "noninv_fmax": 30,

    # ---------- surrogates ----------
    "n_surrogates": 200,
    "random_state": 42,

    # ---------- speed / sampling ----------
    "use_resample": True,
    "resample_sfreq": 250,

    # ---------- groups ----------
    "run_invasive": False,
    "run_noninvasive": True,

    # ---------- parallel ----------
    "n_jobs": 8,

    # ---------- plotting ----------
    # Старый прямоугольный heatmap channel × frequency
    "plot_heatmaps": False,

    # Новый вариант: topomaps только для noninvasive EEG
    "plot_topomaps": True,

    # Можно указать несколько наборов, и код построит несколько картинок
    # например ["canonical", "parkinson_beta", "extended"]
    "topomap_band_group_sets": ["canonical", "parkinson_beta"],

    # ---------- topomap style ----------
    "topomap_cmap": "RdBu_r",
}


TARGET_ALIASES = {
    "velocity": ["MoCap_Vel", "Velocity"],
    "angle": ["MoCap_Ang", "Angle"],
    "acceleration": ["MoCap_Acc", "Acceleration"],
}


# ============================================================
# CHANNELS
# ============================================================

INVASIVE_CHANNELS = [
    "L8-L5", "L8-L6", "L8-L7", "L7-L5", "L6-L5", "L7-L6",
    "L6-L3", "L7-L4", "L5-L2", "L4-L2", "L3-L2", "L4-L3",
    "R8-R5", "R8-R6", "R8-R7", "R7-R5", "R6-R5", "R7-R6",
    "R6-R3", "R7-R4", "R5-R2", "R4-R2", "R3-R2", "R4-R3",
    "R1-R2", "R1-R3", "R1-R4",
]

NONINVASIVE_CHANNELS = [
    "Fp1", "Fp2", "F7", "F8", "T3", "T4", "T5", "P3", "Pz", "P4", "O1", "O2",
]


# ============================================================
# BAND GROUPS FOR TOPOGRAPHIES
# ============================================================

BAND_GROUP_SETS = {
    "canonical": {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    },

    # Вариант, более удобный для Parkinson / beta-related интерпретации
    "parkinson_beta": {
        "theta": (4, 8),
        "alpha": (8, 13),
        "low_beta": (13, 20),
        "high_beta": (20, 35),
    },

    # Работает только если noninvasive_fmax достаточно высокий.
    # При noninv_fmax=30 gamma просто будет пропущена.
    "extended": {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
        "low_gamma": (30, 60),
        "high_gamma": (60, 90),
    },
}


# ============================================================
# SMALL UTILITIES
# ============================================================

def now_string():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def sanitize_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")
    return text


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def build_run_id(config):
    file_stem = sanitize_name(Path(config["file_path"]).stem)
    target_for_id = (
        config.get("target_resolved")
        or config.get("target_requested")
        or config["target_channel"]
    )

    if config["use_resample"]:
        fs_part = f"fs{config['resample_sfreq']}"
    else:
        fs_part = "fsOriginal"

    run_id = (
        f"{file_stem}"
        f"__target-{sanitize_name(target_for_id)}"
        f"__{fs_part}"
        f"__w{config['band_width']}_s{config['band_step']}"
        f"__inv{config['inv_fmin']}-{config['inv_fmax']}"
        f"__eeg{config['noninv_fmin']}-{config['noninv_fmax']}"
        f"__N{config['n_surrogates']}"
        f"__seed{config['random_state']}"
    )

    return run_id


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run MI/surrogate analysis for one or more FIF files. "
            "CLI values override the top-level CONFIG defaults."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "FIF files or directories. Directories are searched recursively. "
            "If omitted, CONFIG['file_path'] is used."
        ),
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Optional glob applied to file name/full path; may be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Optional exclusion glob; may be repeated.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Targets: velocity, angle, acceleration, or exact channel names.",
    )
    parser.add_argument(
        "--run-mode",
        choices=["compute", "plot_only"],
        default=None,
    )
    parser.add_argument("--out-root", default=None)
    parser.add_argument(
        "--groups",
        choices=["invasive", "noninvasive", "both"],
        default=None,
    )
    parser.add_argument("--n-surrogates", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--resample-sfreq", type=float, default=None)
    parser.add_argument("--band-width", type=float, default=None)
    parser.add_argument("--band-step", type=float, default=None)
    parser.add_argument("--inv-fmin", type=float, default=None)
    parser.add_argument("--inv-fmax", type=float, default=None)
    parser.add_argument("--noninv-fmin", type=float, default=None)
    parser.add_argument("--noninv-fmax", type=float, default=None)
    parser.add_argument(
        "--plot-heatmaps",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--plot-topomaps",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--topomap-band-group-sets",
        nargs="+",
        choices=sorted(BAND_GROUP_SETS),
        default=None,
    )
    parser.add_argument(
        "--existing-run-policy",
        choices=["ask", "overwrite", "append", "copy", "cancel"],
        default=None,
    )
    parser.add_argument(
        "--file-exists-policy",
        choices=["ask", "overwrite", "copy", "skip", "cancel"],
        default=None,
    )
    return parser.parse_args(argv)


def apply_cli_overrides(config, args):
    config = dict(config)
    simple_overrides = {
        "run_mode": args.run_mode,
        "out_root": args.out_root,
        "n_surrogates": args.n_surrogates,
        "n_jobs": args.n_jobs,
        "random_state": args.random_state,
        "band_width": args.band_width,
        "band_step": args.band_step,
        "inv_fmin": args.inv_fmin,
        "inv_fmax": args.inv_fmax,
        "noninv_fmin": args.noninv_fmin,
        "noninv_fmax": args.noninv_fmax,
        "plot_heatmaps": args.plot_heatmaps,
        "plot_topomaps": args.plot_topomaps,
        "topomap_band_group_sets": args.topomap_band_group_sets,
        "existing_run_policy": args.existing_run_policy,
        "file_exists_policy": args.file_exists_policy,
    }

    for key, value in simple_overrides.items():
        if value is not None:
            config[key] = value

    if args.resample_sfreq is not None:
        if args.resample_sfreq <= 0:
            raise ValueError("--resample-sfreq must be greater than zero")
        config["use_resample"] = True
        config["resample_sfreq"] = args.resample_sfreq

    if args.groups == "invasive":
        config["run_invasive"] = True
        config["run_noninvasive"] = False
    elif args.groups == "noninvasive":
        config["run_invasive"] = False
        config["run_noninvasive"] = True
    elif args.groups == "both":
        config["run_invasive"] = True
        config["run_noninvasive"] = True

    return config


def path_matches_glob(path, pattern):
    pattern = pattern.casefold()
    return (
        fnmatch.fnmatch(path.name.casefold(), pattern)
        or fnmatch.fnmatch(path.as_posix().casefold(), pattern)
    )


def collect_fif_files(inputs, include_patterns=None, exclude_patterns=None):
    files = set()

    for input_value in inputs:
        path = Path(input_value).expanduser().resolve()
        if path.is_file():
            if path.suffix.casefold() != ".fif":
                raise ValueError(f"Input file is not a FIF file: {path}")
            files.add(path)
        elif path.is_dir():
            files.update(candidate.resolve() for candidate in path.rglob("*.fif"))
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")

    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []
    selected = []

    for path in sorted(files):
        if include_patterns and not any(
            path_matches_glob(path, pattern) for pattern in include_patterns
        ):
            continue
        if any(path_matches_glob(path, pattern) for pattern in exclude_patterns):
            continue
        selected.append(path)

    return selected


def resolve_target_from_channel_names(ch_names, target_requested):
    requested = str(target_requested).strip()
    if not requested:
        raise ValueError("Target name cannot be empty")

    names_casefolded = {name.casefold(): name for name in ch_names}
    requested_key = requested.casefold()
    candidates = TARGET_ALIASES.get(requested_key, [requested])

    for candidate in candidates:
        resolved = names_casefolded.get(candidate.casefold())
        if resolved is not None:
            return {
                "target_requested": requested,
                "target_resolved": resolved,
                "target_source_channel": resolved,
                "target_derivation": None,
            }

    if requested_key == "acceleration":
        for velocity_candidate in TARGET_ALIASES["velocity"]:
            velocity_channel = names_casefolded.get(velocity_candidate.casefold())
            if velocity_channel is not None:
                return {
                    "target_requested": requested,
                    "target_resolved": f"Acceleration_from_{velocity_channel}",
                    "target_source_channel": velocity_channel,
                    "target_derivation": "gradient_from_velocity",
                }

    raise ValueError(
        f"Could not resolve target {requested!r}. Tried channels: {candidates}."
    )


def resolve_target_for_file(file_path, target_requested):
    raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
    try:
        return resolve_target_from_channel_names(raw.ch_names, target_requested)
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()


def ask_choice(prompt, choices, default=None):
    choices_text = "/".join(choices)

    while True:
        answer = input(f"{prompt} [{choices_text}]").strip().lower()

        if answer == "" and default is not None:
            return default

        if answer in choices:
            return answer

        print(f"Please choose one of: {choices_text}")


def prepare_run_dir(config):
    if config["existing_run_dir"]:
        run_dir = Path(config["existing_run_dir"])
        if not run_dir.exists():
            raise FileNotFoundError(f"existing_run_dir does not exist:\n{run_dir}")
        return run_dir

    out_root = Path(config["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = build_run_id(config)
    run_dir = out_root / run_id

    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    if config["run_mode"] == "plot_only":
        print("\nRun folder already exists and run_mode='plot_only'.")
        print(f"Using existing folder:\n{run_dir}")
        return run_dir

    policy = config["existing_run_policy"]

    if policy == "ask":
        print("\nRun folder already exists:")
        print(run_dir)
        print("\nWhat do you want to do?")
        print("overwrite = delete folder and recompute")
        print("append    = keep folder and add/overwrite selected outputs")
        print("copy      = create new folder with timestamp suffix")
        print("cancel    = stop")
        policy = ask_choice(
            "Choose",
            choices=["overwrite", "append", "copy", "cancel"],
            default="cancel",
        )

    if policy == "overwrite":
        print(f"\nDeleting existing folder:\n{run_dir}")
        shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    if policy == "append":
        print(f"\nUsing existing folder without deleting:\n{run_dir}")
        return run_dir

    if policy == "copy":
        new_run_dir = Path(str(run_dir) + f"__copy_{now_string()}")
        print(f"\nCreating copy run folder:\n{new_run_dir}")
        new_run_dir.mkdir(parents=True, exist_ok=False)
        return new_run_dir

    if policy == "cancel":
        raise SystemExit("Run cancelled because folder already exists.")

    raise ValueError(f"Unknown existing_run_policy: {policy}")


def prepare_failed_run_dir(config):
    out_root = Path(config["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_dir = out_root / f"{build_run_id(config)}__failed_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_output_path(path, config):
    path = Path(path)

    if not path.exists():
        return path

    policy = config["file_exists_policy"]

    if policy == "ask":
        print("\nOutput file already exists:")
        print(path)
        print("\nWhat do you want to do?")
        print("overwrite = overwrite this file")
        print("copy      = save with timestamp suffix")
        print("skip      = do not save this file")
        print("cancel    = stop")
        policy = ask_choice(
            "Choose",
            choices=["overwrite", "copy", "skip", "cancel"],
            default="skip",
        )

    if policy == "overwrite":
        return path

    if policy == "copy":
        return path.with_name(f"{path.stem}__copy_{now_string()}{path.suffix}")

    if policy == "skip":
        print(f"Skipping save:\n{path}")
        return None

    if policy == "cancel":
        raise SystemExit("Run cancelled because output file already exists.")

    raise ValueError(f"Unknown file_exists_policy: {policy}")


def log_message(run_dir, message, also_print=True):
    run_dir = Path(run_dir)
    log_path = run_dir / "run_log.txt"

    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    if also_print:
        print(message)


def save_json(obj, path, config):
    path = resolve_output_path(path, config)
    if path is None:
        return None

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text_lines(lines, path, config):
    path = resolve_output_path(path, config)
    if path is None:
        return None

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")

    return path


def read_text_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_npy(array, path, config):
    path = resolve_output_path(path, config)
    if path is None:
        return None

    np.save(path, array)
    return path


# ============================================================
# DATA / BANDS
# ============================================================

def make_sliding_bands(fmin, fmax, width, step):
    """
    Создаёт sliding frequency bands.

    Пример:
    fmin=1, fmax=30, width=3, step=1
    -> (1,4), (2,5), (3,6), ...
    """
    if width >= (fmax - fmin):
        return [(fmin, fmax)]

    bands = []
    low = fmin

    while low + width <= fmax:
        high = low + width
        bands.append((low, high))
        low += step

    return bands


def save_bands_json(bands, path, config):
    bands_as_lists = [[float(low), float(high)] for low, high in bands]
    save_json(bands_as_lists, path, config)


def load_bands_json(path):
    bands_as_lists = load_json(path)
    return [(float(low), float(high)) for low, high in bands_as_lists]


def load_raw(config, run_dir):
    log_message(run_dir, "Loading raw FIF file...")
    raw = mne.io.read_raw_fif(config["file_path"], preload=True, verbose=False)

    original_sfreq = float(raw.info["sfreq"])
    log_message(run_dir, f"Original sampling frequency: {original_sfreq} Hz")

    max_requested_freq = 0

    if config["run_invasive"]:
        max_requested_freq = max(max_requested_freq, config["inv_fmax"])

    if config["run_noninvasive"]:
        max_requested_freq = max(max_requested_freq, config["noninv_fmax"])

    if config["use_resample"]:
        new_sfreq = float(config["resample_sfreq"])
        nyquist = new_sfreq / 2

        if nyquist <= max_requested_freq:
            raise ValueError(
                f"Resample frequency is too low.\n"
                f"resample_sfreq = {new_sfreq} Hz, Nyquist = {nyquist} Hz,\n"
                f"but max requested frequency = {max_requested_freq} Hz.\n"
                f"Increase resample_sfreq or reduce frequency range."
            )

        log_message(run_dir, f"Resampling raw to {new_sfreq} Hz...")
        raw.resample(new_sfreq)

    final_sfreq = float(raw.info["sfreq"])
    log_message(run_dir, f"Final sampling frequency: {final_sfreq} Hz")
    log_message(run_dir, f"Duration: {raw.times[-1]:.3f} sec")
    log_message(run_dir, f"Number of channels in file: {len(raw.ch_names)}")

    runtime_info = {
        "original_sfreq": original_sfreq,
        "final_sfreq": final_sfreq,
        "duration_sec": float(raw.times[-1]),
        "n_channels_total": len(raw.ch_names),
        "raw_ch_names": list(raw.ch_names),
        "target_requested": config.get("target_requested"),
        "target_resolved": config.get("target_resolved"),
        "target_source_channel": config.get("target_source_channel"),
        "target_derivation": config.get("target_derivation"),
    }

    save_json(runtime_info, Path(run_dir) / "runtime_info.json", config)

    return raw


def extract_target(raw, config, run_dir):
    target_requested = config["target_requested"]
    target_resolved = config["target_resolved"]
    source_channel = config["target_source_channel"]
    target_derivation = config.get("target_derivation")

    if source_channel not in raw.ch_names:
        raise ValueError(f"Target source channel not found: {source_channel}")

    source = raw.copy().pick([source_channel]).get_data()[0]

    if target_derivation == "gradient_from_velocity":
        target = np.gradient(source, 1.0 / float(raw.info["sfreq"]))
    else:
        target = source

    log_message(run_dir, f"Target requested: {target_requested}")
    log_message(run_dir, f"Target resolved: {target_resolved}")
    if target_derivation is not None:
        log_message(
            run_dir,
            f"Target derivation: {target_derivation} from {source_channel}",
        )
    log_message(run_dir, f"Target shape: {target.shape}")
    log_message(run_dir, f"Target mean: {np.mean(target):.6f}")
    log_message(run_dir, f"Target std: {np.std(target):.6f}")

    return target


# ============================================================
# SURROGATES
# ============================================================

def phase_randomize_signal(signal, rng):
    """
    Phase randomization surrogate:
    сохраняет амплитудный спектр target,
    но рандомизирует фазы.
    """
    signal = np.asarray(signal)

    fft_vals = np.fft.rfft(signal)
    amplitudes = np.abs(fft_vals)
    phases = np.angle(fft_vals)

    random_phases = rng.uniform(0, 2 * np.pi, size=len(phases))

    # DC component не трогаем
    random_phases[0] = phases[0]

    # Nyquist component для real-valued signal тоже лучше не трогать
    if len(random_phases) > 1:
        random_phases[-1] = phases[-1]

    randomized_fft = amplitudes * np.exp(1j * random_phases)
    surrogate = np.fft.irfft(randomized_fft, n=len(signal))

    return surrogate


def make_phase_randomized_surrogates(target, n_surrogates=200, random_state=42):
    rng = np.random.default_rng(random_state)

    surrogates = []
    for _ in range(n_surrogates):
        surrogates.append(phase_randomize_signal(target, rng))

    return np.array(surrogates)


# ============================================================
# MI COMPUTATION
# ============================================================

def compute_mi_values_for_one_feature(
    x,
    target,
    target_surrogates,
    mi_random_state,
):
    """
    x = Hilbert envelope для одного channel × band.
    """
    x_2d = x.reshape(-1, 1)

    real_mi = mutual_info_regression(
        x_2d,
        target,
        random_state=mi_random_state,
    )[0]

    surrogate_mis = np.zeros(target_surrogates.shape[0])

    for i in range(target_surrogates.shape[0]):
        surrogate_mis[i] = mutual_info_regression(
            x_2d,
            target_surrogates[i],
            random_state=mi_random_state,
        )[0]

    surr_mean = np.mean(surrogate_mis)
    surr_std = np.std(surrogate_mis)

    if surr_std > 0:
        z_mi = (real_mi - surr_mean) / surr_std
    else:
        z_mi = np.nan

    return real_mi, surr_mean, surr_std, z_mi, surrogate_mis


def process_one_channel_band(
    data,
    sfreq,
    ch_idx,
    band_idx,
    f_low,
    f_high,
    target,
    target_surrogates,
    mi_random_state,
):
    """
    Одна независимая задача:
    один канал × одна частотная полоса.
    """
    signal = data[ch_idx]

    filtered = mne.filter.filter_data(
        signal[np.newaxis, :],
        sfreq=sfreq,
        l_freq=f_low,
        h_freq=f_high,
        verbose=False,
    )[0]

    envelope = np.abs(hilbert(filtered))

    real_mi, surr_mean, surr_std, z_mi, surrogate_mis = compute_mi_values_for_one_feature(
        x=envelope,
        target=target,
        target_surrogates=target_surrogates,
        mi_random_state=mi_random_state,
    )

    return ch_idx, band_idx, real_mi, surr_mean, surr_std, z_mi, surrogate_mis


def compute_mi_heatmap_with_surrogates_parallel(
    raw,
    channels,
    target,
    target_surrogates,
    bands,
    group_name,
    n_jobs,
    mi_random_state,
    run_dir,
):
    channels_present = [ch for ch in channels if ch in raw.ch_names]

    if len(channels_present) == 0:
        raise ValueError(f"No channels found for group: {group_name}")

    raw_group = raw.copy().pick(channels_present)
    data = raw_group.get_data()
    sfreq = raw.info["sfreq"]

    n_channels = len(channels_present)
    n_bands = len(bands)

    log_message(run_dir, f"\nComputing MI + surrogates for {group_name}")
    log_message(run_dir, f"{group_name}: channels = {n_channels}")
    log_message(run_dir, f"{group_name}: bands = {n_bands}")
    log_message(run_dir, f"{group_name}: surrogates = {target_surrogates.shape[0]}")
    log_message(run_dir, f"{group_name}: parallel jobs = {n_jobs}")

    tasks = []

    for ch_idx in range(n_channels):
        for band_idx, (f_low, f_high) in enumerate(bands):
            tasks.append((ch_idx, band_idx, f_low, f_high))

    log_message(run_dir, f"{group_name}: total tasks = {len(tasks)}")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_one_channel_band)(
            data=data,
            sfreq=sfreq,
            ch_idx=ch_idx,
            band_idx=band_idx,
            f_low=f_low,
            f_high=f_high,
            target=target,
            target_surrogates=target_surrogates,
            mi_random_state=mi_random_state,
        )
        for ch_idx, band_idx, f_low, f_high in tasks
    )

    n_surrogates = target_surrogates.shape[0]
    mi_real = np.zeros((n_channels, n_bands), dtype=np.float32)
    mi_surr_mean = np.zeros((n_channels, n_bands), dtype=np.float32)
    mi_surr_std = np.zeros((n_channels, n_bands), dtype=np.float32)
    mi_z = np.zeros((n_channels, n_bands), dtype=np.float32)
    mi_surr_all = np.zeros(
        (n_surrogates, n_channels, n_bands),
        dtype=np.float32,
    )

    for ch_idx, band_idx, real_mi, surr_mean, surr_std, z_mi, surrogate_mis in results:
        mi_real[ch_idx, band_idx] = real_mi
        mi_surr_mean[ch_idx, band_idx] = surr_mean
        mi_surr_std[ch_idx, band_idx] = surr_std
        mi_z[ch_idx, band_idx] = z_mi
        mi_surr_all[:, ch_idx, band_idx] = np.asarray(
            surrogate_mis,
            dtype=np.float32,
        )

    return (
        mi_real,
        mi_surr_mean,
        mi_surr_std,
        mi_z,
        mi_surr_all,
        channels_present,
    )


# ============================================================
# SAVE / LOAD RESULTS
# ============================================================

def save_group_results(
    run_dir,
    group_name,
    mi_real,
    mi_surr_mean,
    mi_surr_std,
    mi_z,
    mi_surr_all,
    channels,
    bands,
    config,
):
    run_dir = Path(run_dir)

    mi_real = np.asarray(mi_real, dtype=np.float32)
    mi_surr_mean = np.asarray(mi_surr_mean, dtype=np.float32)
    mi_surr_std = np.asarray(mi_surr_std, dtype=np.float32)
    mi_z = np.asarray(mi_z, dtype=np.float32)
    mi_surr_all = np.asarray(mi_surr_all, dtype=np.float32)

    save_npy(mi_real, run_dir / f"{group_name}_MI_real.npy", config)
    save_npy(mi_surr_mean, run_dir / f"{group_name}_MI_surr_mean.npy", config)
    save_npy(mi_surr_std, run_dir / f"{group_name}_MI_surr_std.npy", config)
    save_npy(mi_z, run_dir / f"{group_name}_MI_z.npy", config)
    save_npy(mi_surr_all, run_dir / f"{group_name}_MI_surr_all.npy", config)

    save_text_lines(channels, run_dir / f"{group_name}_channels.txt", config)
    save_bands_json(bands, run_dir / f"{group_name}_bands_sliding.json", config)

    metadata = {
        "group_name": group_name,
        "target_requested": config.get("target_requested"),
        "target_resolved": config.get("target_resolved"),
        "n_channels": len(channels),
        "n_bands": len(bands),
        "mi_real_shape": list(mi_real.shape),
        "mi_z_shape": list(mi_z.shape),
        "mi_surr_all_shape": list(mi_surr_all.shape),
        "mi_surr_all_dtype": str(mi_surr_all.dtype),
        "array_dtypes": {
            "MI_real": str(mi_real.dtype),
            "MI_surr_mean": str(mi_surr_mean.dtype),
            "MI_surr_std": str(mi_surr_std.dtype),
            "MI_z": str(mi_z.dtype),
            "MI_surr_all": str(mi_surr_all.dtype),
        },
        "channels": list(channels),
        "bands": [[float(low), float(high)] for low, high in bands],
    }

    save_json(metadata, run_dir / f"{group_name}_metadata.json", config)

    log_message(run_dir, f"Saved {group_name} results.")


def load_group_results(run_dir, group_name):
    run_dir = Path(run_dir)

    mi_real_path = run_dir / f"{group_name}_MI_real.npy"
    mi_surr_mean_path = run_dir / f"{group_name}_MI_surr_mean.npy"
    mi_surr_std_path = run_dir / f"{group_name}_MI_surr_std.npy"
    mi_z_path = run_dir / f"{group_name}_MI_z.npy"
    mi_surr_all_path = run_dir / f"{group_name}_MI_surr_all.npy"
    channels_path = run_dir / f"{group_name}_channels.txt"
    bands_path = run_dir / f"{group_name}_bands_sliding.json"

    required = [
        mi_real_path,
        mi_surr_mean_path,
        mi_surr_std_path,
        mi_z_path,
        mi_surr_all_path,
        channels_path,
        bands_path,
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot load group results. Missing files:\n" + "\n".join(missing)
        )

    mi_real = np.load(mi_real_path)
    mi_surr_mean = np.load(mi_surr_mean_path)
    mi_surr_std = np.load(mi_surr_std_path)
    mi_z = np.load(mi_z_path)
    mi_surr_all = np.load(mi_surr_all_path, mmap_mode="r")
    channels = read_text_lines(channels_path)
    bands = load_bands_json(bands_path)

    return (
        mi_real,
        mi_surr_mean,
        mi_surr_std,
        mi_z,
        mi_surr_all,
        channels,
        bands,
    )


# ============================================================
# PLOTTING: RECTANGULAR HEATMAP
# ============================================================

def plot_heatmap(
    mi_matrix,
    channels,
    bands,
    title,
    out_path,
    config,
    colorbar_label="z-scored MI",
):
    out_path = resolve_output_path(out_path, config)
    if out_path is None:
        return None

    fig = plt.figure(figsize=(14, 8))

    plt.imshow(
        mi_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    plt.colorbar(label=colorbar_label)

    plt.yticks(
        ticks=np.arange(len(channels)),
        labels=channels,
        fontsize=8,
    )

    band_labels = [f"{low:g}-{high:g}" for low, high in bands]
    step = max(1, len(band_labels) // 12)

    plt.xticks(
        ticks=np.arange(len(band_labels))[::step],
        labels=band_labels[::step],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    plt.xlabel("Frequency band (Hz)")
    plt.ylabel("Channel")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved heatmap: {out_path}")
    return out_path


# ============================================================
# PLOTTING: TOPOGRAPHIES
# ============================================================

def average_z_scores_by_frequency_ranges(
    mi_z,
    sliding_bands,
    band_groups,
    run_dir=None,
):
    """
    Усредняет zMI по крупным частотным диапазонам.

    mi_z shape:
        n_channels × n_sliding_bands

    sliding_bands:
        [(1,4), (2,5), ...]

    band_groups:
        {"alpha": (8,13), "beta": (13,30), ...}
    """
    band_centers = np.array([
        (low + high) / 2
        for low, high in sliding_bands
    ])

    averaged = {}
    included_bands = {}

    for band_name, (fmin, fmax) in band_groups.items():
        mask = (band_centers >= fmin) & (band_centers < fmax)
        included = [sliding_bands[i] for i in np.where(mask)[0]]
        included_bands[band_name] = included

        if run_dir is not None:
            log_message(
                run_dir,
                f"Topomap band {band_name} {fmin}-{fmax} Hz includes sliding bands: {included}",
            )

        if not np.any(mask):
            averaged[band_name] = np.full(mi_z.shape[0], np.nan)
        else:
            averaged[band_name] = np.nanmean(mi_z[:, mask], axis=1)

    return averaged, included_bands


def make_eeg_info_for_topomap(channels, sfreq):
    """
    Создаёт MNE Info для topomap.
    Старые 10-20 названия переводятся в современные:
    T3->T7, T4->T8, T5->P7, T6->P8.
    """
    rename_for_montage = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }

    plot_channels = [
        rename_for_montage.get(ch, ch)
        for ch in channels
    ]

    info = mne.create_info(
        ch_names=plot_channels,
        sfreq=sfreq,
        ch_types="eeg",
    )

    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="raise")

    return info, plot_channels


def plot_noninvasive_topomaps(
    z_by_band,
    channels,
    sfreq,
    title,
    out_path,
    config,
):
    out_path = resolve_output_path(out_path, config)
    if out_path is None:
        return None

    # Пропускаем bands, где вообще нет конечных значений.
    valid_band_names = []
    for band_name, values in z_by_band.items():
        values = np.asarray(values)
        if np.any(np.isfinite(values)):
            valid_band_names.append(band_name)
        else:
            print(f"Skipping topomap band with no finite values: {band_name}")

    if len(valid_band_names) == 0:
        print("No valid bands for topomap. Nothing to plot.")
        return None

    z_by_band_valid = {
        name: np.asarray(z_by_band[name])
        for name in valid_band_names
    }

    info, plot_channels = make_eeg_info_for_topomap(
        channels=channels,
        sfreq=sfreq,
    )

    all_values = np.concatenate([
        values
        for values in z_by_band_valid.values()
    ])

    vmax = np.nanmax(np.abs(all_values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    n_maps = len(valid_band_names)

    fig, axes = plt.subplots(
        1,
        n_maps,
        figsize=(4 * n_maps, 4),
        constrained_layout=True,
    )

    if n_maps == 1:
        axes = [axes]

    last_im = None

    for ax, band_name in zip(axes, valid_band_names):
        data = z_by_band_valid[band_name]

        try:
            last_im, _ = mne.viz.plot_topomap(
                data,
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap=config["topomap_cmap"],
                vlim=(-vmax, vmax),
            )
        except TypeError:
            last_im, _ = mne.viz.plot_topomap(
                data,
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap=config["topomap_cmap"],
                vmin=-vmax,
                vmax=vmax,
            )

        ax.set_title(band_name)

    fig.suptitle(title, fontsize=14)

    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes,
            shrink=0.75,
            label="z-scored MI",
        )

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved topomaps: {out_path}")
    return out_path


# ============================================================
# CONFIG / SUMMARY SAVING
# ============================================================

def save_initial_config(config, run_dir):
    config_to_save = dict(config)
    config_to_save["run_id"] = build_run_id(config)
    config_to_save["created_at"] = datetime.now().isoformat(timespec="seconds")

    if config["run_mode"] == "compute":
        filename = "config.json"
    else:
        filename = f"plot_config_{now_string()}.json"

    save_json(config_to_save, Path(run_dir) / filename, config)


def save_analysis_summary(run_dir, summary, config):
    summary = dict(summary)
    summary["saved_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(summary, Path(run_dir) / "analysis_summary.json", config)


def save_summary_csv(run_dir, group_results, config):
    path = resolve_output_path(Path(run_dir) / "summary.csv", config)
    if path is None:
        return None

    fieldnames = [
        "group",
        "channel",
        "band_fmin",
        "band_fmax",
        "MI_real",
        "MI_surr_mean",
        "MI_surr_std",
        "MI_z",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in group_results:
            for ch_idx, channel in enumerate(result["channels"]):
                for band_idx, (fmin, fmax) in enumerate(result["bands"]):
                    writer.writerow({
                        "group": result["group"],
                        "channel": channel,
                        "band_fmin": float(fmin),
                        "band_fmax": float(fmax),
                        "MI_real": float(result["mi_real"][ch_idx, band_idx]),
                        "MI_surr_mean": float(result["mi_surr_mean"][ch_idx, band_idx]),
                        "MI_surr_std": float(result["mi_surr_std"][ch_idx, band_idx]),
                        "MI_z": float(result["mi_z"][ch_idx, band_idx]),
                    })

    log_message(run_dir, f"Saved tabular summary: {path}")
    return path


def get_git_info():
    git_info = {
        "branch": None,
        "commit": None,
        "dirty": None,
    }

    try:
        script_dir = Path(__file__).resolve().parent

        def git_output(*args):
            result = subprocess.run(
                ["git", "-C", str(script_dir), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()

        git_info["branch"] = git_output("rev-parse", "--abbrev-ref", "HEAD")
        git_info["commit"] = git_output("rev-parse", "HEAD")
        git_info["dirty"] = bool(git_output("status", "--porcelain"))
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    return git_info


def get_input_file_info(file_path):
    path = Path(file_path).expanduser().resolve()
    info = {
        "path": str(path),
        "size_bytes": None,
        "modified_at": None,
    }

    try:
        stat = path.stat()
        info["size_bytes"] = int(stat.st_size)
        info["modified_at"] = datetime.fromtimestamp(
            stat.st_mtime,
        ).astimezone().isoformat(timespec="seconds")
    except OSError:
        pass

    return info


def collect_updated_output_files(run_dir, started_timestamp):
    run_dir = Path(run_dir)
    output_files = []

    if not run_dir.exists():
        return output_files

    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        try:
            if path.stat().st_mtime >= started_timestamp - 1.0:
                output_files.append(str(path.relative_to(run_dir)))
        except OSError:
            continue

    if "run_manifest.json" not in output_files:
        output_files.append("run_manifest.json")

    return sorted(output_files)


def write_run_manifest(run_dir, manifest):
    """Always write the canonical latest-run manifest, bypassing file policies."""
    path = Path(run_dir) / "run_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def append_analysis_run(out_root, record):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "analysis_runs.jsonl"

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


# ============================================================
# RUN MODES
# ============================================================

def run_compute(config, run_dir, activity=None):
    log_message(run_dir, "========== START COMPUTE RUN ==========")
    save_initial_config(config, run_dir)

    if activity is None:
        activity = {}
    activity.setdefault("computed_groups", [])
    activity.setdefault("plotted_heatmaps", [])
    activity.setdefault("plotted_topomaps", [])
    group_results = []

    if not config["run_invasive"] and not config["run_noninvasive"]:
        log_message(
            run_dir,
            "WARNING: both run_invasive and run_noninvasive are False; "
            "no group MI results will be computed.",
        )

    if config["n_surrogates"] < 2:
        log_message(
            run_dir,
            "WARNING: n_surrogates < 2. zMI will be NaN or not meaningful.",
        )

    if config["n_surrogates"] < 20:
        log_message(
            run_dir,
            "WARNING: n_surrogates is low. This is okay for code testing, not for final analysis.",
        )

    raw = load_raw(config, run_dir)
    sfreq = float(raw.info["sfreq"])

    target = extract_target(raw=raw, config=config, run_dir=run_dir)

    log_message(run_dir, f"Creating {config['n_surrogates']} phase-randomized surrogates...")

    target_surrogates = make_phase_randomized_surrogates(
        target=target,
        n_surrogates=config["n_surrogates"],
        random_state=config["random_state"],
    )

    save_npy(
        target_surrogates,
        Path(run_dir) / "target_surrogates.npy",
        config,
    )

    invasive_bands = make_sliding_bands(
        config["inv_fmin"],
        config["inv_fmax"],
        config["band_width"],
        config["band_step"],
    )

    noninvasive_bands = make_sliding_bands(
        config["noninv_fmin"],
        config["noninv_fmax"],
        config["band_width"],
        config["band_step"],
    )

    save_bands_json(
        invasive_bands,
        Path(run_dir) / "invasive_bands_sliding.json",
        config,
    )

    save_bands_json(
        noninvasive_bands,
        Path(run_dir) / "noninvasive_bands_sliding.json",
        config,
    )

    log_message(run_dir, f"Invasive bands: {len(invasive_bands)}")
    log_message(run_dir, f"Noninvasive bands: {len(noninvasive_bands)}")

    summary = {
        "target_channel": config["target_channel"],
        "target_requested": config["target_requested"],
        "target_resolved": config["target_resolved"],
        "sfreq": sfreq,
        "n_surrogates": config["n_surrogates"],
        "random_state": config["random_state"],
        "band_width": config["band_width"],
        "band_step": config["band_step"],
        "run_invasive": config["run_invasive"],
        "run_noninvasive": config["run_noninvasive"],
    }

    mi_inv_z = None
    inv_channels = None

    mi_noninv_z = None
    noninv_channels = None

    # ---------- invasive ----------
    if config["run_invasive"]:
        mi_inv, mi_inv_surr_mean, mi_inv_surr_std, mi_inv_z, mi_inv_surr_all, inv_channels = compute_mi_heatmap_with_surrogates_parallel(
            raw=raw,
            channels=INVASIVE_CHANNELS,
            target=target,
            target_surrogates=target_surrogates,
            bands=invasive_bands,
            group_name="invasive",
            n_jobs=config["n_jobs"],
            mi_random_state=config["random_state"],
            run_dir=run_dir,
        )
        activity["computed_groups"].append("invasive")

        save_group_results(
            run_dir=run_dir,
            group_name="invasive",
            mi_real=mi_inv,
            mi_surr_mean=mi_inv_surr_mean,
            mi_surr_std=mi_inv_surr_std,
            mi_z=mi_inv_z,
            mi_surr_all=mi_inv_surr_all,
            channels=inv_channels,
            bands=invasive_bands,
            config=config,
        )

        summary["invasive_channels"] = inv_channels
        summary["invasive_mi_z_shape"] = list(mi_inv_z.shape)
        summary["invasive_mi_surr_all_shape"] = list(mi_inv_surr_all.shape)
        group_results.append({
            "group": "invasive",
            "channels": inv_channels,
            "bands": invasive_bands,
            "mi_real": mi_inv,
            "mi_surr_mean": mi_inv_surr_mean,
            "mi_surr_std": mi_inv_surr_std,
            "mi_z": mi_inv_z,
        })

        if config["plot_heatmaps"]:
            heatmap_path = plot_heatmap(
                mi_matrix=mi_inv_z,
                channels=inv_channels,
                bands=invasive_bands,
                title=f"Invasive zMI with {config['target_channel']}",
                out_path=Path(run_dir) / "zMI_invasive_rect_heatmap.png",
                config=config,
                colorbar_label="z-scored MI",
            )
            if heatmap_path is not None:
                activity["plotted_heatmaps"].append("invasive")

    # ---------- noninvasive ----------
    if config["run_noninvasive"]:
        mi_noninv, mi_noninv_surr_mean, mi_noninv_surr_std, mi_noninv_z, mi_noninv_surr_all, noninv_channels = compute_mi_heatmap_with_surrogates_parallel(
            raw=raw,
            channels=NONINVASIVE_CHANNELS,
            target=target,
            target_surrogates=target_surrogates,
            bands=noninvasive_bands,
            group_name="noninvasive",
            n_jobs=config["n_jobs"],
            mi_random_state=config["random_state"],
            run_dir=run_dir,
        )
        activity["computed_groups"].append("noninvasive")

        save_group_results(
            run_dir=run_dir,
            group_name="noninvasive",
            mi_real=mi_noninv,
            mi_surr_mean=mi_noninv_surr_mean,
            mi_surr_std=mi_noninv_surr_std,
            mi_z=mi_noninv_z,
            mi_surr_all=mi_noninv_surr_all,
            channels=noninv_channels,
            bands=noninvasive_bands,
            config=config,
        )

        summary["noninvasive_channels"] = noninv_channels
        summary["noninvasive_mi_z_shape"] = list(mi_noninv_z.shape)
        summary["noninvasive_mi_surr_all_shape"] = list(mi_noninv_surr_all.shape)
        group_results.append({
            "group": "noninvasive",
            "channels": noninv_channels,
            "bands": noninvasive_bands,
            "mi_real": mi_noninv,
            "mi_surr_mean": mi_noninv_surr_mean,
            "mi_surr_std": mi_noninv_surr_std,
            "mi_z": mi_noninv_z,
        })

        if config["plot_heatmaps"]:
            heatmap_path = plot_heatmap(
                mi_matrix=mi_noninv_z,
                channels=noninv_channels,
                bands=noninvasive_bands,
                title=f"Noninvasive zMI with {config['target_channel']}",
                out_path=Path(run_dir) / "zMI_noninvasive_rect_heatmap.png",
                config=config,
                colorbar_label="z-scored MI",
            )
            if heatmap_path is not None:
                activity["plotted_heatmaps"].append("noninvasive")

        if config["plot_topomaps"]:
            for set_name in config["topomap_band_group_sets"]:
                if set_name not in BAND_GROUP_SETS:
                    raise ValueError(f"Unknown band group set: {set_name}")

                band_groups = BAND_GROUP_SETS[set_name]

                z_by_band, included_bands = average_z_scores_by_frequency_ranges(
                    mi_z=mi_noninv_z,
                    sliding_bands=noninvasive_bands,
                    band_groups=band_groups,
                    run_dir=run_dir,
                )

                save_json(
                    {
                        "set_name": set_name,
                        "band_groups": band_groups,
                        "included_sliding_bands": {
                            name: [[float(low), float(high)] for low, high in bands]
                            for name, bands in included_bands.items()
                        },
                    },
                    Path(run_dir) / f"topomap_band_groups_{set_name}.json",
                    config,
                )

                topomap_path = plot_noninvasive_topomaps(
                    z_by_band=z_by_band,
                    channels=noninv_channels,
                    sfreq=sfreq,
                    title=f"Noninvasive EEG zMI with {config['target_channel']} | {set_name}",
                    out_path=Path(run_dir) / f"zMI_noninvasive_topomaps_{set_name}.png",
                    config=config,
                )
                if topomap_path is not None:
                    activity["plotted_topomaps"].append(set_name)

    if config["plot_topomaps"] and not config["run_noninvasive"]:
        log_message(
            run_dir,
            "WARNING: plot_topomaps=True, but run_noninvasive=False. "
            "Skipping noninvasive topomaps.",
        )

    save_summary_csv(run_dir, group_results, config)
    summary["activity"] = activity

    save_analysis_summary(run_dir, summary, config)

    log_message(run_dir, "========== COMPUTE RUN FINISHED ==========")
    return activity


def run_plot_only(config, run_dir, activity=None):
    log_message(run_dir, "========== START PLOT-ONLY RUN ==========")
    save_initial_config(config, run_dir)

    if activity is None:
        activity = {}
    activity.setdefault("loaded_groups", [])
    activity.setdefault("plotted_heatmaps", [])
    activity.setdefault("plotted_topomaps", [])
    loaded_results = {}

    def try_load_group(group_name):
        if group_name in loaded_results:
            return loaded_results[group_name]

        try:
            result = load_group_results(run_dir, group_name)
        except (FileNotFoundError, OSError, ValueError) as error:
            log_message(
                run_dir,
                f"WARNING: requested {group_name} results are unavailable. "
                f"Skipping requested plots for this group. Details: {error}",
            )
            loaded_results[group_name] = None
            return None

        loaded_results[group_name] = result
        activity["loaded_groups"].append(group_name)
        return result

    if not config["plot_heatmaps"] and not config["plot_topomaps"]:
        log_message(
            run_dir,
            "WARNING: plot_heatmaps=False and plot_topomaps=False; no plots requested.",
        )

    if config["plot_heatmaps"]:
        if config["run_invasive"]:
            invasive_result = try_load_group("invasive")
            if invasive_result is not None:
                _, _, _, mi_inv_z, _, inv_channels, invasive_bands = invasive_result
                heatmap_path = plot_heatmap(
                    mi_matrix=mi_inv_z,
                    channels=inv_channels,
                    bands=invasive_bands,
                    title="Invasive zMI with saved target",
                    out_path=Path(run_dir) / f"zMI_invasive_rect_heatmap_plotonly_{now_string()}.png",
                    config=config,
                    colorbar_label="z-scored MI",
                )
                if heatmap_path is not None:
                    activity["plotted_heatmaps"].append("invasive")

        if config["run_noninvasive"]:
            noninvasive_result = try_load_group("noninvasive")
            if noninvasive_result is not None:
                _, _, _, mi_noninv_z, _, noninv_channels, noninvasive_bands = noninvasive_result
                heatmap_path = plot_heatmap(
                    mi_matrix=mi_noninv_z,
                    channels=noninv_channels,
                    bands=noninvasive_bands,
                    title="Noninvasive zMI with saved target",
                    out_path=Path(run_dir) / f"zMI_noninvasive_rect_heatmap_plotonly_{now_string()}.png",
                    config=config,
                    colorbar_label="z-scored MI",
                )
                if heatmap_path is not None:
                    activity["plotted_heatmaps"].append("noninvasive")

    if config["plot_topomaps"]:
        if not config["run_noninvasive"]:
            log_message(
                run_dir,
                "WARNING: plot_topomaps=True, but run_noninvasive=False. "
                "Skipping noninvasive topomaps.",
            )
        else:
            noninvasive_result = try_load_group("noninvasive")
            if noninvasive_result is not None:
                _, _, _, mi_noninv_z, _, noninv_channels, noninvasive_bands = noninvasive_result

                # Для topomap sfreq не принципиальна, нужна только для MNE info.
                runtime_info_path = Path(run_dir) / "runtime_info.json"
                if runtime_info_path.exists():
                    runtime_info = load_json(runtime_info_path)
                    sfreq = float(runtime_info["final_sfreq"])
                else:
                    sfreq = float(config["resample_sfreq"]) if config["use_resample"] else 1000.0

                for set_name in config["topomap_band_group_sets"]:
                    if set_name not in BAND_GROUP_SETS:
                        raise ValueError(f"Unknown band group set: {set_name}")

                    band_groups = BAND_GROUP_SETS[set_name]

                    z_by_band, included_bands = average_z_scores_by_frequency_ranges(
                        mi_z=mi_noninv_z,
                        sliding_bands=noninvasive_bands,
                        band_groups=band_groups,
                        run_dir=run_dir,
                    )

                    save_json(
                        {
                            "set_name": set_name,
                            "band_groups": band_groups,
                            "included_sliding_bands": {
                                name: [[float(low), float(high)] for low, high in bands]
                                for name, bands in included_bands.items()
                            },
                        },
                        Path(run_dir) / f"topomap_band_groups_{set_name}_plotonly_{now_string()}.json",
                        config,
                    )

                    topomap_path = plot_noninvasive_topomaps(
                        z_by_band=z_by_band,
                        channels=noninv_channels,
                        sfreq=sfreq,
                        title=f"Noninvasive EEG zMI | plot-only | {set_name}",
                        out_path=Path(run_dir) / f"zMI_noninvasive_topomaps_{set_name}_plotonly_{now_string()}.png",
                        config=config,
                    )
                    if topomap_path is not None:
                        activity["plotted_topomaps"].append(set_name)

    log_message(run_dir, "========== PLOT-ONLY RUN FINISHED ==========")
    return activity


# ============================================================
# MAIN
# ============================================================

def run_single(config, cli_arguments=None):
    config = dict(config)
    config.setdefault("target_requested", config["target_channel"])
    config["target_channel"] = config["target_requested"]
    cli_arguments = json_safe(cli_arguments or {})
    started_at = datetime.now().astimezone()
    started_timestamp = time.time()
    started_counter = time.perf_counter()
    run_dir = None
    status = "running"
    activity = {
        "computed_groups": [],
        "loaded_groups": [],
        "plotted_heatmaps": [],
        "plotted_topomaps": [],
    }
    error_info = None
    finalization_error = None

    try:
        target_info = resolve_target_for_file(
            config["file_path"],
            config["target_requested"],
        )
        config.update(target_info)
        config["target_channel"] = config["target_resolved"]

        run_dir = prepare_run_dir(config)

        print("\nRun directory:")
        print(run_dir)

        if config["run_mode"] == "compute":
            run_compute(config, run_dir, activity=activity)

        elif config["run_mode"] == "plot_only":
            run_plot_only(config, run_dir, activity=activity)

        else:
            raise ValueError(
                f"Unknown run_mode: {config['run_mode']}. "
                f"Use 'compute' or 'plot_only'."
            )

        status = "success"
        print("\nDone.")
        print("Results are in:")
        print(run_dir)

    except BaseException as error:
        status = "failed"
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

        if run_dir is None:
            try:
                run_dir = prepare_failed_run_dir(config)
            except OSError:
                run_dir = None

        if run_dir is not None:
            try:
                log_message(
                    run_dir,
                    f"ERROR: {error_info['type']}: {error_info['message']}\n"
                    f"{error_info['traceback']}",
                )
            except OSError:
                pass

        raise

    finally:
        finished_at = datetime.now().astimezone()
        duration_seconds = time.perf_counter() - started_counter
        run_id = Path(run_dir).name if run_dir is not None else build_run_id(config)
        output_files = (
            collect_updated_output_files(run_dir, started_timestamp)
            if run_dir is not None
            else []
        )

        manifest = {
            "run_id": run_id,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": duration_seconds,
            "status": status,
            "script_path": str(Path(__file__).resolve()),
            "command_line": subprocess.list2cmdline([str(arg) for arg in sys.argv]),
            "command_line_arguments": cli_arguments,
            "input_file": get_input_file_info(config["file_path"]),
            "target_channel": config["target_channel"],
            "target_requested": config.get("target_requested"),
            "target_resolved": config.get("target_resolved"),
            "run_mode": config["run_mode"],
            "run_invasive": config["run_invasive"],
            "run_noninvasive": config["run_noninvasive"],
            "plot_heatmaps": config["plot_heatmaps"],
            "plot_topomaps": config["plot_topomaps"],
            "config": config,
            "run_directory": str(Path(run_dir).resolve()) if run_dir is not None else None,
            "activity": activity,
            "output_files": output_files,
            "git": get_git_info(),
            "error": error_info,
        }

        jsonl_record = {
            "run_id": run_id,
            "started_at": manifest["started_at"],
            "finished_at": manifest["finished_at"],
            "duration_seconds": duration_seconds,
            "status": status,
            "input_file": manifest["input_file"]["path"],
            "target_channel": config["target_channel"],
            "run_mode": config["run_mode"],
            "run_directory": manifest["run_directory"],
            "groups": activity,
            "script_path": manifest["script_path"],
            "command_line": manifest["command_line"],
            "command_line_arguments": cli_arguments,
            "config": config,
            "target_requested": config.get("target_requested"),
            "target_resolved": config.get("target_resolved"),
            "output_files": output_files,
            "git": manifest["git"],
            "error": error_info,
            "error_message": error_info["message"] if error_info else None,
        }

        try:
            if run_dir is not None:
                write_run_manifest(run_dir, manifest)
            append_analysis_run(config["out_root"], jsonl_record)
        except Exception as error:
            finalization_error = error
            print(
                f"WARNING: could not finish run manifest/JSONL logging: {error}",
                file=sys.stderr,
            )

    if finalization_error is not None:
        raise RuntimeError(
            "Analysis completed, but run manifest/JSONL logging failed."
        ) from finalization_error

    return manifest


def main(argv=None):
    parser_args = parse_cli_args(argv)

    try:
        base_config = apply_cli_overrides(CONFIG, parser_args)
        input_values = parser_args.inputs or [base_config["file_path"]]
        files = collect_fif_files(
            input_values,
            include_patterns=parser_args.include,
            exclude_patterns=parser_args.exclude,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Input/configuration error: {error}") from error

    if not files:
        raise SystemExit("No FIF files matched the provided inputs/include/exclude patterns.")

    requested_targets = parser_args.targets or [
        base_config.get("target_requested", base_config["target_channel"])
    ]
    targets = []
    seen_targets = set()
    for target in requested_targets:
        key = str(target).casefold()
        if key not in seen_targets:
            seen_targets.add(key)
            targets.append(str(target))

    n_runs = len(files) * len(targets)
    if base_config.get("existing_run_dir") and n_runs > 1:
        raise SystemExit(
            "CONFIG['existing_run_dir'] can only be used for a single file × target run. "
            "Clear it before batch execution."
        )

    cli_arguments = json_safe(vars(parser_args))
    successes = []
    failures = []

    print(f"Selected FIF files: {len(files)}")
    print(f"Requested targets: {targets}")
    print(f"Planned runs: {n_runs}")

    run_number = 0
    for file_path in files:
        for target_requested in targets:
            run_number += 1
            run_config = dict(base_config)
            run_config["file_path"] = str(file_path)
            run_config["target_requested"] = target_requested
            run_config.pop("target_resolved", None)
            run_config.pop("target_source_channel", None)
            run_config.pop("target_derivation", None)

            print("\n" + "=" * 80)
            print(f"BATCH RUN {run_number}/{n_runs}")
            print(f"File: {file_path}")
            print(f"Target requested: {target_requested}")

            try:
                manifest = run_single(
                    run_config,
                    cli_arguments=cli_arguments,
                )
            except (Exception, SystemExit) as error:
                failures.append({
                    "file": str(file_path),
                    "target_requested": target_requested,
                    "error": str(error),
                })
                print(
                    f"FAILED: {file_path.name} × {target_requested}: {error}",
                    file=sys.stderr,
                )
                continue

            successes.append({
                "file": str(file_path),
                "target_requested": target_requested,
                "target_resolved": manifest.get("target_resolved"),
                "run_directory": manifest.get("run_directory"),
            })

    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print(f"Successful runs: {len(successes)}")
    print(f"Failed runs: {len(failures)}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(
                f"- {failure['file']} × {failure['target_requested']}: "
                f"{failure['error']}"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
