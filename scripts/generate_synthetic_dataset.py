"""
Generates a synthetic, SWaT-shaped dataset for CTMAS.

The real SWaT (Secure Water Treatment) dataset is distributed by iTrust,
Singapore University of Technology and Design, under a data-use agreement,
and cannot be redistributed inside this repository. This script produces a
drop-in replacement with the same column schema (per-stage sensor/actuator
tags following the FIT/LIT/AIT/DPIT/PIT/MV/P/UV naming convention, plus a
`Normal/Attack` label column) so the rest of the pipeline -- data loading,
federated training, anomaly detection, SHAP attribution, and threat mapping
-- is runnable end-to-end without access to the restricted dataset.

To use the real dataset instead, request access from iTrust
(https://itrust.sutd.edu.sg/itrust-labs_datasets/) and drop the official
`normal.csv` / `attack.csv` files into `dataset/`, replacing the synthetic
ones -- no code changes are required.

Usage:
    python scripts/generate_synthetic_dataset.py [--seed 42]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config  # noqa: E402

# A representative (non-exhaustive) subset of real SWaT tags per stage.
# Continuous process variables (flow/level/analyzer/pressure sensors) get
# smooth cyclical signals; actuators (motorized valves, pumps, UV lamp) get
# discrete on/off style signals.
CONTINUOUS_PREFIXES = ("FIT", "LIT", "AIT", "DPIT", "PIT")

STAGE_TAGS = {
    1: ["FIT101", "LIT101", "MV101", "P101"],
    2: ["AIT201", "AIT202", "FIT201", "MV201", "P201"],
    3: ["DPIT301", "FIT301", "LIT301", "MV301", "P301"],
    4: ["AIT401", "FIT401", "LIT401", "P401", "UV401"],
    5: ["AIT501", "FIT501", "PIT501", "P501"],
    6: ["FIT601", "P601"],
}

ALL_TAGS = [tag for tags in STAGE_TAGS.values() for tag in tags]

# (stage, tag, kind, start, length) attack windows injected into attack.csv.
# kind is one of: "spike" (abrupt fault / sensor spoofing), "drift" (slow,
# stealthy ramp -- the scenario EWMA scoring is designed to catch), or
# "flip" (actuator manipulation / denial-of-service on control).
ATTACK_WINDOWS = [
    (1, "FIT101", "spike", 30, 30),
    (2, "AIT201", "drift", 90, 30),
    (3, "MV301", "flip", 150, 30),
    (4, "AIT401", "spike", 210, 30),
    (5, "PIT501", "drift", 270, 30),
]


def _tag_signal(tag, n_rows, rng):
    is_continuous = tag.startswith(CONTINUOUS_PREFIXES)
    t = np.arange(n_rows)
    if is_continuous:
        base = rng.uniform(20, 80)
        amplitude = rng.uniform(2, 6)
        period = rng.uniform(80, 200)
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, 0.3, size=n_rows)
        return base + amplitude * np.sin(2 * np.pi * t / period + phase) + noise
    # Discrete actuator: slow square-wave cycling (e.g. pump duty cycle).
    period = rng.integers(40, 90)
    jitter = rng.normal(0, 0.02, size=n_rows)
    return ((t // period) % 2).astype(float) + jitter


def _apply_attack(series, kind, start, length, rng):
    series = series.copy()
    end = min(start + length, len(series))
    if kind == "spike":
        series[start:end] += rng.uniform(8, 15) * np.sign(rng.normal())
    elif kind == "drift":
        ramp = np.linspace(0, rng.uniform(6, 12), end - start)
        series[start:end] += ramp
    elif kind == "flip":
        series[start:end] = 1.0 - np.round(series[start:end])
    return series


def _build_frame(n_rows, rng, inject_attacks):
    data = {"Timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="s")}
    labels = np.zeros(n_rows, dtype=bool)

    for stage, tags in STAGE_TAGS.items():
        for tag in tags:
            data[tag] = _tag_signal(tag, n_rows, rng)

    if inject_attacks:
        for stage, tag, kind, start, length in ATTACK_WINDOWS:
            if start >= n_rows:
                continue
            data[tag] = _apply_attack(data[tag], kind, start, length, rng)
            end = min(start + length, n_rows)
            labels[start:end] = True

    data["Normal/Attack"] = np.where(labels, "Attack", "Normal")
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normal-rows", type=int, default=Config.TRAIN_SAMPLES_PER_STAGE + 200
    )
    parser.add_argument(
        "--attack-rows", type=int, default=Config.TEST_SAMPLES_PER_STAGE
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(Config.DATASET_DIR, exist_ok=True)

    normal_path = os.path.join(Config.DATASET_DIR, Config.NORMAL_DATA_FILE)
    attack_path = os.path.join(Config.DATASET_DIR, Config.ATTACK_DATA_FILE)

    if not args.force and os.path.exists(normal_path) and os.path.exists(attack_path):
        print(f"'{normal_path}' and '{attack_path}' already exist. Use --force to regenerate.")
        return

    normal_df = _build_frame(args.normal_rows, rng, inject_attacks=False)
    attack_df = _build_frame(args.attack_rows, rng, inject_attacks=True)

    normal_df.to_csv(normal_path, index=False)
    attack_df.to_csv(attack_path, index=False)

    n_attack_rows = int((attack_df["Normal/Attack"] == "Attack").sum())
    print(f"Wrote {len(normal_df)} normal rows to '{normal_path}'")
    print(
        f"Wrote {len(attack_df)} rows ({n_attack_rows} labelled Attack) to "
        f"'{attack_path}'"
    )
    print(f"Tags: {ALL_TAGS}")


if __name__ == "__main__":
    main()
