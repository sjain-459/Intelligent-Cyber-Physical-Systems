"""
Downloads the real SWaT dataset from its Kaggle mirror and normalizes it
into the schema `data_pipeline.py` expects (dataset/normal.csv,
dataset/attack.csv).

The original SWaT (Secure Water Treatment) dataset is distributed by
iTrust, Singapore University of Technology and Design, under a data-use
agreement -- see https://itrust.sutd.edu.sg/itrust-labs_datasets/ for the
authoritative source. The Kaggle listing this script pulls from
(https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system)
is a third-party mirror, not an official iTrust distribution; check its
license/terms before using it beyond personal research, and prefer
requesting the dataset directly from iTrust for anything else.

This script never commits the downloaded data anywhere -- `dataset/` is
git-ignored, same as the synthetic files.

Two ways to get the source files:

1. Automatic (needs a Kaggle API token): `pip install kagglehub`, configure
   credentials as usual (~/.kaggle/kaggle.json, or the KAGGLE_USERNAME /
   KAGGLE_KEY environment variables) -- this script does not read, request,
   or handle those credentials itself, kagglehub picks them up from your
   existing Kaggle configuration -- then run with no arguments.
2. Manual (no API token needed): click "Download" on the Kaggle page
   (https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system),
   extract the zip, then run with --source-dir pointing at the extracted
   folder.

Usage:
    python scripts/download_kaggle_dataset.py
    python scripts/download_kaggle_dataset.py --source-dir ~/Downloads/swat-dataset-secure-water-treatment-system
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config  # noqa: E402

KAGGLE_DATASET = "vishala28/swat-dataset-secure-water-treatment-system"


def _normalize(df):
    """Cleans up the common formatting quirks of raw SWaT CSV exports."""
    df.columns = df.columns.str.strip()

    # Some exports carry an extra leading title/description row that lands
    # in the header itself, or an unnamed index column - drop junk columns.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # The label column has appeared as "Normal/Attack" or, in some raw
    # exports, with a typo ("Attack " / "A ttack"). Normalize to two
    # canonical values.
    label_col = next(c for c in df.columns if "attack" in c.lower())
    df = df.rename(columns={label_col: "Normal/Attack"})
    df["Normal/Attack"] = df["Normal/Attack"].astype(str).str.strip()
    df["Normal/Attack"] = df["Normal/Attack"].apply(
        lambda v: "Normal" if v.strip().lower().startswith("normal") else "Attack"
    )

    # Sensor columns are sometimes exported with comma decimal separators.
    # Checking `dtype == object` is not reliable here: pandas >= 3.0 uses a
    # native `str` dtype for text columns instead of `object`, so that check
    # silently never matches under the pandas version this project actually
    # installs (pandas>=2.0 currently resolves to 3.x). Check numeric-ness
    # directly instead, which is correct across pandas versions.
    for col in df.columns:
        if col in ("Timestamp", "Normal/Attack"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    return df


def _find_file(root, keyword):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if keyword in name.lower() and name.lower().endswith((".csv", ".xlsx")):
                return os.path.join(dirpath, name)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset/ files")
    parser.add_argument(
        "--source-dir",
        help="Path to an already-downloaded-and-extracted copy of the Kaggle "
        "dataset (skips kagglehub/API credentials entirely).",
    )
    args = parser.parse_args()

    normal_out = os.path.join(Config.DATASET_DIR, Config.NORMAL_DATA_FILE)
    attack_out = os.path.join(Config.DATASET_DIR, Config.ATTACK_DATA_FILE)
    if not args.force and os.path.exists(normal_out) and os.path.exists(attack_out):
        print(f"'{normal_out}' and '{attack_out}' already exist. Use --force to re-download.")
        return

    if args.source_dir:
        dataset_root = os.path.expanduser(args.source_dir)
        if not os.path.isdir(dataset_root):
            sys.exit(f"'{dataset_root}' is not a directory.")
        print(f"Using local copy at: {dataset_root}")
    else:
        try:
            import kagglehub
        except ImportError:
            sys.exit(
                "kagglehub is required for automatic download. Install it with:\n"
                "    pip install kagglehub\n"
                "and make sure your Kaggle API credentials are configured "
                "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).\n"
                "Alternatively, download the dataset manually from Kaggle and "
                "re-run with --source-dir pointing at the extracted folder."
            )

        print(f"Downloading '{KAGGLE_DATASET}' via kagglehub (uses your local Kaggle credentials)...")
        dataset_root = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"Downloaded to: {dataset_root}")

    normal_src = _find_file(dataset_root, "normal")
    attack_src = _find_file(dataset_root, "attack")
    if not normal_src or not attack_src:
        sys.exit(
            f"Could not locate normal/attack files under '{dataset_root}'. "
            "The mirror's file layout may have changed -- inspect it manually "
            "and place normalized `normal.csv` / `attack.csv` under "
            f"'{Config.DATASET_DIR}/' yourself."
        )

    print(f"Normalizing '{normal_src}' -> '{normal_out}'")
    normal_df = _normalize(pd.read_csv(normal_src) if normal_src.endswith(".csv") else pd.read_excel(normal_src))
    os.makedirs(Config.DATASET_DIR, exist_ok=True)
    normal_df.to_csv(normal_out, index=False)

    print(f"Normalizing '{attack_src}' -> '{attack_out}'")
    attack_df = _normalize(pd.read_csv(attack_src) if attack_src.endswith(".csv") else pd.read_excel(attack_src))
    attack_df.to_csv(attack_out, index=False)

    n_attack_rows = int((attack_df["Normal/Attack"] == "Attack").sum())
    print(f"Wrote {len(normal_df)} normal rows and {len(attack_df)} attack-window rows "
          f"({n_attack_rows} labelled Attack).")
    print("Done. The pipeline will auto-size its feature padding to this dataset's real tag counts.")


if __name__ == "__main__":
    main()
