import numpy as np
import pandas as pd
import pytest

from config import Config
from data_pipeline import (
    apply_stage_features,
    create_sliding_windows,
    get_stage_dataloaders,
    infer_num_features,
)


def test_stage_tag_matching_is_disjoint():
    """
    Regression test: stage assignment must key off the FIRST digit of a
    tag's numeric suffix (SWaT convention), not "does this string contain
    the stage digit anywhere". AIT201 (stage 2) must never leak into P1's
    client just because it contains the character '1'.
    """
    df = pd.DataFrame({
        "FIT101": [1.0], "LIT101": [1.0],
        "AIT201": [1.0], "FIT201": [1.0],
        "Timestamp": ["t0"], "Normal/Attack": ["Normal"],
    })
    stage1_df, stage1_cols = apply_stage_features(df.copy(), stage_id=0)
    stage2_df, stage2_cols = apply_stage_features(df.copy(), stage_id=1)

    assert set(stage1_cols) == {"FIT101", "LIT101"}
    assert set(stage2_cols) == {"AIT201", "FIT201"}
    assert set(stage1_cols).isdisjoint(stage2_cols)


def test_apply_stage_features_raises_for_unmatched_stage():
    df = pd.DataFrame({"FIT101": [1.0], "Normal/Attack": ["Normal"]})
    with pytest.raises(ValueError):
        apply_stage_features(df, stage_id=5)  # P6 has no matching tag here


def test_create_sliding_windows_shapes_and_labelling():
    data = np.arange(100 * 4).reshape(100, 4).astype(float)
    labels = np.zeros(100)
    labels[50:55] = 1  # a short anomalous region

    x, y = create_sliding_windows(data, labels, seq_length=15)

    assert x.shape[1:] == (15, 4)
    assert len(x) == len(y)
    assert y.max() == 1  # at least one window overlaps the labelled region
    assert y.min() == 0  # and some windows are purely normal


def test_infer_num_features_never_truncates_widest_stage(tmp_path):
    """
    Regression test: a real dataset can have a stage with more sensors than
    the synthetic default's padding width. The pipeline must size its
    padding to the widest stage actually present, not a fixed guess.
    """
    df = pd.DataFrame({
        "FIT101": [1.0], "LIT101": [1.0],  # stage 1: 2 tags
        "AIT201": [1.0], "FIT201": [1.0], "MV201": [1.0], "P201": [1.0],  # stage 2: 4 tags
        "Normal/Attack": ["Normal"],
    })
    csv_path = tmp_path / "normal.csv"
    df.to_csv(csv_path, index=False)

    assert infer_num_features(str(csv_path)) == 4


@pytest.mark.parametrize("stage_id", range(Config.NUM_CLIENTS))
def test_get_stage_dataloaders_all_stages(stage_id):
    train_loader, test_loader, x_test, y_test, features = get_stage_dataloaders(stage_id)

    assert len(features) > 0
    x_batch, _ = next(iter(train_loader))
    assert x_batch.shape[1:] == (Config.SEQ_LENGTH, Config.NUM_FEATURES)
    assert x_test.shape[1:] == (Config.SEQ_LENGTH, Config.NUM_FEATURES)
    assert not np.isnan(x_test).any()
