import numpy as np

from config import Config
from models import get_model
from xai_explainer import XAIExplainer


def test_explain_anomaly_returns_requested_top_k_features():
    model = get_model()
    baseline = np.random.randn(20, Config.SEQ_LENGTH, Config.NUM_FEATURES).astype(np.float32)
    explainer = XAIExplainer(model, baseline)

    anomalous = np.random.randn(1, Config.SEQ_LENGTH, Config.NUM_FEATURES).astype(np.float32)
    features_list = [f"F{i}" for i in range(Config.NUM_FEATURES)]

    top_features = explainer.explain_anomaly(anomalous, features_list, top_k=3)

    assert len(top_features) == 3
    assert all(f in features_list for f in top_features)
