import numpy as np

from config import Config
from threat_intelligence import ThreatIntelligence


def test_ewma_rises_with_sustained_error_and_crosses_critical_threshold():
    ti = ThreatIntelligence()
    scores = []
    for err in [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0]:
        score, _ = ti.calculate_early_warning_score(np.array([err]))
        scores.append(score)

    assert scores == sorted(scores)  # monotonically rising while error stays high
    assert scores[-1] > Config.EARLY_WARNING_CRITICAL_SCORE


def test_ewma_flags_critical_using_configured_threshold():
    ti = ThreatIntelligence()
    ti.ewma_score = Config.EARLY_WARNING_CRITICAL_SCORE + 0.01
    _, is_critical = ti.calculate_early_warning_score(np.array([ti.ewma_score]))
    assert is_critical is True


def test_stride_mitre_mapping_covers_known_and_unknown_prefixes():
    alerts = ThreatIntelligence.map_to_mitre_and_stride(
        ["FIT101", "LIT101", "MV101", "P101", "AIT201", "ZZZ999"], stage_id=0
    )
    assert len(alerts) == 6
    for alert in alerts:
        assert alert["Stage"] == "P1"
        assert alert["STRIDE Threat"]
        assert alert["MITRE Class"]
