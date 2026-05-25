import pytest
from src.neurosync_engine import NeuroSyncEngine
from src.defense_analyzer import DefenseAnalyzer
from src.liveness_detector import KinematicLivenessDetector

def test_neurosync_warmup():
    engine = NeuroSyncEngine(window_size=10)
    # Give it one frame
    engine.update("subj_1", {'EAR': 0.2, 'MAR': 0.1}, 70, 50, (0.0, 0.0, 0.0), 80.0)
    
    # Should return 0,0 because it needs warmup (window/2 frames)
    prob, cog = engine.predict_pre_threat("subj_1")
    assert prob == 0.0
    assert cog == 0.0

def test_neurosync_high_threat():
    engine = NeuroSyncEngine(window_size=10)
    # Simulate high stress, high pose volatility, high mouth movement
    for i in range(10):
        engine.update(
            "subj_1", 
            {'EAR': 0.1, 'MAR': 0.5 + (i%2)*0.4}, # Fluctuating MAR
            120, 
            90, # High stress
            (float(i*10), 0.0, 0.0), # Moving head fast
            80.0
        )
    prob, cog = engine.predict_pre_threat("subj_1")
    assert prob > 50.0 # Should be elevated/critical
    assert cog > 40.0

def test_defense_analyzer_spoof():
    analyzer = DefenseAnalyzer()
    status, action = analyzer.analyze(pre_threat_prob=10.0, cog_load=10.0, liveness_score=20.0)
    assert status == "SPOOF_DETECTED"
    assert action == "VERIFY_LIVENESS"

def test_defense_analyzer_threat():
    analyzer = DefenseAnalyzer(pre_threat_threshold=75.0)
    status, action = analyzer.analyze(pre_threat_prob=80.0, cog_load=50.0, liveness_score=90.0)
    assert status == "CRITICAL_THREAT"
    assert action == "INTERCEPT"
