# ==============================================================================
# MediVigil
# Coder / Lead Developer: Drupad H
# ==============================================================================

class PatientAnalyzer:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def analyze(self, distress_score, drowsiness_score, agitation_score, breathing_score, help_score, sleeping_score, is_calibrating=False):
        if is_calibrating:
            return "CALIBRATING", "KEEP NEUTRAL FACE", 0.0

        status = "STABLE"
        action = "MONITOR"
        max_score = max(distress_score, drowsiness_score, agitation_score, breathing_score, help_score, sleeping_score)

        if breathing_score >= self.thresholds.get('breathing_difficulty', 80.0):
            status = "BREATHING_DIFFICULTY"
            action = "IMMEDIATE_CHECK"
        elif distress_score >= self.thresholds.get('critical_distress', 85.0):
            status = "CRITICAL_DISTRESS"
            action = "DISPATCH_NURSE"
        elif help_score >= self.thresholds.get('needs_help', 75.0):
            status = "NEEDS_HELP"
            action = "ASSIST_PATIENT"
        elif sleeping_score >= 80.0:
            status = "SLEEPING"
            action = "MONITOR"
        elif drowsiness_score >= self.thresholds.get('drowsiness', 80.0):
            status = "DROWSY"
            action = "LOG_STATE"
        elif agitation_score >= self.thresholds.get('agitation', 75.0):
            status = "AGITATED"
            action = "CALM_PATIENT"

        return status, action, max_score
