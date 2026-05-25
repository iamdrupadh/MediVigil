# ==============================================================================
# MediVigil
# Coder / Lead Developer: Drupad H
# ==============================================================================

import csv
import json
import os
import time
from datetime import datetime

class AlertSystem:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file = os.path.join(self.log_dir, f"medivigil_log_{int(time.time())}.csv")
        self.json_file = os.path.join(self.log_dir, f"medivigil_log_{int(time.time())}.json")
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'patient_id', 'status', 'confidence', 'action'])

    def trigger_alert(self, patient_id, status, confidence, action, raw_data=None):
        timestamp = datetime.now().isoformat()
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, patient_id, status, f"{confidence:.2f}", action])
            
        if status not in ["STABLE", "NORMAL"]:
            log_entry = {
                "timestamp": timestamp,
                "patient_id": patient_id,
                "status": status,
                "metrics": {
                    "confidence": confidence,
                    "action_required": action
                },
                "raw_mediguard_data": raw_data
            }
            with open(self.json_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
                
            self._play_sound(status)

    def _play_sound(self, status):
        try:
            import winsound
            if status in ["CRITICAL_DISTRESS", "BREATHING_DIFFICULTY", "NEEDS_HELP"]:
                winsound.Beep(1000, 500)
            elif status in ["DROWSY", "AGITATED"]:
                winsound.Beep(800, 300)
        except Exception:
            pass
