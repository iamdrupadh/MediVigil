# ==============================================================================
# MediVigil
# Coder / Lead Developer: Drupad H
# ==============================================================================

import numpy as np
from collections import deque

class MediGuardEngine:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.patient_histories = {}
        
    def _ensure_patient(self, patient_id):
        if patient_id not in self.patient_histories:
            self.patient_histories[patient_id] = {
                'kinematics': deque(maxlen=self.window_size),
                'pose': deque(maxlen=self.window_size),
                'baseline': None,
                'calibrating_frames': 0
            }

    def update(self, patient_id, kinematics_data, pose):
        self._ensure_patient(patient_id)
        self.patient_histories[patient_id]['kinematics'].append(kinematics_data)
        self.patient_histories[patient_id]['pose'].append(pose)
        self.patient_histories[patient_id]['calibrating_frames'] += 1

    def analyze_patient(self, patient_id):
        if patient_id not in self.patient_histories:
            return 0.0, 0.0, 0.0, 0.0, 0.0, False
            
        hist = self.patient_histories[patient_id]
        
        # Require a full window (e.g. 30 frames) to calibrate to the user's specific face
        if hist['calibrating_frames'] < self.window_size:
            return 0.0, 0.0, 0.0, 0.0, 0.0, True
            
        ear_vals = [k.get('EAR', 0) for k in hist['kinematics']]
        mar_vals = [k.get('MAR', 0) for k in hist['kinematics']]
        asym_vals = [k.get('Mouth_Asym', 0) for k in hist['kinematics']]
        tension_vals = [k.get('Eyebrow_Tension', 0) for k in hist['kinematics']]
        smile_vals = [k.get('Smile_Ratio', 0) for k in hist['kinematics']]
        pose_arr = np.array(list(hist['pose']))
        
        # Calculate baseline once
        if hist['baseline'] is None:
            hist['baseline'] = {
                'EAR': np.mean(ear_vals),
                'MAR': np.mean(mar_vals),
                'Asym': np.mean(asym_vals),
                'Tension': np.mean(tension_vals)
            }
            
        base = hist['baseline']
        
        avg_smile = np.mean(smile_vals[-5:]) if len(smile_vals) >= 5 else 0
        is_smiling = avg_smile > 1.45
        
        # 1. Drowsiness / Unconsciousness
        recent_ears = ear_vals[-5:] if len(ear_vals) >= 5 else ear_vals
        # Awake if EAR is above 80% of THEIR normal open eye EAR
        awake_threshold = base['EAR'] * 0.8
        is_awake = np.mean(recent_ears) > awake_threshold
        
        if is_awake:
            drowsiness_score = 0.0
            sleeping_score = 0.0
        else:
            # Closed if EAR drops below 65% of THEIR normal open eye EAR
            closed_threshold = base['EAR'] * 0.65
            perclos = np.sum([1 for e in ear_vals if e < closed_threshold]) / len(ear_vals)
            max_continuous_closure = 0
            current_closure = 0
            for e in ear_vals:
                if e < closed_threshold:
                    current_closure += 1
                    max_continuous_closure = max(max_continuous_closure, current_closure)
                else:
                    current_closure = 0
            
            drowsiness_score = perclos * 100.0  # High ratio of closed eyes over time
            sleeping_score = min(max_continuous_closure / 15.0, 1.0) * 100.0  # Fully shut for 15+ frames
        
        # 2. Pain / Distress
        avg_asym = np.mean(asym_vals)
        avg_tension = np.mean(tension_vals)
        
        # Calculate differences purely based on THEIR natural resting face
        asym_diff = max(0, avg_asym - (base['Asym'] + 0.05))
        tension_diff = max(0, avg_tension - (base['Tension'] + 0.1))
        
        distress_score = min(100.0, (asym_diff * 300) + (tension_diff * 200))
        
        if is_smiling:
            distress_score = 0.0
        
        # 3. Agitation / Confusion (High-frequency rapid movement)
        if len(pose_arr) > 1:
            # Normal activity (looking around) changes angles smoothly.
            # Agitation (thrashing) changes angles violently frame-to-frame.
            pose_diffs = np.abs(np.diff(pose_arr, axis=0))
            avg_total_vel = np.sum(np.mean(pose_diffs, axis=0))
            
            # Noise/Normal movement is usually < 5 degrees per frame.
            # We set a high threshold (8.0) so only erratic/violent movement triggers it.
            agitation_score = min(100.0, max(0, avg_total_vel - 8.0) * 12.0)
        else:
            agitation_score = 0.0
        
        # 4. Breathing Difficulty
        # High MAR relative to THEIR baseline mouth openness
        high_mar_threshold = base['MAR'] + 0.2
        high_mar_ratio = np.sum([1 for m in mar_vals if m > high_mar_threshold]) / len(mar_vals)
        mar_volatility = np.var(mar_vals) if len(mar_vals) > 0 else 0
        
        breathing_score = min(100.0, (high_mar_ratio * 100 * 0.7) + (max(0, mar_volatility - 0.05) * 500 * 0.3))
        
        # 5. Needs Help
        mouth_cycles = 0
        for i in range(1, len(mar_vals)):
            if mar_vals[i] > high_mar_threshold and mar_vals[i-1] <= high_mar_threshold:
                mouth_cycles += 1
        help_score = min(100.0, mouth_cycles * 20)
        
        return distress_score, drowsiness_score, agitation_score, breathing_score, help_score, sleeping_score, False

    def get_orbit_data(self, patient_id):
        if patient_id not in self.patient_histories:
            return []
        return list(self.patient_histories[patient_id]['pose'])
