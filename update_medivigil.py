import os

files = {
    "README.md": """# MediVigil: Hospital Patient Facial Monitoring System

**MediVigil** is a real-time hospital bedside monitoring system. It fuses multi-modal facial dynamics and kinematics to track patient well-being, detecting distress, drowsiness, breathing difficulties, and agitation with high accuracy and minimal light dependency.

## Features
- **MediGuard Engine**: Advanced temporal smoothing (30 frames) for high-accuracy assessment of:
  1. Drowsiness / Unconsciousness (PERCLOS & long eye closure).
  2. Pain / Distress (Mouth asymmetry, eyebrow tension).
  3. Agitation / Confusion (High head pose volatility).
  4. Breathing Difficulty (Prolonged mouth opening, abnormal kinematics).
  5. Needs Help (Repeated mouth opening).
- **Hardware-First**: Auto-detects NVIDIA RTX GPUs (CUDA, FP16) for real-time processing.
- **Robust Architecture**: YOLO11 (CUDA) + MediaPipe Face Mesh.
- **Nurse Command Post**: Streamlit-based dashboard for real-time multi-patient monitoring.

## Run Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Main System
Launch the bedside monitoring engine:
```bash
python src/main.py --device cuda
```

### 3. Launch the Nurse Command Post
In a new terminal, launch the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

## License
MIT License.
""",
    "config.yaml": """# MediVigil Configuration (Hospital Edition)
hardware:
  gpu_enabled: true
  device: "cuda"
  half_precision: true

detection:
  yolo_model: "models/yolo11n.pt"
  max_faces: 4

mediguard:
  window_size: 30
  fps_target: 60

thresholds:
  critical_distress: 85.0
  drowsiness: 80.0
  agitation: 75.0
  breathing_difficulty: 80.0
  needs_help: 75.0
""",
    "src/utils.py": """import cv2
import numpy as np
import torch
from scipy.spatial import distance

def get_device(requested_device: str = "cuda") -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_gpu_memory_usage() -> str:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return f"Mem: {allocated:.1f}MB/{reserved:.1f}MB"
    return "CPU Mode"

def calculate_ear(eye_landmarks: list) -> float:
    if len(eye_landmarks) != 6:
        return 0.0
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def calculate_mar(mouth_landmarks: list) -> float:
    if len(mouth_landmarks) < 6:
        return 0.0
    A = distance.euclidean(mouth_landmarks[1], mouth_landmarks[7])
    B = distance.euclidean(mouth_landmarks[2], mouth_landmarks[6])
    C = distance.euclidean(mouth_landmarks[3], mouth_landmarks[5])
    D = distance.euclidean(mouth_landmarks[0], mouth_landmarks[4])
    if D == 0:
        return 0.0
    return (A + B + C) / (3.0 * D)

def calculate_mouth_asymmetry(mouth_landmarks: list) -> float:
    if len(mouth_landmarks) < 6:
        return 0.0
    center_x = (mouth_landmarks[2][0] + mouth_landmarks[6][0]) / 2.0
    center_y = (mouth_landmarks[2][1] + mouth_landmarks[6][1]) / 2.0
    center = (center_x, center_y)
    dist_left = distance.euclidean(mouth_landmarks[0], center)
    dist_right = distance.euclidean(mouth_landmarks[4], center)
    if dist_left + dist_right == 0:
        return 0.0
    return abs(dist_left - dist_right) / ((dist_left + dist_right) / 2.0)

def calculate_eyebrow_tension(eyebrow_inner_left: tuple, eyebrow_inner_right: tuple, ref_dist: float) -> float:
    dist = distance.euclidean(eyebrow_inner_left, eyebrow_inner_right)
    if ref_dist == 0:
        return 0.0
    return max(0.0, 1.0 - (dist / (ref_dist * 1.5)))

def get_landmark_point(landmark, image_shape) -> tuple[int, int]:
    h, w, _ = image_shape
    return int(landmark.x * w), int(landmark.y * h)

def draw_medical_hud(frame, fps, max_score, highest_status, gpu_stats):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (15, 25, 35), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    color = (0, 255, 0)
    if highest_status in ["CRITICAL_DISTRESS", "BREATHING_DIFFICULTY", "NEEDS_HELP"]:
        color = (0, 0, 255)
    elif highest_status in ["DROWSY", "AGITATED"]:
        color = (0, 165, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, gpu_stats, (10, 60), font, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"STATUS: {highest_status}", (frame.shape[1] - 400, 30), font, 0.8, color, 2)
    cv2.putText(frame, f"CONFIDENCE: {max_score:.1f}%", (frame.shape[1] - 400, 60), font, 0.6, (200, 255, 255), 1)

    return frame
""",
    "src/patient_analyzer.py": """class PatientAnalyzer:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def analyze(self, distress_score, drowsiness_score, agitation_score, breathing_score, help_score):
        status = "STABLE"
        action = "MONITOR"
        max_score = max(distress_score, drowsiness_score, agitation_score, breathing_score, help_score)

        if breathing_score >= self.thresholds.get('breathing_difficulty', 80.0):
            status = "BREATHING_DIFFICULTY"
            action = "IMMEDIATE_CHECK"
        elif distress_score >= self.thresholds.get('critical_distress', 85.0):
            status = "CRITICAL_DISTRESS"
            action = "DISPATCH_NURSE"
        elif help_score >= self.thresholds.get('needs_help', 75.0):
            status = "NEEDS_HELP"
            action = "ASSIST_PATIENT"
        elif drowsiness_score >= self.thresholds.get('drowsiness', 80.0):
            status = "DROWSY"
            action = "LOG_STATE"
        elif agitation_score >= self.thresholds.get('agitation', 75.0):
            status = "AGITATED"
            action = "CALM_PATIENT"

        return status, action, max_score
""",
    "src/mediguard_engine.py": """import numpy as np
from collections import deque

class MediGuardEngine:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.patient_histories = {}
        
    def _ensure_patient(self, patient_id):
        if patient_id not in self.patient_histories:
            self.patient_histories[patient_id] = {
                'kinematics': deque(maxlen=self.window_size),
                'pose': deque(maxlen=self.window_size)
            }

    def update(self, patient_id, kinematics_data, pose):
        self._ensure_patient(patient_id)
        self.patient_histories[patient_id]['kinematics'].append(kinematics_data)
        self.patient_histories[patient_id]['pose'].append(pose)

    def analyze_patient(self, patient_id):
        if patient_id not in self.patient_histories:
            return 0.0, 0.0, 0.0, 0.0, 0.0
            
        hist = self.patient_histories[patient_id]
        if len(hist['kinematics']) < self.window_size // 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0
            
        ear_vals = [k.get('EAR', 0) for k in hist['kinematics']]
        mar_vals = [k.get('MAR', 0) for k in hist['kinematics']]
        asym_vals = [k.get('Mouth_Asym', 0) for k in hist['kinematics']]
        tension_vals = [k.get('Eyebrow_Tension', 0) for k in hist['kinematics']]
        pose_arr = np.array(list(hist['pose']))
        
        # 1. Drowsiness / Unconsciousness (PERCLOS + Eye Closure)
        perclos = np.sum([1 for e in ear_vals if e < 0.22]) / len(ear_vals)
        max_continuous_closure = 0
        current_closure = 0
        for e in ear_vals:
            if e < 0.22:
                current_closure += 1
                max_continuous_closure = max(max_continuous_closure, current_closure)
            else:
                current_closure = 0
        drowsiness_score = (perclos * 100) * 0.5 + (min(max_continuous_closure / self.window_size, 1.0) * 100) * 0.5
        
        # 2. Pain / Distress (Mouth Asymmetry + Eyebrow Tension)
        avg_asym = np.mean(asym_vals)
        avg_tension = np.mean(tension_vals)
        distress_score = min(100.0, (avg_asym * 150) + (avg_tension * 100))
        
        # 3. Agitation / Confusion (Head pose volatility)
        pose_volatility = np.sum(np.var(pose_arr, axis=0)) if len(pose_arr) > 0 else 0
        agitation_score = min(100.0, pose_volatility * 40)
        
        # 4. Breathing Difficulty (Prolonged open mouth)
        high_mar_ratio = np.sum([1 for m in mar_vals if m > 0.4]) / len(mar_vals)
        mar_volatility = np.var(mar_vals) if len(mar_vals) > 0 else 0
        breathing_score = min(100.0, (high_mar_ratio * 100 * 0.6) + (mar_volatility * 500 * 0.4))
        
        # 5. Needs Help (Repeated mouth opening)
        mouth_cycles = 0
        for i in range(1, len(mar_vals)):
            if mar_vals[i] > 0.4 and mar_vals[i-1] <= 0.4:
                mouth_cycles += 1
        help_score = min(100.0, mouth_cycles * 25)
        
        return distress_score, drowsiness_score, agitation_score, breathing_score, help_score

    def get_orbit_data(self, patient_id):
        if patient_id not in self.patient_histories:
            return []
        return list(self.patient_histories[patient_id]['pose'])
""",
    "src/alert_system.py": """import csv
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
                f.write(json.dumps(log_entry) + "\\n")
                
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
""",
    "src/main.py": """import argparse
import cv2
import time
import yaml
import torch
import mediapipe as mp
import scipy.spatial.distance as distance

from src.yolo_detector import YoloDetector
from src.head_pose import HeadPoseEstimator
from src.mediguard_engine import MediGuardEngine
from src.patient_analyzer import PatientAnalyzer
from src.alert_system import AlertSystem
from src.utils import (get_device, get_gpu_memory_usage, calculate_ear, 
                       calculate_mar, calculate_mouth_asymmetry, calculate_eyebrow_tension, 
                       draw_medical_hud)

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_main():
    parser = argparse.ArgumentParser(description="MediVigil CLI")
    parser.add_argument("--device", type=str, default="cuda", help="Force device (cuda/cpu)")
    args = parser.parse_args()

    config = load_config()
    target_device = args.device if args.device else config['hardware']['device']
    device = get_device(target_device)
    use_half = config['hardware']['half_precision']
    
    print(f"--- MediVigil Initializing on {device} ---")
    
    yolo = YoloDetector(model_path=config['detection']['yolo_model'], device=str(device), half=use_half)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=config['detection']['max_faces'],
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    pose_estimator = HeadPoseEstimator()
    medi_engine = MediGuardEngine(window_size=config['mediguard']['window_size'])
    analyzer = PatientAnalyzer(thresholds=config['thresholds'])
    alert_system = AlertSystem()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    target_fps = config['mediguard']['fps_target']
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-9)
        prev_time = current_time

        detections = yolo.detect(frame)
        
        max_score_global = 0.0
        highest_status_global = "STABLE"
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                patient_id = f"patient_{idx}"
                landmarks = face_landmarks.landmark
                h, w, _ = frame.shape
                
                pose_pts = [
                    (landmarks[1].x * w, landmarks[1].y * h),
                    (landmarks[152].x * w, landmarks[152].y * h),
                    (landmarks[33].x * w, landmarks[33].y * h),
                    (landmarks[263].x * w, landmarks[263].y * h),
                    (landmarks[61].x * w, landmarks[61].y * h),
                    (landmarks[291].x * w, landmarks[291].y * h)
                ]
                
                yaw, pitch, roll = pose_estimator.estimate_pose(frame, pose_pts)
                
                l_eye = [(landmarks[i].x*w, landmarks[i].y*h) for i in [33, 160, 158, 133, 153, 144]]
                r_eye = [(landmarks[i].x*w, landmarks[i].y*h) for i in [362, 385, 387, 263, 373, 380]]
                ear = (calculate_ear(l_eye) + calculate_ear(r_eye)) / 2.0
                
                mouth = [(landmarks[i].x*w, landmarks[i].y*h) for i in [78, 81, 13, 311, 308, 402, 14, 178]]
                mar = calculate_mar(mouth)
                
                mouth_asym = calculate_mouth_asymmetry(mouth)
                
                eyebrow_inner_left = (landmarks[107].x*w, landmarks[107].y*h)
                eyebrow_inner_right = (landmarks[336].x*w, landmarks[336].y*h)
                eye_dist = distance.euclidean(l_eye[3], r_eye[0])
                tension = calculate_eyebrow_tension(eyebrow_inner_left, eyebrow_inner_right, eye_dist)
                
                kinematics = {
                    'EAR': ear, 'MAR': mar, 
                    'Mouth_Asym': mouth_asym, 'Eyebrow_Tension': tension
                }
                
                medi_engine.update(patient_id, kinematics, (yaw, pitch, roll))
                scores = medi_engine.analyze_patient(patient_id)
                status, action, max_score = analyzer.analyze(*scores)
                
                if max_score > max_score_global:
                    max_score_global = max_score
                    highest_status_global = status
                
                if status != "STABLE":
                    raw_data = str(list(medi_engine.patient_histories[patient_id]['pose']))
                    alert_system.trigger_alert(patient_id, status, max_score, action, raw_data)
                
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                face_bbox = [min(xs), min(ys), max(xs), max(ys)]
                cv2.rectangle(frame, (int(face_bbox[0]), int(face_bbox[1])), (int(face_bbox[2]), int(face_bbox[3])), (255, 165, 0), 2)
                cv2.putText(frame, f"ID:{patient_id} {status}", (int(face_bbox[0]), int(face_bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        gpu_stats = get_gpu_memory_usage()
        frame = draw_medical_hud(frame, fps, max_score_global, highest_status_global, gpu_stats)
        
        cv2.imshow("MediVigil - Bedside Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()
""",
    "dashboard/app.py": """import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import json
from utils_dashboard import parse_logs, get_system_stats

st.set_page_config(page_title="Nurse Command Post", layout="wide", initial_sidebar_state="expanded")

st.markdown(\"\"\"
<style>
    .reportview-container { background: #0E1117; }
    .stMetric .metric-value { color: #00FF00 !important; }
    .critical-alert { color: #FF0000; font-weight: bold; animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
\"\"\", unsafe_allow_html=True)

st.title("🏥 MediVigil Nurse Command Post")
st.markdown("### Real-Time Hospital Patient Facial Monitoring System")

with st.sidebar:
    st.header("Control Panel")
    log_dir = st.text_input("Log Directory", "../logs")
    refresh_rate = st.slider("Refresh Rate (sec)", 1, 10, 2)
    
    st.markdown("---")
    st.subheader("System Status")
    cpu, gpu = get_system_stats()
    st.metric("CPU Usage", f"{cpu}%")
    st.metric("GPU Usage", gpu)
    
df_logs, recent_json = parse_logs(log_dir)

if df_logs is not None and not df_logs.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events Analyzed", len(df_logs))
    with col2:
        criticals = len(df_logs[df_logs['status'].isin(['CRITICAL_DISTRESS', 'BREATHING_DIFFICULTY', 'NEEDS_HELP'])])
        if criticals > 0:
            st.markdown(f"### <div class='critical-alert'>Urgent Alerts: {criticals}</div>", unsafe_allow_html=True)
        else:
            st.metric("Urgent Alerts", 0)
    with col3:
        drowsy = len(df_logs[df_logs['status'] == 'DROWSY'])
        st.metric("Drowsy/Unconscious", drowsy)
    with col4:
         max_conf = df_logs['confidence'].max()
         st.metric("Max Alert Confidence", f"{max_conf:.1f}%")

    st.markdown("---")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Alert Timeline")
        fig = px.line(df_logs, x='timestamp', y='confidence', color='patient_id', markers=True,
                      color_discrete_sequence=px.colors.qualitative.Set1)
        fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Status Distribution")
        fig2 = px.pie(df_logs, names='status', title='Patient Alerts by Category')
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Head Movement Orbit (Latest Critical Event)")
    
    if recent_json and 'raw_mediguard_data' in recent_json and recent_json['raw_mediguard_data']:
        try:
            pose_data = eval(recent_json['raw_mediguard_data'])
            if pose_data:
                yaws = [p[0] for p in pose_data]
                pitches = [p[1] for p in pose_data]
                rolls = [p[2] for p in pose_data]
                
                fig3 = go.Figure(data=[go.Scatter3d(
                    x=yaws, y=pitches, z=rolls,
                    mode='lines+markers',
                    marker=dict(size=4, color=yaws, colorscale='Viridis', opacity=0.8),
                    line=dict(color='darkblue', width=2)
                )])
                fig3.update_layout(title="6DoF Head Trajectory", template="plotly_dark")
                st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Could not parse orbit data: {e}")
    else:
        st.info("No 3D orbit data available from recent critical events.")
        
    st.subheader("Event Log Database")
    st.dataframe(df_logs.tail(20), use_container_width=True)
    
else:
    st.warning(f"No log files found in {log_dir}. Run the main engine to generate data.")

st.button("Manual Refresh")
""",
    "dashboard/utils_dashboard.py": """import os
import glob
import pandas as pd
import json
import psutil
import subprocess

def parse_logs(log_dir):
    if not os.path.exists(log_dir):
        return None, None
        
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    json_files = glob.glob(os.path.join(log_dir, "*.json"))
    
    df = None
    recent_json = None
    
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        try:
            df = pd.read_csv(latest_csv)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            pass
            
    if json_files:
        latest_json = max(json_files, key=os.path.getctime)
        try:
            with open(latest_json, 'r') as f:
                lines = f.readlines()
                if lines:
                    recent_json = json.loads(lines[-1])
        except Exception:
            pass
            
    return df, recent_json

def get_system_stats():
    cpu = psutil.cpu_percent()
    gpu_info = "N/A"
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True, check=True
        )
        if result.stdout:
            gpu_util = result.stdout.strip().split('\\n')[0]
            gpu_info = f"{gpu_util}%"
    except Exception:
        gpu_info = "Not Found/CPU Mode"
        
    return cpu, gpu_info
""",
    "src/head_pose.py": """import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             
            (0.0, -330.0, -65.0),        
            (-225.0, 170.0, -135.0),     
            (225.0, 170.0, -135.0),      
            (-150.0, -150.0, -125.0),    
            (150.0, -150.0, -125.0)      
        ])

    def estimate_pose(self, image, landmarks2d):
        if len(landmarks2d) != 6:
            return 0.0, 0.0, 0.0

        size = image.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype = "double"
        )

        dist_coeffs = np.zeros((4,1))
        image_points = np.array(landmarks2d, dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        return yaw, pitch, roll
""",
    "src/yolo_detector.py": """import torch
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path: str = "yolo11n.pt", device: str = "cuda", half: bool = True):
        self.device = device
        self.half = half
        
        self.model = YOLO(model_path)
        
        if self.device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
            if self.half:
                self.model.half()
            print(f"YOLO initialized on CUDA, FP16: {self.half}")
        else:
            print("YOLO initialized on CPU")

    def detect(self, frame):
        results = self.model(frame, verbose=False, classes=[0], device=self.device, half=self.half)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                detections.append({
                    "bbox": b,
                    "confidence": conf,
                    "class": int(cls)
                })
        return detections
"""
}

import shutil
for f_name, content in files.items():
    with open(f_name, "w", encoding="utf-8") as f:
        f.write(content)

# Clean up old files
old_files = ["src/rppg_pulse.py", "src/liveness_detector.py", "src/defense_analyzer.py", "src/neurosync_engine.py"]
for f in old_files:
    if os.path.exists(f):
        os.remove(f)
