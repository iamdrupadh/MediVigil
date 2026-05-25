# ==============================================================================
# MediVigil
# Coder / Lead Developer: Drupad H
# ==============================================================================

import argparse
import cv2
import time
import yaml
import torch
import sys
import os
import mediapipe as mp
import scipy.spatial.distance as distance

# Fix import path automatically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    # Move to absolute path of config for easier execution
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(root_dir, "config.yaml")
    config = load_config(config_path)
    
    target_device = args.device if args.device else config['hardware']['device']
    device = get_device(target_device)
    use_half = config['hardware']['half_precision']
    
    print(f"--- MediVigil Initializing on {device} ---")
    
    yolo_model_path = os.path.join(root_dir, config['detection']['yolo_model'])
    yolo = YoloDetector(model_path=yolo_model_path, device=str(device), half=use_half)
    
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
    
    log_dir_path = os.path.join(root_dir, "logs")
    alert_system = AlertSystem(log_dir=log_dir_path)

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
                mouth_width = distance.euclidean(mouth[0], mouth[4])
                
                eyebrow_inner_left = (landmarks[107].x*w, landmarks[107].y*h)
                eyebrow_inner_right = (landmarks[336].x*w, landmarks[336].y*h)
                eye_dist = distance.euclidean(l_eye[3], r_eye[0])
                smile_ratio = mouth_width / eye_dist if eye_dist > 0 else 0
                
                tension = calculate_eyebrow_tension(eyebrow_inner_left, eyebrow_inner_right, eye_dist)
                
                kinematics = {
                    'EAR': ear, 'MAR': mar, 
                    'Mouth_Asym': mouth_asym, 'Eyebrow_Tension': tension,
                    'Smile_Ratio': smile_ratio
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
                
                box_color = (255, 255, 0) if status == "CALIBRATING" else (255, 165, 0)
                cv2.rectangle(frame, (int(face_bbox[0]), int(face_bbox[1])), (int(face_bbox[2]), int(face_bbox[3])), box_color, 2)
                
                text = f"CALIBRATING: KEEP STILL" if status == "CALIBRATING" else f"ID:{patient_id} {status}"
                text_color = (255, 255, 0) if status == "CALIBRATING" else (0, 255, 255)
                cv2.putText(frame, text, (int(face_bbox[0]), int(face_bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        gpu_stats = get_gpu_memory_usage()
        frame = draw_medical_hud(frame, fps, max_score_global, highest_status_global, gpu_stats)
        
        cv2.imshow("MediVigil - Bedside Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()
