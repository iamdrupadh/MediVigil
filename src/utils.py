# ==============================================================================
# MediVigil
# Coder / Lead Developer: Drupad H
# ==============================================================================

import cv2
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
    # True tension usually brings inner eyebrows closer than the inner eye corners
    return max(0.0, 1.0 - (dist / ref_dist))

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
    elif highest_status == "SLEEPING":
        color = (255, 105, 180)  # Calm Pink for sleeping
    elif highest_status == "CALIBRATING":
        color = (255, 255, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, gpu_stats, (10, 60), font, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"STATUS: {highest_status}", (frame.shape[1] - 400, 30), font, 0.8, color, 2)
    cv2.putText(frame, f"CONFIDENCE: {max_score:.1f}%", (frame.shape[1] - 400, 60), font, 0.6, (200, 255, 255), 1)

    return frame
