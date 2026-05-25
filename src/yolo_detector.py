# ==============================================================================
# MediVigil
# Coder / Lead Developer: Drupad H
# ==============================================================================

import torch
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path: str = "models/yolo11n.pt", device: str = "cuda", half: bool = True):
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
