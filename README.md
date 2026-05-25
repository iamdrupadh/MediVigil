# MediVigil: Hospital Patient Facial Monitoring System
**Coder / Lead Developer:** Drupad H

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
*(You can also use `python -m src.main` to avoid import path issues)*

### 3. Launch the Nurse Command Post
In a new terminal, launch the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

## License
MIT License.
