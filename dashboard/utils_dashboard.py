# ==============================================================================
# MediVigil
# Coder / Lead Developer: Drupad H
# ==============================================================================

import os
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
            gpu_util = result.stdout.strip().split('\n')[0]
            gpu_info = f"{gpu_util}%"
    except Exception:
        gpu_info = "Not Found/CPU Mode"
        
    return cpu, gpu_info
