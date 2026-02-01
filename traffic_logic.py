# traffic_logic.py
import cv2
import torch
import numpy as np
import time

# Load YOLOv5 model once
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

vehicle_time_weights = {
    'car': 2,
    'motorcycle': 1.2,
    'bicycle': 1,
    'truck': 3
}

def detect_vehicles_by_type(frame):
    results = model(frame)
    data = results.pandas().xyxy[0]
    vehicle_counts = {}
    for v_type in vehicle_time_weights.keys():
        vehicle_counts[v_type] = int(len(data[data['name'] == v_type]))
    return vehicle_counts

def allocate_green_time(vehicle_counts, min_green=10, max_green=45):
    total_time = 0
    for v_type, count in vehicle_counts.items():
        total_time += count * vehicle_time_weights[v_type]
    green_time = int(np.clip(total_time, min_green, max_green))
    return green_time

def process_video(video_path, cycles=10):
    cap = cv2.VideoCapture(video_path)
    signal_timings = []

    for cycle in range(cycles):
        ret, frame = cap.read()
        if not ret:
            break

        vehicle_counts = detect_vehicles_by_type(frame)
        green_time = allocate_green_time(vehicle_counts)
        entry = {
            "cycle": cycle + 1,
            "vehicle_counts": vehicle_counts,
            "green_time": green_time
        }
        signal_timings.append(entry)

    cap.release()

    # ✅ Only keep the last cycle
    last_entry = signal_timings[-1] if signal_timings else None

    console_summary = ""
    if last_entry:
        console_summary += f"Cycle {last_entry['cycle']}:\n"
        for v_type, count in last_entry['vehicle_counts'].items():
            console_summary += f"Vehicle: {v_type}, Count: {count}\n"
        console_summary += f"Green Light Time Allocated: {last_entry['green_time']} seconds\n"

    return {
        "signal_timings": signal_timings,
        "console_summary": console_summary
    }
