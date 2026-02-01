import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
        vehicle_counts[v_type] = len(data[data['name'] == v_type])
    return vehicle_counts

def allocate_green_time(vehicle_counts, min_green=10, max_green=45):
    total_time = 0
    for v_type, count in vehicle_counts.items():
        total_time += count * vehicle_time_weights[v_type]
    green_time = int(np.clip(total_time, min_green, max_green))
    return green_time

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        signal_timings = process_video(filepath)

        return render_template('result.html', timings=signal_timings)

    return render_template('index.html')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    signal_timings = []
    cycles = 10

    for cycle in range(cycles):
        ret, frame = cap.read()
        if not ret:
            break
        vehicle_counts = detect_vehicles_by_type(frame)
        green_time = allocate_green_time(vehicle_counts)
        signal_timings.append({
            'cycle': cycle+1,
            'vehicles': vehicle_counts,
            'green_time': green_time
        })
    cap.release()
    return signal_timings

if __name__ == '__main__':
    app.run(debug=True)
