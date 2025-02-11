from flask import Flask, render_template, request, jsonify
from pathlib import Path
import cv2
import torch
import time
import threading
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Global variables
last_saved_time = 0
model = None

# Load the model based on the type
def load_model(model_type):
    global model
    if model_type == "custom":
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='/app/models/best_20250202.pt')
    elif model_type == "yolo5s":
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='/app/models/yolov5s.pt')

# PERFORMS LIVE VIDEO INFERENCE
def live_camera_inference(pathSavedImages):
    camera = cv2.VideoCapture(0)  # 0 for default camera

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        results.render()
        frame = results.ims[0]

        cv2.imshow('YOLOv5 Live', frame)

        threading.Thread(target=save_pictures, args=(frame, results, pathSavedImages)).start()

        if cv2.waitKey(1) & 0xFF == ord('q'):  # break the loop when press q
            break

    camera.release()
    cv2.destroyAllWindows()

# SAVES PICTURES EVERY 10 SECONDS IF OBJECT DETECTED WITH CONFIDENCE ABOVE THRESHOLD
def save_pictures(frame, results, pathSavedImages):
    CONFIDENCE_THRESHOLD = 0.5

    global last_saved_time
    current_time = time.time()

    if (current_time - last_saved_time >= 10):  # save images every 10 seconds if object detected with confidence above 50%
        detections = results.xyxy[0].cpu().numpy()  # results.xyxy[0] contains all detected objects in format [x1, y1, x2, y2, confidence, class]

        # check if any object has confidence above threshold
        save_image = any(det[4] > CONFIDENCE_THRESHOLD for det in detections)

        if save_image:
            base_path = Path(pathSavedImages)
            image_name = Path(f"capture_{int(current_time)}.jpg")
            filename = base_path / image_name
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")

        last_saved_time = current_time  # Update timestamp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_live_inference', methods=['POST'])
def start_live_inference():
    pathSavedImages = request.form.get('pathSavedImages', 'saved_images')
    if not os.path.exists(pathSavedImages):
        os.makedirs(pathSavedImages)

    threading.Thread(target=live_camera_inference, args=(pathSavedImages,)).start()
    return jsonify({"status": "Live inference started"})

if __name__ == '__main__':
    load_model("custom")  # TODO: CHANGE THE MODEL U WANT TO LOAD HERE
    app.run(host="0.0.0.0", port=5000)