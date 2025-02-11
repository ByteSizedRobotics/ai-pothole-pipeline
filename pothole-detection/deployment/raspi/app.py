from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pathlib
import platform
import cv2
import torch
import os
import time
import threading
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Global variables
last_saved_time = 0
model = None

# Load the model
def load_model():
    global model

    # NEED THIS OR ELSE DOESN"T WORK ON LINUX
    # Need to force the load to be done on LINUX
    # SOLUTION found on https://github.com/ultralytics/yolov5/issues/12911
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

    model_path = str(Path("/app/models/best_20250210.pt"))
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

    #     model_path = str(Path("/app/models/yolov5s.pt"))
    #     model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# PERFORMS LIVE VIDEO INFERENCE
def live_camera_inference(save_path):
    camera = cv2.VideoCapture(0)  # 0 for default camera

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        results.render()
        frame = results.ims[0]

        # cv2.imshow('YOLOv5 Live', frame)

        threading.Thread(target=save_pictures, args=(frame, results, save_path)).start()

        if cv2.waitKey(1) & 0xFF == ord('q'):  # break the loop when press q
            break

    camera.release()
    cv2.destroyAllWindows()

# SAVES PICTURES EVERY 10 SECONDS IF OBJECT DETECTED WITH CONFIDENCE ABOVE THRESHOLD
def save_pictures(frame, results, save_path):
    CONFIDENCE_THRESHOLD = 0.5

    global last_saved_time
    current_time = time.time()

    if (current_time - last_saved_time >= 10):  # save images every 10 seconds if object detected with confidence above 50%
        detections = results.xyxy[0].cpu().numpy()  # results.xyxy[0] contains all detected objects in format [x1, y1, x2, y2, confidence, class]

        # check if any object has confidence above threshold
        save_image = any(det[4] > CONFIDENCE_THRESHOLD for det in detections)

        if save_image:
            base_path = Path(save_path)
            image_name = Path(f"capture_{int(current_time)}.jpg")
            image_path = base_path / image_name
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")

        last_saved_time = current_time  # Update timestamp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_live_inference', methods=['POST'])
def start_live_inference():
    # USED IF WANT TO SAVE IMAGES TO SPECIFIC FOLDER SPECIFIED BY USER
    # pathSavedImages = request.form.get('pathSavedImages', 'saved_images')
    # path = Path(pathSavedImages)
    # if not path.exists():
    #     path.mkdir(parents=True, exist_ok=True)

    # Save images to default location on container image mount
    save_path = Path(os.environ.get("SAVE_PATH", "/app/saved_images")) # get the save path on the container
    save_path.mkdir(parents=True, exist_ok=True)

    threading.Thread(target=live_camera_inference, args=(save_path,)).start()
    return jsonify({"status": "Live inference started"})

if __name__ == '__main__':
    load_model()
    app.run(host="0.0.0.0", port=5000) # TODO: NATHAN UPDATE so doesn't use flask server for prod