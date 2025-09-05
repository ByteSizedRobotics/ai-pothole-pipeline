import os
import cv2
import torch
import argparse
import time
import threading

last_saved_time = 0

def load_model():
    global model
    # if model_type == "custom":
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../../train-runs/2025-03-01_combined1.1/run/weights/best.pt')
    # elif model_type == "yolo5s":
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # elif model_type == "yolo5m":
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    # elif model_type == "yolo5l":
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    # elif model_type == "yolo5x":
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# PERFORMS LIVE VIDEO INFERENCE
def live_camera_inference(pathSavedImages):
    camera = cv2.VideoCapture(0)  # 0 for  default camera

    while True:
        success, frame = camera.read()
        if not success:
            break

        start_time = time.time()
        results = model(frame)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.4f} seconds")

        results.render()
        frame = results.ims[0]

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('YOLOv5 Live', frame)

        threading.Thread(target=save_pictures, args=(frame, results, pathSavedImages)).start()

        if cv2.waitKey(1) & 0xFF == ord('q'): # break the loop when press q
            break

    camera.release()
    cv2.destroyAllWindows()

# SAVES PICTURES EVERY 10 SECONDS IF OBJECT DETECTED WITH CONFIDENCE ABOVE THRESHOLD
def save_pictures(frame, results, pathSavedImages):
    CONFIDENCE_THRESHOLD = 0.5

    global last_saved_time
    current_time = time.time()

    if (current_time - last_saved_time >= 10): # save images every 10 seconds if object detected with confidence above 50% 
        detections = results.xyxy[0].cpu().numpy() # results.xyxy[0] contains all detected objects in format [x1, y1, x2, y2, confidence, class]

        # check if any object has confidence above threshold
        save_image = any(det[4] > CONFIDENCE_THRESHOLD for det in detections)

        if (save_image):
            filename = os.path.join(pathSavedImages, f"capture_{int(current_time)}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")
        
        last_saved_time = current_time  # Update timestamp

# PERFORMS INFERENCE ON IMAGES IN A FOLDER
def image_inference(dataset_path, model):
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            
            results = model(image)
        
            if results.xyxy[0].shape[0] > 0:
                results.show()
    cv2.destroyAllWindows()

# Testing 
parser = argparse.ArgumentParser(description="")

# parser.add_argument("--modelType", type=str, help="Select Model: custom, yolo5s, yolo5m, yolo5l, yolo5x")
parser.add_argument("--mode", type=int, help="1 = image inference, 0 = video inference")
parser.add_argument("--pathImg", type=str, help="Path to image folder, only needed if image inference was selected")
parser.add_argument("--pathSaveImg", type=str, help="Location where you want to save the detected images from the video")
args = parser.parse_args()

load_model()

if (args.mode == 1):
    # example: 'C:/Users/natha/Documents/GitHub/ai-pothole-models/pothole-detection/datasets/3-potholes-normal-no-annotations/normal'
    # for some reason might need full path
    image_inference(args.pathImg, model)
else:
    live_camera_inference(args.pathSaveImg)
