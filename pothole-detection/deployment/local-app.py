import os
import cv2
import torch

def live_camera_inference():
    # Open the camera
    camera = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Perform inference
        results = model(frame)

        # Render bounding boxes and labels on the frame
        results.render()  # This modifies the frame in-place
        frame = results.ims[0]  # Extract the modified image

        # Display the frame
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('YOLOv5 Live', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()

def image_inference(dataset_path, model):
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            
            results = model(image)
        
            # Check if any objects were detected
            if results.xyxy[0].shape[0] > 0:
                results.show()
    cv2.destroyAllWindows()

# Load the YOLOv5 model with specific weights

# Testing 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../train-runs/2025-01-28/run/weights/best.pt')

image_inference('C:/Users/natha/Documents/GitHub/ai-pothole-models/pothole-detection/datasets/3-potholes-normal-no-annotations/normal', model)


