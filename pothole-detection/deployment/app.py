import cv2
import torch
from flask import Flask, Response

app = Flask(__name__)

# Load the classificaiton model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../train-runs/2025-01-26/run/weights/best.pt')

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use 0 for the default camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform inference
            results = model(frame)
            results.render()  # Draw bounding boxes and labels on the frame

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)