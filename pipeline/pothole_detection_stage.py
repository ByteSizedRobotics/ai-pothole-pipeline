import os
import cv2

class PotholeDetectionStage:
    """Stage 1: Detect potholes in images
       
       Pothole detection is done by using the YOLOv5s model trained on custom pothole dataset.
       The custom trained weights are loaded and used to detect potholes in the image.
    """
    
    def __init__(self, pothole_detector):
        self.detector = pothole_detector
    
    def process(self, img_path):
        print(f"[Stage 1] Detecting potholes in {os.path.basename(img_path)}")
        
        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            image = cv2.imread(img_path) # load image            
            detections = self.detector.detect(image) # detect potholes

            return detections
            