import os
import cv2

class PotholeDetectionStage:
    """Stage 1: Detect potholes in images
       
       Pothole detection is done by using the YOLOv5s model trained on custom pothole dataset.
       The custom trained weights are loaded and used to detect potholes in the image.
    """
    
    def __init__(self, pothole_detector):
        self.detector = pothole_detector
    
    def process(self, image_path):
        print(f"[Stage 1] Detecting potholes in {os.path.basename(image_path)}")
        
        if image_path.endswith(".jpg") or image_path.endswith(".png"):
            image = cv2.imread(image_path) # load image            
            detections = self.detector.detect(image) # detect potholes

            return detections
            