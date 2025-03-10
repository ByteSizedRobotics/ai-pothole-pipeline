import os
import cv2

class PotholeDetectionStage:
    """Stage 1: Detect potholes in images"""
    
    def __init__(self, pothole_detector):
        self.detector = pothole_detector
    
    def process(self, image_path):
        print(f"[Stage 1] Detecting potholes in {os.path.basename(image_path)}")
        
        if image_path.endswith(".jpg") or image_path.endswith(".png"):

            # Load image            
            image = cv2.imread(image_path)

            # Detect potholes
            detections = self.detector.detect(image)

            # Return results
            return detections
            