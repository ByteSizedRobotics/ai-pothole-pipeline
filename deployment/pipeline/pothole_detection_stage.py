# pipeline/detection_stage.py
import os
import cv2

class PotholeDetectionStage:
    """Stage 1: Detect potholes in images"""
    
    def __init__(self, pothole_detector):
        self.detector = pothole_detector
    
    def process(self, image_path):
        """
        Process an image through the pothole detection stage.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing:
            - 'image': The original OPEN CV image (uses BGR format)
            - 'detections': List of pothole detections with format [(confidence, [x1, y1, x2, y2]), ...]
            - 'image_path': Original image path
        """
        print(f"[Stage 1] Detecting potholes in {os.path.basename(image_path)}")
        
        if image_path.endswith(".jpg") or image_path.endswith(".png"):

            # Load image            
            image = cv2.imread(image_path)

            # Detect potholes
            detections = self.detector.detect(image)
        
            # Return results
            return {
                'image': image,
                'detections': detections,
                'image_path': image_path
            }