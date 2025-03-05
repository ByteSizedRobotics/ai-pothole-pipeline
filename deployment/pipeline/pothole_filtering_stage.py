# pipeline/filtering_stage.py
import os
import numpy as np

class PotholeFilteringStage:
    """Stage 3: Filter pothole detections based on road segmentation"""
    
    def __init__(self, config):
        self.min_overlap_threshold = config.MIN_OVERLAP_THRESHOLD
    
    def process(self, img_path, pothole_detections, road_segmentations):
        """
        Process the output from stage 2 to filter potholes.
        
        Args:
            stage2_output: Output dictionary from stage 2
            
        Returns:
            Dictionary containing all items from stage2_output plus:
            - 'filtered_detections': List of filtered detections with format 
              [(confidence, bbox, is_on_road), ...]
        """
        image_path = img_path
        print(f"[Stage 3] Filtering potholes in {os.path.basename(image_path)}")
        
        # Get road mask and detections from previous stages
        full_segmentation = road_segmentations['full_segmentation']
        # road_mask = road_segmentations['road_mask']
        detections = pothole_detections
        
        # Filter detections based on road mask
        filtered_detections = []
        for *bbox, confidence, classType in detections.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate overlap with road
            # patch = road_mask[y1:y2, x1:x2]
            # pothole_area = (x2 - x1) * (y2 - y1)
            
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            corner_on_road = 0
            is_on_road = False

            for x,y in corners:
                if 0 <= x < full_segmentation.shape[1] and 0 <= y < full_segmentation.shape[0]:
                    if full_segmentation[y,x] == 1:
                        corner_on_road += 1

            if corner_on_road >= 3:
                is_on_road = True
            
            # if pothole_area > 0:
            #     road_pixels = np.sum(patch)
            #     overlap_percentage = (road_pixels / pothole_area) * 100
            #     is_on_road = overlap_percentage >= self.min_overlap_threshold
            # else:
            #     overlap_percentage = 0.0
            #     is_on_road = False
            
            # Add to filtered detections with additional info
            filtered_detections.append((confidence, bbox, is_on_road))
                
        return filtered_detections