# pipeline/filtering_stage.py
import os
import numpy as np

class PotholeFilteringStage:
    """Stage 3: Filter pothole detections based on road segmentation"""
    
    def __init__(self, config):
        self.min_overlap_threshold = config.MIN_OVERLAP_THRESHOLD
    
    def process(self, stage2_output):
        """
        Process the output from stage 2 to filter potholes.
        
        Args:
            stage2_output: Output dictionary from stage 2
            
        Returns:
            Dictionary containing all items from stage2_output plus:
            - 'filtered_detections': List of filtered detections with format 
              [(confidence, bbox, is_on_road, overlap_percentage), ...]
        """
        image_path = stage2_output['image_path']
        print(f"[Stage 3] Filtering potholes in {os.path.basename(image_path)}")
        
        # Get road mask and detections from previous stages
        road_mask = stage2_output['road_mask']
        detections = stage2_output['detections']
        
        # Filter detections based on road mask
        filtered_detections = []
        for confidence, bbox in detections:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate overlap with road
            patch = road_mask[y1:y2, x1:x2]
            pothole_area = (x2 - x1) * (y2 - y1)
            
            if pothole_area > 0:
                road_pixels = np.sum(patch)
                overlap_percentage = (road_pixels / pothole_area) * 100
                is_on_road = overlap_percentage >= self.min_overlap_threshold
            else:
                overlap_percentage = 0.0
                is_on_road = False
            
            # Add to filtered detections with additional info
            filtered_detections.append((confidence, bbox, is_on_road, overlap_percentage))
        
        # Add results to output
        result = stage2_output.copy()
        result['filtered_detections'] = filtered_detections
        
        return result