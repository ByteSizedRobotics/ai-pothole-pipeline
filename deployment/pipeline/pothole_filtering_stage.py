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
        # full_segmentation = road_segmentations['full_segmentation']
        road_mask = road_segmentations['road_mask']
        detections = pothole_detections
        
        # Filter detections based on road mask
        filtered_detections = []
        count = 0
        for *bbox, confidence, classType in detections.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate overlap with road
            # patch = road_mask[y1:y2, x1:x2]
            # pothole_area = (x2 - x1) * (y2 - y1)

            # display the size of the road mask
            print("Road Mask Size:",road_mask.shape[0],road_mask.shape[1])
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            print("Pothole corners",count,":",corners)
            corner_on_road = 0
            is_on_road = False

            for x,y in corners:
                if 0 <= x-1 <= road_mask.shape[1] and 0 <= y-1 <= road_mask.shape[0]:
                    if road_mask[y-1,x-1] == 1:
                        corner_on_road += 1
                    print("Corner",count,":",y,x)
                    print("Pothole",count,":",road_mask[y-1,x-1])
                print("\n")


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
            filtered_detections.append((confidence, bbox, is_on_road, corner_on_road))
            count += 1    
        return filtered_detections