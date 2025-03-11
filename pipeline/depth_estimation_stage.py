import os
import cv2
import numpy as np

class DepthEstimationStage:
    """Stage 5: Estimate the depth of the potholes
       
       The depth estimation is done by using DepthAnythingV2 model.
       The detected potholes are cropped and passed into the DepthAnythingV2 model to estimate the depth.
       The relative depth is then estimated by taking the maximum depth value - minimum depth value in the depth map.
    """
    
    def __init__(self, depth_estimator):
        self.depth_estimator = depth_estimator
    
    def process(self, img_path, filtered_detections, percentile_filter, percentile_low_value, percentile_high_value):
        print(f"[Stage 5] Estimating relative depth of pothole {os.path.basename(img_path)}")

        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            image = cv2.imread(img_path)
        
        cropped_potholes = []
        depth_maps = []
        estimated_depths = []

        for _, bbox, is_on_road, _ in filtered_detections:
            if is_on_road:
                x1, y1, x2, y2 = bbox
                
                x1, y1 = max(0, int(x1)), max(0, int(y1)) # make sure x1 and y1 are not negative
                x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2)) # make sure x2 and y2 are within image bounds
                
                cropped_pothole = image[y1:y2, x1:x2]
                cropped_potholes.append(cropped_pothole)

                depth_map = self.depth_estimator.detect(cropped_pothole)
                depth_maps.append(depth_map)

                if percentile_filter: # use percentiles to filter out outliers
                    min_depth = np.percentile(depth_map, percentile_low_value)
                    max_depth = np.percentile(depth_map, percentile_high_value)

                estimated_depth = max_depth - min_depth
                print('Estimated Depth:', estimated_depth)
                print('Max Depth:', max_depth)
                print('Min Depth:', min_depth)

                estimated_depths.append(estimated_depth)
            else:
                cropped_potholes.append(None)
                depth_maps.append(None)
                estimated_depths.append(-1) # -1 means pothole is not on the road

        return {
            'cropped_potholes': cropped_potholes,
            'depth_maps': depth_maps,
            'estimated_depths': estimated_depths
        }