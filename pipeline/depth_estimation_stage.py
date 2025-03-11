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
        
        cropped_potholes = [] # stores the cropped potholes
        depth_maps = [] # stores the depth maps of the potholes
        relative_depths = [] # stores the relative depths of the potholes (max depth - min depth)
        normalized_depths = [] # stores the normalized depths of the potholes

        for _, bbox, is_on_road, _ in filtered_detections:
            if is_on_road:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                
                x1, y1 = max(0, int(x1)), max(0, int(y1)) # make sure x1 and y1 are not negative
                x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2)) # make sure x2 and y2 are within image bounds
                
                cropped_pothole = image[y1:y2, x1:x2]
                cropped_potholes.append(cropped_pothole)

                depth_map = self.depth_estimator.detect(cropped_pothole) # perform depth estimation with DepthAnythingV2 model
                depth_maps.append(depth_map)

                if percentile_filter: # use percentiles to filter out outliers
                    min_depth = np.percentile(depth_map, percentile_low_value)
                    max_depth = np.percentile(depth_map, percentile_high_value)
                else: # if not using percentiles, use min and max values
                    min_depth = np.min(depth_map)
                    max_depth = np.max(depth_map)

                relative_depth = max_depth - min_depth
                relative_depths.append(relative_depth)

                # Normalize the depth by dividing the relative depth by the square root of the area
                # Or else bigger potholes will have higher depth values than smaller potholes 
                # regardless of the actual depth of the pothole.
                normalized_depth = relative_depth / np.sqrt(area) * 1000
                normalized_depths.append(normalized_depth)

            else:
                cropped_potholes.append(None)
                depth_maps.append(None)
                relative_depths.append(-1) # -1 means pothole is not on the road
                normalized_depths.append(-1)
            
            print('Relative Depth:', relative_depth)
            print('Normalized Depth:', normalized_depth)
            print('Max Depth:', max_depth)
            print('Min Depth:', min_depth, '\n')

        return {
            'cropped_potholes': cropped_potholes,
            'depth_maps': depth_maps,
            'relative_depths': relative_depths,
            'normalized_depths': normalized_depths
        }