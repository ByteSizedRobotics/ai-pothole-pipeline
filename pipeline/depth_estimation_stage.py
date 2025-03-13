import math
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

    def resize_with_padding(self, image, target_size):
        target_width, target_height = target_size
        h, w = image.shape[:2]
        
        # Calculate scaling factor to preserve aspect ratio
        scale = min(target_width / w, target_height / h)
        
        # Calculate new dimensions
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize the image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a black canvas of target size
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_w = (target_width - new_w) // 2
        pad_h = (target_height - new_h) // 2
        
        # Place the resized image on the canvas
        result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return result
    
    def process(self, img_path, filtered_detections, pothole_areas, percentile_filter, percentile_low_value, percentile_high_value):
        print(f"[Stage 5] Estimating depth of the potholes in {os.path.basename(img_path)}")

        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            image = cv2.imread(img_path)
        
        cropped_potholes = [] # stores the cropped potholes
        depth_maps = [] # stores the depth maps of the potholes
        relative_depths = [] # stores the relative depths of the potholes (max depth - min depth)
        relative_depths_divided_area = [] # stores the normalized depths of the potholes

        for i, (_, bbox, is_on_road, _) in enumerate(filtered_detections):
            if is_on_road:
                x1, y1, x2, y2 = bbox
                
                x1, y1 = max(0, int(x1)), max(0, int(y1)) # make sure x1 and y1 are not negative
                x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2)) # make sure x2 and y2 are within image bounds
                
                cropped_pothole = image[y1:y2, x1:x2]
                # cropped_potholes.append(cropped_pothole)

                # Resize the cropped pothole to the target size
                resized_pothole = cv2.resize(cropped_pothole, (512, 256), interpolation=cv2.INTER_AREA)
                # resized_pothole = self.resize_with_padding(cropped_pothole, (512, 512))
                cropped_potholes.append(resized_pothole)

                depth_map = self.depth_estimator.detect(resized_pothole) # perform depth estimation with DepthAnythingV2 model
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
                relative_depth_divided_area = (relative_depth / np.cbrt(pothole_areas[i])) / 1000
                # math.sqrtrroot(pothole_areas[i])
                relative_depths_divided_area.append(relative_depth_divided_area)

                # print('Relative Depth:', relative_depth)
                # print('Normalized Depth:', normalized_depth)
                # print('Max Depth:', max_depth)
                # print('Min Depth:', min_depth, '\n')

            else:
                cropped_potholes.append(None)
                depth_maps.append(None)
                relative_depths.append(-1) # -1 means pothole is not on the road
                relative_depths_divided_area.append(-1)

        return {
            'cropped_potholes': cropped_potholes,
            'depth_maps': depth_maps,
            'relative_depths': relative_depths,
            'relative_depths_divided_area': relative_depths_divided_area
        }