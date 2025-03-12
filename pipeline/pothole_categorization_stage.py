import os
import math

class PotholeCategorizationStage:
    """Stage 6: Based on the estimated depth and area of the potholes, categorize them into different classes
       
    """
    
    def __init__(self, config):
        self.resolution = config.IMAGE_RESOLUTION
    
    def process(self, img_path, filtered_detections, pothole_areas, normalized_depths):
        print(f"[Stage 6] Categorizing the detected potholes in {os.path.basename(img_path)}")
        pothole_categories = []
        pothole_scores = []

        for i, (_, _, is_on_road, _) in enumerate(filtered_detections):
            total_score = -1
            if is_on_road:
                area = pothole_areas[i]
                depth = normalized_depths[i]

                # Normalization of the area values using normalization expression (max value of 1.0)
                # TODO: NATHAN update this to have the different supported resolutions we are planning to use
                if self.resolution == (3280, 2464):
                    width = 3280
                    height = 2464
                elif self.resolution == (1280, 720):
                    width = 1280
                    height = 720
                    
                max_area = (width/2) * (height/2)
                min_area = 528
                area_norm_func = (area - min_area) / (max_area - min_area)

                # Normalization of the depth values using a modified sigmoid function to normalize the depth values
                # Modified sigmoid function is bounded between 0 and 1 for the positive
                # values of x. Default sigmoid function is bounded between 0 and 1 for
                # the negative and positive values of x. 
                depth_norm_func = 2 * (1 / (1+math.exp(-depth)) - 0.5) # highest value of 1

                total_score = depth_norm_func + area_norm_func

            # Categorization of the potholes based on the normalized area and depth values
            if 1.8 <= total_score < 2.0:
                category = "Critical"
            elif 1.2 <= total_score < 1.8:
                category = "High"
            elif 0.6 <= total_score < 1.2:
                category = "Moderate"
            elif 0.0 <= total_score < 0.6:
                category = "Low"
            else: # for potholes 'not on the road' => in that case total_score = -1
                category = "NA"

            pothole_categories.append(category)
            pothole_scores.append(total_score)

        return { 'categories' : pothole_categories, 'scores' : pothole_scores }