import os
import math

class PotholeCategorizationStage:
    """Stage 6: Based on the estimated depth and area of the potholes, categorize them into different classes
       
    """
    
    def __init__(self, config):
        self.resolution = config.IMAGE_RESOLUTION
    
    def process(self, img_path, filtered_detections, pothole_areas, depth_values):
        print(f"[Stage 6] Categorizing the detected potholes in {os.path.basename(img_path)}")
        pothole_categories = []
        pothole_scores = []
        normalized_areas = []
        normalized_depths = []

        for i, (_, _, is_on_road, _) in enumerate(filtered_detections):
            total_score = -1
            area_norm = -1
            depth_norm = -1
            if is_on_road:
                estimated_area = pothole_areas[i]
                depth = depth_values[i]
                
                """
                AREA VALUE NORMALIZATION
                """
                # Normalization of the area values using normalization expression (max value of 1.0)
                # TODO: NATHAN update this to have the different supported resolutions we are planning to use
                if self.resolution == (3280, 2464):
                    img_width = 3280
                    img_height = 2464
                    a = -0.00022788150560381466
                    b = -249.57544427170112
                    c = 0.6036184827412467
                    d = -0.00036558500089469364

                    # these values are estimated as the smallest bounding box which can be detected
                    min_area_top_left_coord = (1565, 868)
                    min_area_bot_right_coord = (1718, 880)

                elif self.resolution == (1280, 720):
                    img_width = 1280
                    img_height = 720

                    # these values are estimated as the smallest bounding box which can be detected
                    min_area_top_left_coord = (602, 261)
                    min_area_bot_right_coord = (678, 266)
                

                ##### calculate the MAX area taking into account the scaling factor
                max_width = img_width/2
                max_height = img_height/2
                max_area = max_width * max_height # assuming the biggest pothole can be half of the images
                max_area_y_distance_middle_pothole = max_height/2 # y distance in pixels to middle of bounding box

                scaling_factor = a/(b + c*max_area_y_distance_middle_pothole + d*(max_area_y_distance_middle_pothole**2))
                max_area_final = scaling_factor * max_area 

                ##### calculate the MIN area taking into account the scaling factor
                min_width = min_area_bot_right_coord[0] - min_area_top_left_coord[0] # using the min_area coordinates
                min_height = min_area_bot_right_coord[1] - min_area_top_left_coord[1]
                min_area = min_width * min_height
                min_area_y_distance_middle_pothole = (min_area_top_left_coord[1] + min_area_bot_right_coord[1]) / 2

                scaling_factor = a/(b + c*min_area_y_distance_middle_pothole + d*(min_area_y_distance_middle_pothole**2))
                min_area_final = scaling_factor * min_area

                ##### calculate the normalized area => [0, 1]
                area_norm = (estimated_area - min_area_final) / (max_area_final - min_area_final)


                """
                DEPTH VALUE NORMALIZATION
                """
                # Normalization of the depth values using a modified sigmoid function to normalize the depth values
                # Modified sigmoid function is bounded between 0 and 1 for the positive
                # values of x. Default sigmoid function is bounded between 0 and 1 for
                # the negative and positive values of x. 
                depth_norm = 2 * (1 / (1+math.exp(-depth)) - 0.5) # highest value of 1

                total_score = depth_norm * area_norm
                print(f"Area Norm: {area_norm}, Depth Norm: {depth_norm}, Total Score: {total_score}")

            # Categorization of the potholes based on the normalized area and depth values
            if 0.8 <= total_score < 1.0:
                category = "Critical"
            elif 0.6 <= total_score < 0.8:
                category = "High"
            elif 0.3 <= total_score < 0.6:
                category = "Moderate"
            elif 0.0 <= total_score < 0.3:
                category = "Low"
            else: # for potholes 'not on the road' => in that case total_score = -1
                category = "NA"

            pothole_categories.append(category)
            pothole_scores.append(total_score)
            normalized_areas.append(area_norm)
            normalized_depths.append(depth_norm)

        return { 'categories' : pothole_categories, 'scores' : pothole_scores , 'normalized_areas': normalized_areas, 'normalized_depths': normalized_depths}