import os

class AreaEstimationStage:
    """Stage 4: Estimate the area of the detected potholes
       
       The area estimation is done by using the bounding box area.
       This does not provide an accurate area of the actual pothole BUT
       allows us to gain a relative understanding of how big the pothole is compared to others.
       The function used to calculate the scaling factor to multiply the bounding box area
       is a found by fitting a curve to data points (see section modules/area_estimation/ for more details).
    """
    
    def __init__(self, config):
        self.resolution = config.IMAGE_RESOLUTION
    
    def process(self, img_path, filtered_detections):
        print(f"[Stage 4] Estimating area of the potholes in {os.path.basename(img_path)}")
        
        pothole_areas = []

        for _, bbox, is_on_road, _ in filtered_detections:
            area = 0
            if (is_on_road):
                x1, y1, x2, y2 = bbox
                bounding_box_area = (x2-x1)*(y2-y1)
                y_distance_middle_pothole = (y1+y2)/2
                x_distance_middle_pothole = (x1+x2)/2

                # TODO: NATHAN update this to have the different supported resolutions we are planning to use
                # TODO: NATHAN need to calculate the other scaling factors for the other resolutions
                if self.resolution == (3280, 2464):
                    a = -0.00022788150560381466
                    b = -249.57544427170112
                    c = 0.6036184827412467
                    d = -0.00036558500089469364
                    
                    # Attempt 1 - using just y scaling factor
                    y_scaling_factor = a/(b + c*y_distance_middle_pothole + d*(y_distance_middle_pothole**2))

                    # Attempt 2 - using just y scaling factor but stopping drop-down in curve
                    # if y_distance_middle_pothole >= 800:
                    #     y_scaling_factor = a/(b + c*y_distance_middle_pothole + d*(y_distance_middle_pothole**2))
                    # else: # if the pothole is above the 800 pixel mark, we use the scaling factor at 800 pixels (this is due to the properties of the curve => starts dropping down after 800 pixels)
                    #     y_scaling_factor = a/(b + c*800 + d*(800**2))

                    # Attempt 3 - using both x and y scaling factors
                    if x_distance_middle_pothole <= 1640:
                        x_scaling_factor = 4*(10**(-9))+6*(10**(-6))
                    elif x_distance_middle_pothole > 1640:
                        x_scaling_factor = -4*(10**(-9))+2*(10**(-5))

                # elif self.resolution == (1280, 720):
                
                area = x_scaling_factor/2 * y_scaling_factor * bounding_box_area * 100000
                pothole_areas.append(area)
            else:
                pothole_areas.append(-1) # -1 means pothole is not on the road
            # print(f"Area of pothole: {area}")

        return pothole_areas