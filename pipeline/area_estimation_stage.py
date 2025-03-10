import os

class AreaEstimationStage:
    """Stage 4: Estimate the area of the detected potholes
       
       The area estimation is done by using the bounding box area.
       This does not provide an accurate area of the actual pothole BUT
       allows us to gain a relative understanding of how big the pothole is compared to others.
       The function used to calculate the scaling factor to multiply the bounding box area
       is a found by fitting a curve to data points (see section modules/area_estimation/ for more details).
    """
    
    def __init__(self, road_segmenter):
        self.segmenter = road_segmenter
    
    def process(self, img_path, filtered_detections):
        print(f"[Stage 4] Estimating relative area of pothole {os.path.basename(img_path)}")
        
        pothole_areas = []

        a = -0.00022788150560381466
        b = -249.57544427170112
        c = 0.6036184827412467
        d = -0.00036558500089469364

        for _, bbox, is_on_road, _ in filtered_detections:
            area = 0
            if (is_on_road):
                x1, y1, x2, y2 = bbox
                bounding_box_area = (x2-x1)*(y2-y1)
                y_distance_middle_pothole = (y1+y2)/2
                scaling_factor = a/(b + c*y_distance_middle_pothole + d*(y_distance_middle_pothole**2))
                area = scaling_factor * bounding_box_area

                pothole_areas.append(area)
            print(f"Area of pothole: {area}")

        return pothole_areas