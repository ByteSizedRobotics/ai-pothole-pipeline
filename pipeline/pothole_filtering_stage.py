import math
import os

class PotholeFilteringStage:
    """Stage 3: Filter pothole detections based on road segmentation
       
       Filtering is done by checking if majority of sampled pixel points in a given
       pothole bounding box are classified as road. If >= MIN_PIXELS_ROAD_THRESHOLD
       are on the road => pothole is considered to be on the road.
    """
    
    def __init__(self, config):
        self.min_road_threshold = config.MIN_PIXELS_ROAD_THRESHOLD
    
    def process(self, img_path, pothole_detections, road_segmentations):
        print(f"[Stage 3] Filtering potholes in {os.path.basename(img_path)}")
        
        # Get road mask and detections from previous stages
        # full_segmentation = road_segmentations['full_segmentation']
        road_mask = road_segmentations['road_mask']
        full_seg = road_segmentations['full_segmentation']
        detections = pothole_detections
        
        # Filter detections based on road mask
        filtered_detections = []
        count = 0
        for *bbox, confidence, classType in detections.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, bbox)

            # display the size of the road mask
            # print("Road Mask Size:",road_mask.shape[0],road_mask.shape[1])
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            # print("Pothole corners",count,":",corners)
            corner_on_road = 0
            is_on_road = False

            # OG method to check if pothole is on road
            # checks what the segmented class for each of the corners of the bounding box
            # if at least 3 corners = road => assume pothole is on the road
            # for x,y in corners:
            #     if 0 <= x-1 <= road_mask.shape[1] and 0 <= y-1 <= road_mask.shape[0]: # IMPORTANT: x and y are swapped in numpy array for the segmentation matrix
            #         if road_mask[y-1,x-1] == 1:
            #             corner_on_road += 1
            #         # print("Corner",count,":",y,x)
            #         # print("Pothole",count,":",road_mask[y-1,x-1])
            #     print("\n")
            # if corner_on_road >= 3:
            #     is_on_road = True
            
            # Better method
            # checks a range of points within the bounding box
            # if minimum 60% are = road => assume pothole is on the road
            total_num_points = 0
            num_points_on_road = 0
            num_points_on_vegetation = 0

            # DYNAMIC SCALING: ensures at least a step of 5 and scales with the size of the bounding box
            # total_area = (x2-x1)*(y2-y1)
            # step = max(5, int(math.sqrt(total_area) // 100))
            # print(step)
            # print("Total Area:",total_area)

            step = 5
            for x in range(x1, x2, step):
                for y in range(y1, y2, step):
                    x = x-1
                    y = y-1
                    if 0 <= x <= road_mask.shape[1] and 0 <= y <= road_mask.shape[0]:
                        total_num_points += 1
                        if road_mask[y, x] == 1:
                            num_points_on_road += 1
                        elif full_seg[y, x] == 8:
                            num_points_on_vegetation += 1 

            percentage_pixels_on_road = (num_points_on_road / total_num_points)
            percentage_pixels_on_vegetation = (num_points_on_vegetation / total_num_points)
            print("Percentage of pixels on road:",percentage_pixels_on_road)
            print("Percentage of pixels on vegetation:",percentage_pixels_on_vegetation)

            if percentage_pixels_on_road >= self.min_road_threshold:
                is_on_road = True
            elif confidence >= 0.80 and percentage_pixels_on_road >= 0.30:
                is_on_road = True
            elif confidence >= 0.75 and percentage_pixels_on_vegetation >= 0.5 and percentage_pixels_on_road >= 0.05:
                is_on_road = True
            
            # Add to filtered detections with additional info
            filtered_detections.append((confidence, bbox, is_on_road, percentage_pixels_on_road))
            count += 1    
        return filtered_detections