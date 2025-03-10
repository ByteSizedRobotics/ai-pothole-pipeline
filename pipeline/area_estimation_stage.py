import os
from PIL import Image

class RoadSegmentationStage:
    """Stage 4: Estimate the area of the detected potholes"""
    
    def __init__(self, road_segmenter):
        self.segmenter = road_segmenter
    
    def process(self, image_path):
        print(f"[Stage 2] Segmenting road in {os.path.basename(image_path)}")
        
        # Get the image from stage 1
        image = Image.open(image_path).convert('RGB')

        # Perform road segmentation
        road_mask, full_segmentation = self.segmenter.segment_image(image)

        return {'road_mask': road_mask, 'full_segmentation': full_segmentation}