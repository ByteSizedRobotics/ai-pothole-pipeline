import os
from PIL import Image

class RoadSegmentationStage:
    """Stage 2: Segment the road in the image
    
       Road segmentation is done by using the DeepLabV3+ model trained on Cityscapes dataset.
       Both the full Cityscapes and binary road mask are returned.
    """
    
    def __init__(self, road_segmenter):
        self.segmenter = road_segmenter
    
    def process(self, image_path):
        print(f"[Stage 2] Segmenting road in {os.path.basename(image_path)}")
        
        image = Image.open(image_path).convert('RGB')

        road_mask, full_segmentation = self.segmenter.segment_image(image) # road segmentation

        return {'road_mask': road_mask, 'full_segmentation': full_segmentation}