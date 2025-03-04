# pipeline/segmentation_stage.py
import os
from PIL import Image

class RoadSegmentationStage:
    """Stage 2: Segment the road in the image"""
    
    def __init__(self, road_segmenter):
        self.segmenter = road_segmenter
    
    def process(self, stage1_output):
        """
        Process the output from stage 1 to segment the road.
        
        Args:
            stage1_output: Output dictionary from stage 1
            
        Returns:
            Dictionary containing all items from stage1_output plus:
            - 'road_mask': Binary road mask
            - 'full_segmentation': Complete segmentation result
        """
        image_path = stage1_output['image_path']
        print(f"[Stage 2] Segmenting road in {os.path.basename(image_path)}")
        
        # Get the image from stage 1
        image = Image.open(image_path).convert('RGB')

        # Perform road segmentation
        road_mask, full_segmentation = self.segmenter.segment_image(image)
        
        # Add results to output
        result = stage1_output.copy()
        result['road_mask'] = road_mask
        result['full_segmentation'] = full_segmentation
        
        return result