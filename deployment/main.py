# main.py
import os
import argparse
from time import time
from PIL import Image

# Import configuration
from config import Config

# Import models
from modules import RoadSegmentation, PotholeDetection

# Import pipeline stages
from pipeline import PotholeDetectionStage, RoadSegmentationStage, PotholeFilteringStage

# Import utilities
from utils_lib.visualization import visualize_pipeline_results, create_results_file
from utils_lib.io_utils import get_image_files

def parse_args():
    parser = argparse.ArgumentParser(description='Pothole Detection Pipeline')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input image or directory') # by default it is set to data/images
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output directory') # by default it is set to data/results
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Update config with command line arguments
    if args.input:
        Config.INPUT_PATH = args.input
    if args.output:
        Config.OUTPUT_PATH = args.output
    
    # Create output directory
    Config.create_dirs()
    
    # Initialize models
    print("Initializing models...")
    road_segmenter = RoadSegmentation(Config)
    pothole_detector = PotholeDetection(Config)
    
    # Create pipeline stages
    print("Setting up pipeline stages...")
    detection_stage = PotholeDetectionStage(pothole_detector)
    segmentation_stage = RoadSegmentationStage(road_segmenter)
    filtering_stage = PotholeFilteringStage(Config)
    
    # Get list of images to process
    image_files = get_image_files(Config.INPUT_PATH)
    
    if not image_files:
        print(f"No images found in {Config.INPUT_PATH}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image through the pipeline
    for img_path in image_files:
        print(f"\nProcessing {os.path.basename(img_path)}...")
        start_time = time()
        
        # Run through pipeline stages
        try:
            # Stage 1: Pothole Detection
            pothole_detections = detection_stage.process(img_path)
            
            # Stage 2: Road Segmentation
            road_segmentation = segmentation_stage.process(img_path)
            
            # Stage 3: Filter Potholes
            filtered_detections = filtering_stage.process(img_path, pothole_detections, road_segmentation)
            
            # Process results
            process_time = time() - start_time
            print(f"Processing completed in {process_time:.2f} seconds")
            
            # Print summary
            all_potholes = len(filtered_detections)
            road_potholes = sum(1 for _, _, is_on_road, _ in filtered_detections if is_on_road)
            print(f"Found {road_potholes} potholes on the road out of {all_potholes} detected")
            
            pipeline_output = {
                'image': Image.open(img_path).convert('RGB'),
                'image_path': img_path,
                'full_segmentation': road_segmentation['full_segmentation'],
                'road_mask': road_segmentation['road_mask'],
                'filtered_detections': filtered_detections,
                'detections' : pothole_detections
            }

            # save pipeline results and create results .txt file
            visualize_pipeline_results(pipeline_output, Config.OUTPUT_PATH)
            create_results_file(filtered_detections, Config.OUTPUT_PATH, img_path)
            print(f"Results saved to {Config.OUTPUT_PATH}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
if __name__ == "__main__":
    main()