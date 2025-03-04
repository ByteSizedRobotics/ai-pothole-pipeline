# main.py
import os
import argparse
from time import time

# Import configuration
from config import Config

# Import models
from modules.road_segmentation import RoadSegmentation
from modules.pothole_detection import PotholeDetection

# Import pipeline stages
from pipeline.pothole_detection_stage import PotholeDetectionStage
from pipeline.road_segmentation_stage import RoadSegmentationStage
from pipeline.pothole_filtering_stage import PotholeFilteringStage

# Import utilities
from utils.visualization import visualize_pipeline_results, save_results_as_images
from utils.io_utils import get_image_files

def parse_args():
    parser = argparse.ArgumentParser(description='Pothole Detection Pipeline')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output directory')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--save_images', action='store_true',
                        help='Save result images')
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
            stage1_output = detection_stage.process(img_path)
            
            # Stage 2: Road Segmentation
            stage2_output = segmentation_stage.process(stage1_output)
            
            # Stage 3: Filter Potholes
            stage3_output = filtering_stage.process(stage2_output)
            
            # Process results
            process_time = time() - start_time
            print(f"Processing completed in {process_time:.2f} seconds")
            
            # Print summary
            all_potholes = len(stage3_output['filtered_detections'])
            road_potholes = sum(1 for _, _, is_on_road, _ in stage3_output['filtered_detections'] if is_on_road)
            print(f"Found {road_potholes} potholes on the road out of {all_potholes} detected")
            
            # Generate output filename
            basename = os.path.splitext(os.path.basename(img_path))[0]
            
            # Visualize if requested
            if args.visualize:
                visualize_pipeline_results(stage3_output)
            
            # Save results if requested
            if args.save_images:
                # Save visualization
                vis_path = os.path.join(Config.OUTPUT_PATH, f"{basename}_visualization.png")
                visualize_pipeline_results(stage3_output, save_path=vis_path)
                
                # Save individual result images
                save_results_as_images(stage3_output, Config.OUTPUT_PATH)
                
                print(f"Results saved to {Config.OUTPUT_PATH}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nAll images processed!")

if __name__ == "__main__":
    main()