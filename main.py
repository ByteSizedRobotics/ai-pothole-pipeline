import os
import argparse
from time import time
from PIL import Image
from config import Config # config file

# import models, pipeline stages, visualization functions and utils required by DeepLabV3+
from modules import RoadSegmentation, PotholeDetection, DepthEstimation
from pipeline import PotholeDetectionStage, RoadSegmentationStage, PotholeFilteringStage, AreaEstimationStage, DepthEstimationStage, PotholeCategorizationStage
from utils_lib.visualization import visualize_original_image, visualize_pothole_detections, visualize_full_segmentation, visualize_road_segmentation, visualize_filtered_detections, visualize_depth_results, visualize_area_depth_results, visualize_combined_results, create_results_file
from utils_lib.io_utils import get_image_files

def parse_args():
    parser = argparse.ArgumentParser(description='Pothole Detection & Analysis Pipeline')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input image or directory') # by default it is set to data/images
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output directory') # by default it is set to data/results
    parser.add_argument('--resolution', type=str, default=None,
                        help='Image resolution (ex: 3280x2464, 1280x720)') # by default it is set to the value in config.py (3280x2464)
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.input: # if input/output are provided, update the config
        Config.INPUT_PATH = args.input
    if args.output:
        Config.OUTPUT_PATH = args.output
    if args.resolution:
        Config.IMAGE_RESOLUTION = tuple(map(int, args.resolution.split('x')))
    
    Config.create_dirs()
    
    print("Initializing models...")
    road_segmenter = RoadSegmentation(Config)
    pothole_detector = PotholeDetection(Config)
    depth_estimator = DepthEstimation(Config)
    
    # PIPELINE STAGES
    print("Setting up pipeline stages...")
    detection_stage = PotholeDetectionStage(pothole_detector)
    segmentation_stage = RoadSegmentationStage(road_segmenter)
    filtering_stage = PotholeFilteringStage(Config)
    area_estimation_stage = AreaEstimationStage(Config)
    depth_estimation_stage = DepthEstimationStage(depth_estimator)
    pothole_categorization_stage = PotholeCategorizationStage(Config)
    
    image_files = get_image_files(Config.INPUT_PATH)
    if not image_files:
        print(f"No images found in {Config.INPUT_PATH}")
        return
    print(f"Found {len(image_files)} images to process")
    
    # Process each image through the pipeline
    for img_path in image_files:
        print(f"\nProcessing {os.path.basename(img_path)}...")
        start_time = time()
        try:
            image = Image.open(img_path).convert('RGB')
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = Config.OUTPUT_PATH
            visualize_original_image(image, save_path, image_name)

            # Stage 1: Pothole Detection
            pothole_detections = detection_stage.process(img_path)
            visualize_pothole_detections(image, pothole_detections, save_path, image_name)

            # TODO: NATHAN add a check here to see if there are any potholes detected
            # if no potholes detected => no need to do segmentation and filtering
            # Stage 2: Road Segmentation
            road_segmentation = segmentation_stage.process(img_path)
            visualize_full_segmentation(road_segmentation['full_segmentation'], save_path, image_name)
            visualize_road_segmentation(road_segmentation['road_mask'], save_path, image_name)

            # Stage 3: Filter Potholes
            filtered_detections = filtering_stage.process(img_path, pothole_detections, road_segmentation)
            visualize_filtered_detections(image, road_segmentation['road_mask'], filtered_detections, save_path, image_name)

            # Stage 4: Area Estimation
            pothole_areas = area_estimation_stage.process(img_path, filtered_detections)

            # Stage 5: Depth Estimation
            depth_estimations = depth_estimation_stage.process(img_path, filtered_detections, pothole_areas, Config.DEPTH_ANYTHING_PERCENTILE_FILTER['percentile_filter'],
                                Config.DEPTH_ANYTHING_PERCENTILE_FILTER['percentile_low_value'], Config.DEPTH_ANYTHING_PERCENTILE_FILTER['percentile_high_value'])
            visualize_depth_results(depth_estimations, save_path, image_name)

            
            # Stage 6: Pothole Categorization
            pothole_categorizations = pothole_categorization_stage.process(img_path, filtered_detections, pothole_areas, depth_estimations['normalized_depths'])
            visualize_area_depth_results(pothole_areas, depth_estimations, pothole_categorizations, save_path, image_name)

            process_time = time() - start_time
            print(f"Processing completed in {process_time:.2f} seconds (including saving the visualizations)")
            
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
                'detections' : pothole_detections,
                'pothole_areas': pothole_areas,
                'depth_estimations': depth_estimations,
                'pothole_categorizations': pothole_categorizations
            }

            visualize_combined_results(pipeline_output, Config.OUTPUT_PATH)

            # save results in .txt file
            create_results_file(filtered_detections, pothole_areas, depth_estimations, pothole_categorizations, Config.OUTPUT_PATH, img_path)
            print(f"Results saved to {Config.OUTPUT_PATH}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
if __name__ == "__main__":
    main()