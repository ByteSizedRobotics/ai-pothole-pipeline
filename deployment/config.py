# Directory structure
# project_root/
# ├── main.py                     # Main script to run the complete pipeline
# ├── config.py                   # Configuration parameters
# ├── modules/
# │   ├── road_segmentation.py       # Road segmentation using DeepLabV3+
# │   ├── pothole_detection.py       # Your pothole detection model
# |   └── models/
# |       ├── DeepLabV3Plus/
# |       └── pothole-detection/
# ├── pipeline/
# │   ├── pothole_detection_stage.py      # Stage 1: Detect potholes
# │   ├── road_segmentation_stage.py      # Stage 2: Road segmentation
# │   └── pothole_filtering_stage.py      # Stage 3: Filter potholes based on road mask
# └── utils/
#     ├── visualization.py        # Visualization utilities
#     └── io_utils.py             # Input/output utilities

# config.py
import os

class Config:
    # Paths
    INPUT_PATH = "data/images"
    OUTPUT_PATH = "data/results"
    DEEPLAB_CHECKPOINT = "modules/ai_models/DeepLabV3Plus/checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth"
    POTHOLE_MODEL_PATH = "modules/ai_models/pothole-detection/train-runs/2025-03-01_combined1.1/run/weights/best.pt"  # TODO:NATHAN update this
    
    # DeepLabV3+ configuration
    DEEPLAB_MODEL = "deeplabv3plus_resnet101"
    DATASET = "cityscapes"
    OUTPUT_STRIDE = 16
    NUM_CLASSES = 19  # Cityscapes
    
    # Pothole filtering configuration
    MIN_OVERLAP_THRESHOLD = 50.0  # Minimum percentage overlap to consider pothole on road
    
    # Device configuration
    GPU_ID = "0"
    
    # Create directories if they don't exist
    @staticmethod
    def create_dirs():
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)