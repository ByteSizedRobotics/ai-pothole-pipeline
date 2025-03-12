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
    # Paths to input and output images
    INPUT_PATH = "data/images"
    OUTPUT_PATH = "data/results"
    IMAGE_RESOLUTION = (3280, 2464)  # TODO: NATHAN update this when needed for different image resolutions

    # Pothole detection configuration
    POTHOLE_MODEL_PATH = "modules/ai_models/pothole-detection/train-runs/2025-03-01_combined1.1/run/weights/best.pt"  # TODO: NATHAN keep this up to date based on best weights
    
    # DeepLabV3+ configuration
    DEEPLAB_CHECKPOINT_FILE = "modules/ai_models/DeepLabV3Plus/checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth"
    DEEPLAB_MODEL = "deeplabv3plus_resnet101"
    DATASET = "cityscapes"
    OUTPUT_STRIDE = 16
    NUM_CLASSES = 19  # Cityscapes

    # DepthAnything configuration
    DEPTH_ANYTHING_ENCODER = 'vitl' # or 'vits', 'vitb'
    DEPTH_ANYTHING_CHECKPOINT_DIR = "modules/ai_models/DepthAnythingV2/checkpoints"
    DEPTH_ANYTHING_MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    DEPTH_ANYTHING_PERCENTILE_FILTER = {
        'percentile_filter' : True,
        'percentile_low_value' : 5,
        'percentile_high_value' : 95
    }
    
    # Pothole filtering configuration
    MIN_PIXELS_ROAD_THRESHOLD = 0.60  # Minimum percentage of pixels in bounding box area to be considered as a pothole on the road
    
    # Device configuration (not rlly used)
    GPU_ID = "0"
    
    # Create directories if they don't exist
    @staticmethod
    def create_dirs():
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)