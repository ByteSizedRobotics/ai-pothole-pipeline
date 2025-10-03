<h1 align="center">
AI-Pothole-Detection-Pipeline
</h1>

## Overview
The AI Pothole Detection Pipeline is an end-to-end pipeline designed to detect potholes and perform pothole analysis on a given image. This AI project is part of the Autonomous Pothole Detection Rover, designed to autonoumously navigate along sidewalks while performing real-time pothole detection and further processing of the potholes on the server-side. Once an image is provided, it passes through 6 pipeline stages for comprehensive pothole analysis.

1. **Pothole Detection**: Identifies potential potholes in images using YOLOv5
2. **Road Segmentation**: Creates a road mask using DeepLabV3+ segmentation model
3. **Pothole Filtering**: Filters pothole detections based on road segmentation
4. **Area Estimation**: Calculates a rough score estimation for the surface area of detected potholes
5. **Depth Estimation**: Estimates pothole depth using DepthAnythingV2
6. **Pothole Categorization**: Classifies potholes based on area and depth metrics

![image](https://github.com/user-attachments/assets/b8fb04f0-22ae-447a-be93-f2df64a94c5d)

## Project Structure
```
project_root/
├── main.py                     # Main script to run the complete pipeline
├── config.py                   # Configuration parameters
├── modules/
│   ├── ai_models/
│   │   ├── DeepLabV3Plus/      # DeepLavV3+ model for road segmentation
│   │   │   └── checkpoints/
│   │   ├── DepthAnythingV2/    # DepthAnythingV2 model for depth estimation
│   │   │   └── checkpoints/
│   │   └── pothole-detection/  # YOLOv5 model and training runs completed for fine-tuning
│   │   │   └── yolov5/
│   ├── area_estimation/        # Calculations for area scaling factor
│   ├── road_segmentation.py    # Road segmentation using DeepLabV3+
│   ├── pothole_detection.py    # Pothole detection using YOLOv5
│   └── depth_estimation.py     # Depth estimation using DepthAnythingV2
├── pipeline/
│   ├── pothole_detection_stage.py      # Stage 1: Detect potholes
│   ├── road_segmentation_stage.py      # Stage 2: Road segmentation
│   ├── pothole_filtering_stage.py      # Stage 3: Filter potholes based on segmentation
│   ├── area_estimation_stage.py        # Stage 4: Estimate area of potholes
│   ├── depth_estimation_stage.py       # Stage 5: Estimate depth of potholes
│   └── pothole_categorization_stage.py # Stage 6: Categorize potholes based on depth and area
└── utils_lib/
    ├── DeepLabV3Plus/          # Contains the DeepLabV3+ required libraries for visualizations
    │   ├── io_utils.py
    │   └── visualization.py
    ├── visualization.py        # To save the visualizations/results of the pipeline
    └── io_utils.py             # Used to get images from a directory or a single file
```

## Instructions
### Configuration Details
The system uses a `Config` class (in `config.py`) with the following key parameters:
- **INPUT_PATH**: `"data/images"` - Directory containing input images
- **OUTPUT_PATH**: `"data/results"` - Directory for output results
- **IMAGE_RESOLUTION**: `(3280, 2464)` - Default image resolution (as of right now, only image resolution supported is 3280x2464)
- **DEPTH_ANYTHING_ENCODER**: `'vitl'` - Vision Transformer encoder size (options: 'vits', 'vitb', 'vitl')
- **MIN_PIXELS_ROAD_THRESHOLD**: `0.60` - Minimum percentage of pixels in bounding box required to be classified as road for pothole filtering/validation

### Prerequisites
Before running the pipeline, ensure you have:

1. Python 3.8+ installed
2. Cloned the GitHub repository locally
3. Installed required packages provided in `requirements.txt` (can create a venv and install the packages)
4. Downloaded and extracted the DeepLabV3+ Cityscapes ResNet101 weights from the DeepLabV3+ [repository](https://github.com/VainF/DeepLabV3Plus-Pytorch/) (Download [HERE](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view). Save the weights file under `modules/ai_models/DeepLabV3Plus/checkpoints` (you might have to create the folder `checkpoints`).
5. Downloaded the desired Depth-Anything-V2 weight file (recommended model is LARGE, but BASE and SMALL are also options). (Download [HERE](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view)). Save the weights file under `modules/ai_models/Depth-Anything-V2/checkpoints` (you might have to create the folder `checkpoints`).

### Running Application
The **IMAGE_RESOLUTION** should be the same as specified in the `config.py`. You can resize the images using the `resolution_converter.py` file under `data/images`. The resolution can be specified at runtime by using the command line arguments. 

**As of right now, the only supported resolution is 3280x2464... the scaling factors for area estimation were calculated based on this resolution, so using a different image resolution may cause worse performance and estimations for the area.**
```bash
# Process a single image
python main.py --input data/images/road_image.jpg

# Process all images in a directory and custom image resolution
python main.py --input data/images/batch1/ --resolution 1280x720

# Specify a custom output location
python main.py --input data/images/road_image.jpg --output results/analysis/
```

### Output Files
The following files will be saved after running the pipeline for a given image. The images are the output results from each pipeline stage. The `combined.png` provides an overview of the pipeline processing for that image and the `results.txt` provides a summary of the results.
- [imageName]_0_original_image.png
- [imageName]_1_original_detections.png
- [imageName]_2_full_segmentation.png
- [imageName]_3_road_segmentation.png
- [imageName]_4_filtered_detections.png
- [imageName]_5_pothole_area_scores.png
- [imageName]_6_depth_scores.png
- [imageName]_7_pothole_categories.png
- [imageName]_combined.png
- [imageName]_results.txt



### Pothole Detection Deployment
Since this project is integrated with an Autonomous Pothole Detection Rover, the pothole detection model is deployed and runs on a Raspberry Pi. There are files under `modules/ai_models/pothole_detection/deployment` which are related to deploying the pothole detection model on hardware components. The actual pothole app runs off-site on SW when the rover navigates. The deployed app is under `docker/main.py` and runs in a docker container.

### Deployment in Autonomous Pothole Detection Rover Project ###
Within the /docker folder, there is a main.py and a Dockerfile which is used in the autonomous pothole detection rover project. The goal is to startup a docker container running off-site on the software while the rover is navigating autonomously and sending back a continuous video stream. This video stream is passed to this main script which performs the AI analysis pipeline stages in real-time. The detection results in the input frames are written directly to the database via API endpoints to allow the user to view the data on the UI. The following commands can be used to build/run the Docker container. NOTE: this docker container requires the IP address of the raspi running on the rover which sends back the video stream, as well as the WebRTC port number to connect to the live stream.
```bash
# BUILD
docker build -t pothole-detection-app docker/

# Run without optional flags
docker run --gpus all -it --name pothole-detection-app-container pothole-detection-app 

# Run with IP and WebRTC flags
docker run --gpus all -it --name pothole-detection-app-container pothole-detection-app -e RASPI_IP=... -e WEBRTC_PORT=...
```
The links below provide the repostitories related to the autonomous pothole detection rover project:
- [Code running on the raspberry pi for rover operation](https://github.com/ByteSizedRobotics/autonomous-navigation)
- [UI and backend logic](https://github.com/ByteSizedRobotics/rover-ui)
- [This current AI repository](https://github.com/ByteSizedRobotics/ai-pothole-pipeline)

## Disclaimer
This project is for non-commercial use only. It utilizes the YOLOv5, DeepLabV3+, and Depth-Anything-V2 models.

The source code and all credit for these models belong to their respective authors and organizations. As a result, this project is also subject to the licenses governing these models:

**YOLOv5**
- Author: Glenn Jocher (Ultralytics)
- Purpose in Project: small model was fine tuned and trained to detect potholes
- [Repository](https://github.com/ultralytics/yolov5)
- [AGPL-3.0 License](modules/ai_models/pothole-detection/yolov5/LICENSE)


**DeepLabV3+**
- Purpose in Project: model used for segmentation with Cityscapes classes and weights file (can be downloaded from the repository linked below)
- [Repository](https://github.com/VainF/DeepLabV3Plus-Pytorch/)
- [MIT License](modules/ai_models/DeepLabV3Plus/LICENSE)

**Depth-Anything-V2**
- Authors: Yang, Lihe; Kang, Bingyi; Huang, Zilong; Zhao, Zhen; Xu, Xiaogang; Feng, Jiashi; Zhao, Hengshuang
- Purpose in Project: model used for pothole depth estimation with the Depth-Anything-V2-Large model 
- [Repository](https://github.com/DepthAnything/Depth-Anything-V2/)
- Small model: [Apache-2.0 License](modules/ai_models/DepthAnythingV2/LICENSE)
- Base, Large, Giant models: CC-BY-NC-4.0 License (non-commercial use only).

**NOTE**
- Docker folder contains a more prod like app used for an Autonomous Pothole Detection Rover project
- Make sure you have the correct torch version installed (i.e support for CUDA)
