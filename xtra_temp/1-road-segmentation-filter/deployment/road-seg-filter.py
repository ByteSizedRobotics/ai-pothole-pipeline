import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.patches as mpatches

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DeepLabV3Plus-PyTorch')))

import utils
import network
from datasets import VOCSegmentation, Cityscapes

# Define Cityscapes class colors and labels
cityscapes_classes = {
    0: ('unlabeled', (0, 0, 0)),
    1: ('road', (128, 64, 128)),
    2: ('sidewalk', (244, 35, 232)),
    3: ('building', (70, 70, 70)),
    4: ('wall', (102, 102, 156)),
    5: ('fence', (190, 153, 153)),
    6: ('pole', (153, 153, 153)),
    7: ('traffic light', (250, 170, 30)),
    8: ('traffic sign', (220, 220, 0)),
    9: ('vegetation', (107, 142, 35)),
    10: ('terrain', (152, 251, 152)),
    11: ('sky', (70, 130, 180)),
    12: ('person', (220, 20, 60)),
    13: ('rider', (255, 0, 0)),
    14: ('car', (0, 0, 142)),
    15: ('truck', (0, 0, 70)),
    16: ('bus', (0, 60, 100)),
    17: ('train', (0, 80, 100)),
    18: ('motorcycle', (0, 0, 230)),
    19: ('bicycle', (119, 11, 32))
}

def create_road_mask(segmentation_output):
    """
    Create a binary mask for road areas from segmentation output.
    
    Args:
        segmentation_output: The predicted class index for each pixel (H, W)
    
    Returns:
        Binary mask where 1 = road, 0 = not road
    """
    # Road class index is 1 in Cityscapes
    road_mask = np.zeros_like(segmentation_output, dtype=np.uint8)
    road_mask[segmentation_output == 1] = 1
    return road_mask

def check_potholes_on_road(road_mask, pothole_bboxes):
    """
    Check if detected potholes are on the road.
    
    Args:
        road_mask: Binary mask where 1 = road, 0 = not road
        pothole_bboxes: List of pothole bounding boxes [x1, y1, x2, y2]
        
    Returns:
        List of booleans indicating if each pothole is on the road
        List of overlap percentages for each pothole with the road
    """
    on_road = []
    overlap_percentages = []
    
    for bbox in pothole_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get the road mask in the pothole region
        pothole_area = (x2 - x1) * (y2 - y1)
        if pothole_area == 0:
            on_road.append(False)
            overlap_percentages.append(0.0)
            continue
            
        patch = road_mask[y1:y2, x1:x2]
        
        # Count road pixels in the bbox
        road_pixels = np.sum(patch)
        
        # Calculate percentage of pothole that is on the road
        overlap_percentage = (road_pixels / pothole_area) * 100
        
        # Consider a pothole "on the road" if at least 50% of it overlaps with road
        is_on_road = overlap_percentage >= 50.0
        
        on_road.append(is_on_road)
        overlap_percentages.append(overlap_percentage)
    
    return on_road, overlap_percentages

def visualize_segmentation_with_potholes(original_img, road_mask, pothole_bboxes, on_road):
    """
    Visualize the original image, road segmentation, and pothole detections.
    
    Args:
        original_img: The original RGB image
        road_mask: Binary mask where 1 = road, 0 = not road
        pothole_bboxes: List of pothole bounding boxes [x1, y1, x2, y2]
        on_road: List of booleans indicating if each pothole is on the road
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display original image
    ax[0].imshow(original_img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Display road mask
    road_vis = np.zeros((road_mask.shape[0], road_mask.shape[1], 3), dtype=np.uint8)
    road_color = np.array(cityscapes_classes[1][1], dtype=np.uint8)  # Road color
    road_vis[road_mask == 1] = road_color
    ax[1].imshow(road_vis)
    ax[1].set_title('Road Segmentation')
    ax[1].axis('off')
    
    # Display original with road overlay and potholes
    road_overlay = np.copy(np.array(original_img))
    # Add semi-transparent road overlay
    overlay_mask = np.stack([road_mask, road_mask, road_mask], axis=2) * 0.3
    road_overlay = road_overlay * (1 - overlay_mask) + road_color * overlay_mask
    
    # Convert to uint8 for display
    road_overlay = road_overlay.astype(np.uint8)
    
    ax[2].imshow(road_overlay)
    
    # Draw pothole bounding boxes
    for i, bbox in enumerate(pothole_bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        if on_road[i]:
            # Green for potholes on road
            color = 'green'
            label = 'On Road'
        else:
            # Red for potholes not on road
            color = 'red'
            label = 'Not on Road'
            
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor=color, linewidth=2)
        ax[2].add_patch(rect)
        ax[2].text(x1, y1-5, label, color=color, fontsize=8, 
                  bbox=dict(facecolor='white', alpha=0.7))
    
    ax[2].set_title('Pothole Detection')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    input_path = "test_images/test.jpg"  # ADD YOUR PATH TO IMAGE OR DIRECTORY
    pothole_detector_path = "path_to_pothole_detector_model"  # Replace with your pothole detector model path
    dataset_name = 'cityscapes'
    model_name = 'deeplabv3plus_resnet101'
    separable_conv = False
    output_stride = 16
    save_val_results_to = 'test_results'
    crop_val = False
    crop_size = 513
    ckpt = 'C:/Users/natha/Documents/GitHub/ai-pothole-models/deployment/checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth'
    gpu_id = '0'

    if dataset_name.lower() == 'voc':
        num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif dataset_name.lower() == 'cityscapes':
        num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(input_path):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(input_path, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(input_path):
        image_files.append(input_path)
    
    # Set up semantic segmentation model
    model = network.modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)
    if separable_conv and 'plus' in model_name:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if ckpt is not None and os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    if crop_val:
        transform = T.Compose([
                T.Resize(crop_size),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        
    if save_val_results_to is not None:
        os.makedirs(save_val_results_to, exist_ok=True)
        
    # # Load your pothole detector model here
    # # This is a placeholder. You should replace it with your actual pothole detector
    # # Example:
    # pothole_detector = YourPotholeDetectorClass()
    # pothole_detector.load_weights(pothole_detector_path)
        
    with torch.no_grad():
        model = model.eval()
        for img_path in image_files:
            print(f"Processing: {img_path}")
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            
            # Load and prepare image for semantic segmentation
            original_image = Image.open(img_path).convert('RGB')
            img_tensor = transform(original_image).unsqueeze(0)  # To tensor of NCHW
            img_tensor = img_tensor.to(device)
            
            # Run semantic segmentation
            pred = model(img_tensor).max(1)[1].cpu().numpy()[0]  # HW
            
            # Create road mask
            road_mask = create_road_mask(pred)
            
            # # Detect potholes (replace with your actual pothole detection code)
            # # Example:
            # pothole_bboxes = pothole_detector.detect(original_image)
            
            # For demonstration, let's create some dummy pothole bounding boxes
            # Replace this with your actual pothole detection results
            # Format: [x1, y1, x2, y2]
            dummy_pothole_bboxes = [
                [100, 150, 150, 200],  # Example pothole 1
                [200, 250, 240, 290],  # Example pothole 2
                [350, 400, 400, 450]   # Example pothole 3
            ]
            
            # Check if potholes are on the road
            on_road, overlap_percentages = check_potholes_on_road(road_mask, dummy_pothole_bboxes)
            
            # Visualize results
            visualize_segmentation_with_potholes(original_image, road_mask, dummy_pothole_bboxes, on_road)
            
            # Print results
            for i, (is_on_road, overlap) in enumerate(zip(on_road, overlap_percentages)):
                status = "ON road" if is_on_road else "NOT on road"
                print(f"Pothole {i+1}: {status} (overlap: {overlap:.2f}%)")
            
            # Save results if needed
            if save_val_results_to:
                road_vis = np.zeros((road_mask.shape[0], road_mask.shape[1], 3), dtype=np.uint8)
                road_color = np.array(cityscapes_classes[1][1], dtype=np.uint8)
                road_vis[road_mask == 1] = road_color
                
                road_img = Image.fromarray(road_vis)
                road_img.save(os.path.join(save_val_results_to, f"{img_name}_road_mask.png"))

if __name__ == '__main__':
    main()