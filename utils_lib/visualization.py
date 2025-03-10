# utils/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import os

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))) 
from DeepLabV3Lib import Cityscapes

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

def visualize_pipeline_results(pipeline_output, save_path):
    """
    Visualize the results of the complete pipeline.
    
    Args:
        pipeline_output: Output dictionary from the pipeline
        save_path: Optional path to save the visualization
    """
    # Extract data from pipeline output
    image = pipeline_output['image']
    image_path = pipeline_output['image_path']
    road_mask = pipeline_output['road_mask']
    full_segmentation = pipeline_output['full_segmentation']
    detections = pipeline_output['detections']
    filtered_detections = pipeline_output['filtered_detections']
    
    # Create figure with 5 subplots
    fig, ax = plt.subplots(2, 3, figsize=(25, 12))
    
    # 1. Original image 
    ax[0, 0].imshow(image)
    ax[0, 0].set_title('Original Image')
    ax[0, 0].axis('off')

    # 2. Original YOLO Detections
    ax[0, 1].imshow(image)
    ax[0, 1].set_title('Original YOLO Detections')
    ax[0, 1].axis('off')
    
    # Draw original YOLO pothole bounding boxes
    for *bbox, confidence, classType in detections.xyxy[0].tolist():
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box in blue for original detections
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor='blue', linewidth=2)
        ax[0, 1].add_patch(rect)
        if classType == 0:
            label = f"Pothole (confidence: {confidence:.2f})"
        else:
            label = f"{classType} (confidence: {confidence:.2f})"
        ax[0, 1].text(x1, y1-35, label, color='blue', fontsize=8,
                  bbox=dict(facecolor='white', alpha=0.7))
    
    # 2. Full Segmentation
    decode_fn = Cityscapes.decode_target
    colorized_preds = decode_fn(full_segmentation).astype('uint8')
    colorized_preds = Image.fromarray(colorized_preds)

    ax[1, 0].imshow(colorized_preds)
    ax[1, 0].set_title('Full Segmentation')
    ax[1, 0].axis('off')

    class_colors = [color for _, color in cityscapes_classes.values()]
    class_names = [name for name, _ in cityscapes_classes.values()]
    
    patches = [mpatches.Patch(color=np.array(color)/255.0, label=class_name) 
               for class_name, color in zip(class_names, class_colors)]
    
    ax[1, 0].legend(handles=patches, bbox_to_anchor=(1.05, 1), 
                    loc='upper left', borderaxespad=0.)
    
    # 3. Road segmentation
    road_vis = np.zeros((road_mask.shape[0], road_mask.shape[1], 3), dtype=np.uint8)
    road_color = np.array(cityscapes_classes[1][1], dtype=np.uint8)  # Road color
    road_vis[road_mask == 1] = road_color
    ax[1, 1].imshow(road_vis)
    ax[1, 1].set_title('Road Segmentation')
    ax[1, 1].axis('off')
    
    # 5. Combined visualization with filtered potholes
    image_array = np.array(image)
    
    # Create overlay with road
    road_overlay = image_array.copy()
    overlay_mask = np.stack([road_mask, road_mask, road_mask], axis=2) * 0.3
    road_overlay = road_overlay * (1 - overlay_mask) + road_color * overlay_mask
    road_overlay = road_overlay.astype(np.uint8)
    
    ax[1, 2].imshow(road_overlay)
    ax[1, 2].set_title('Potholes on Road')
    ax[1, 2].axis('off')
    
    # Draw filtered pothole bounding boxes
    for confidence, bbox, is_on_road, percentage in filtered_detections:
        x1, y1, x2, y2 = bbox
        
        if is_on_road:
            # Green for potholes on road
            color = 'green'
            label = f"On Road (%: {percentage:.2f})"
        else:
            # Red for potholes not on road
            color = 'red'
            label = f"Not on Road (%: {percentage:.2f})"
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor=color, linewidth=4)
        ax[1, 2].add_patch(rect)
        ax[1, 2].text(x1, y1-35, label, color=color, fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    

    combined_save_path = os.path.join(save_path, f'{os.path.splitext(os.path.basename(image_path))[0]}_combined.png')
    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')    
    plt.close()
    #plt.show() # TODO: NATHAN show is not working??

    # Save individual plots for each step of the pipeline
    image_name = os.path.splitext(os.path.basename(image_path))[0]    
    # 1. Original image
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    plt.savefig(os.path.join(save_path, f'{image_name}_original_image.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Original YOLO Detections
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.imshow(image)
    ax2.set_title('Original YOLO Detections')
    ax2.axis('off')
    
    # Draw original YOLO pothole bounding boxes
    for *bbox, confidence, classType in detections.xyxy[0].tolist():
        x1, y1, x2, y2 = map(int, bbox)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor='blue', linewidth=2)
        ax2.add_patch(rect)
        if classType == 0:
            label = f"Pothole (confidence: {confidence:.2f})"
        else:
            label = f"{classType} (confidence: {confidence:.2f})"
        ax2.text(x1, y1-35, label, color='blue', fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.savefig(os.path.join(save_path, f'{image_name}_original_detections.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Full Segmentation
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.imshow(colorized_preds)
    ax3.set_title('Full Segmentation')
    ax3.axis('off')
    plt.savefig(os.path.join(save_path, f'{image_name}_full_segmentation.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Road segmentation
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.imshow(road_vis)
    ax4.set_title('Road Segmentation')
    ax4.axis('off')
    plt.savefig(os.path.join(save_path, f'{image_name}_road_segmentation.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Filtered potholes on road
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.imshow(road_overlay)
    ax5.set_title('Potholes on Road')
    ax5.axis('off')
    
    # Draw filtered pothole bounding boxes
    for confidence, bbox, is_on_road, percentage in filtered_detections:
        x1, y1, x2, y2 = bbox
        
        if is_on_road:
            color = 'green'
            label = f"On Road (%: {percentage:.2f})"
        else:
            color = 'red'
            label = f"Not on Road (%: {percentage:.2f})"
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor=color, linewidth=4)
        ax5.add_patch(rect)
        ax5.text(x1, y1-35, label, color=color, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.savefig(os.path.join(save_path, f'{image_name}_filtered_detections.png'), dpi=300, bbox_inches='tight')
    plt.close(fig5)

def create_results_file(filtered_detections, save_path, image_path):
    """
    Save the results of the pipeline as individual image files.
    
    Args:
        pipeline_output: Output dictionary from the pipeline
        output_dir: Directory to save the results
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]

    with open(os.path.join(save_path, f"{basename}_results.txt"), 'w') as f:
        f.write(f"Results for {image_path}\n")
        f.write(f"Total potholes detected: {len(filtered_detections)}\n")
        
        on_road_count = sum(1 for _, _, is_on_road, _ in filtered_detections if is_on_road)
        f.write(f"Potholes on road: {on_road_count}\n")
        f.write(f"Potholes not on road: {len(filtered_detections) - on_road_count}\n\n")
        
        f.write("Detailed results:\n")
        for i, (confidence, bbox, is_on_road, percentage) in enumerate(filtered_detections):
            status = "ON road" if is_on_road else "NOT on road"
            f.write(f"Pothole {i+1}: {status}\n")
            f.write(f"    Bounding box: {[round(coord, 2) for coord in bbox]}\n")
            f.write(f"    Detection Confidence: {confidence:.2f}\n")
            f.write(f"    % Pixels in box which are road: {percentage:.2f}\n")
