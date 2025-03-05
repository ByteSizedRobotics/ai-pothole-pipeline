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

def visualize_pipeline_results(pipeline_output, save_path=None):
    """
    Visualize the results of the complete pipeline.
    
    Args:
        pipeline_output: Output dictionary from the pipeline
        save_path: Optional path to save the visualization
    """
    # Extract data from pipeline output
    image = pipeline_output['image']
    road_mask = pipeline_output['road_mask']
    full_segmentation = pipeline_output['full_segmentation']
    filtered_detections = pipeline_output['filtered_detections']
    
    # Create figure with 4 subplots
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    
    # 1. Original image 
    ax[0, 0].imshow(image)
    ax[0, 0].set_title('Original Image')
    ax[0, 0].axis('off')
    
    # 2. Full Segmentation
    decode_fn = Cityscapes.decode_target
    colorized_preds = decode_fn(full_segmentation).astype('uint8')
    colorized_preds = Image.fromarray(colorized_preds)

    ax[0, 1].imshow(colorized_preds)
    ax[0, 1].set_title('Full Segmentation')
    ax[0, 1].axis('off')

    class_colors = [color for _, color in cityscapes_classes.values()]
    class_names = [name for name, _ in cityscapes_classes.values()]
    
    patches = [mpatches.Patch(color=np.array(color)/255.0, label=class_name) 
               for class_name, color in zip(class_names, class_colors)]
    
    ax[0, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), 
                    loc='upper left', borderaxespad=0.)
    
    # 3. Road segmentation
    road_vis = np.zeros((road_mask.shape[0], road_mask.shape[1], 3), dtype=np.uint8)
    road_color = np.array(cityscapes_classes[1][1], dtype=np.uint8)  # Road color
    road_vis[road_mask == 1] = road_color
    ax[1, 0].imshow(road_vis)
    ax[1, 0].set_title('Road Segmentation')
    ax[1, 0].axis('off')
    
    # 4. Combined visualization with filtered potholes
    image_array = np.array(image)
    
    # Create overlay with road
    road_overlay = image_array.copy()
    overlay_mask = np.stack([road_mask, road_mask, road_mask], axis=2) * 0.3
    road_overlay = road_overlay * (1 - overlay_mask) + road_color * overlay_mask
    road_overlay = road_overlay.astype(np.uint8)
    
    ax[1, 1].imshow(road_overlay)
    ax[1, 1].set_title('Potholes on Road')
    ax[1, 1].axis('off')
    
    # Draw filtered pothole bounding boxes
    for confidence, bbox, is_on_road, _ in filtered_detections:
        x1, y1, x2, y2 = bbox
        
        if is_on_road:
            # Green for potholes on road
            color = 'green'
            label = f"On Road (confidence: {confidence:.2f})"
        else:
            # Red for potholes not on road
            color = 'red'
            label = f"Not on Road (confidence: {confidence:.2f})"
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor=color, linewidth=2)
        ax[1, 1].add_patch(rect)
        ax[1, 1].text(x1, y1-5, label, color=color, fontsize=8,
                  bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def save_results_as_images(pipeline_output, output_dir):
    """
    Save the results of the pipeline as individual image files.
    
    Args:
        pipeline_output: Output dictionary from the pipeline
        output_dir: Directory to save the results
    """
    # Extract data
    image = pipeline_output['image']
    full_segmentation = pipeline_output['full_segmentation']
    road_mask = pipeline_output['road_mask']
    filtered_detections = pipeline_output['filtered_detections']
    image_path = pipeline_output['image_path']
    
    # Create base filename
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save original image with pothole detections
    image_with_potholes = image.copy()
    draw = ImageDraw.Draw(image_with_potholes)
    
    for _, bbox, is_on_road, _ in filtered_detections:
        x1, y1, x2, y2 = bbox
        color = "green" if is_on_road else "red"
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
    
    image_with_potholes.save(os.path.join(output_dir, f"{basename}_detections.png"))

    # Save full segmentation
    decode_fn = Cityscapes.decode_target
    colorized_preds = decode_fn(full_segmentation).astype('uint8')
    colorized_preds = Image.fromarray(colorized_preds)
    colorized_preds.save(os.path.join(output_dir, f"{basename}_full_segmentation.png"))

    # Save road mask
    road_vis = np.zeros((road_mask.shape[0], road_mask.shape[1], 3), dtype=np.uint8)
    road_color = np.array(cityscapes_classes[1][1], dtype=np.uint8)
    road_vis[road_mask == 1] = road_color
    road_img = Image.fromarray(road_vis)
    road_img.save(os.path.join(output_dir, f"{basename}_road_mask.png"))
    
    # Create results summary text file
    with open(os.path.join(output_dir, f"{basename}_results.txt"), 'w') as f:
        f.write(f"Results for {image_path}\n")
        f.write(f"Total potholes detected: {len(filtered_detections)}\n")
        
        on_road_count = sum(1 for _, _, is_on_road, _ in filtered_detections if is_on_road)
        f.write(f"Potholes on road: {on_road_count}\n")
        f.write(f"Potholes not on road: {len(filtered_detections) - on_road_count}\n\n")
        
        f.write("Detailed results:\n")
        for i, (confidence, bbox, is_on_road, num_corner_on_road) in enumerate(filtered_detections):
            status = "ON road" if is_on_road else "NOT on road"
            f.write(f"Pothole {i+1}: {status} (confidence: {confidence:.2f})\n")
            f.write(f"    Bounding box: {bbox}\n")
            f.write(f"    Number of corners on road: {num_corner_on_road}\n")