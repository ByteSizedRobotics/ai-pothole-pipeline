# utils/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import os

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
    filtered_detections = pipeline_output['filtered_detections']
    
    # Create figure with 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original image with all pothole detections
    ax[0].imshow(image)
    ax[0].set_title('Original Image with All Potholes')
    ax[0].axis('off')
    
    for confidence, bbox, _, _ in filtered_detections:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor='yellow', linewidth=2)
        ax[0].add_patch(rect)
        ax[0].text(x1, y1-5, f"{confidence:.2f}", color='yellow', 
                  bbox=dict(facecolor='black', alpha=0.7))
    
    # 2. Road segmentation
    road_vis = np.zeros((road_mask.shape[0], road_mask.shape[1], 3), dtype=np.uint8)
    road_color = np.array(cityscapes_classes[1][1], dtype=np.uint8)  # Road color
    road_vis[road_mask == 1] = road_color
    ax[1].imshow(road_vis)
    ax[1].set_title('Road Segmentation')
    ax[1].axis('off')
    
    # 3. Combined visualization with filtered potholes
    image_array = np.array(image)
    
    # Create overlay with road
    road_overlay = image_array.copy()
    overlay_mask = np.stack([road_mask, road_mask, road_mask], axis=2) * 0.3
    road_overlay = road_overlay * (1 - overlay_mask) + road_color * overlay_mask
    road_overlay = road_overlay.astype(np.uint8)
    
    ax[2].imshow(road_overlay)
    ax[2].set_title('Filtered Potholes')
    ax[2].axis('off')
    
    # Draw filtered pothole bounding boxes
    for confidence, bbox, is_on_road, overlap in filtered_detections:
        x1, y1, x2, y2 = bbox
        
        if is_on_road:
            # Green for potholes on road
            color = 'green'
            label = f"On Road ({overlap:.1f}%)"
        else:
            # Red for potholes not on road
            color = 'red'
            label = f"Not on Road ({overlap:.1f}%)"
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor=color, linewidth=2)
        ax[2].add_patch(rect)
        ax[2].text(x1, y1-5, label, color=color, fontsize=8,
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
        for i, (confidence, bbox, is_on_road, overlap) in enumerate(filtered_detections):
            status = "ON road" if is_on_road else "NOT on road"
            f.write(f"Pothole {i+1}: {status} (confidence: {confidence:.2f}, overlap: {overlap:.2f}%)\n")
            f.write(f"    Bounding box: {bbox}\n")