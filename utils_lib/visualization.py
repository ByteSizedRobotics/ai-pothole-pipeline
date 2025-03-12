import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
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

# Function to visualize the original image
def visualize_original_image(image, save_path, image_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    ax.set_title('Original Image')
    ax.axis('off')
    plt.savefig(os.path.join(save_path, f'{image_name}_0_original_image.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax

# Function to visualize the original detections
def visualize_pothole_detections(image, detections, save_path, image_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    ax.set_title('Original Detections')
    ax.axis('off')
    
    # Draw original pothole bounding boxes
    for i, (*bbox, confidence, classType) in enumerate(detections.xyxy[0].tolist()):
        x1, y1, x2, y2 = map(int, bbox)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor='blue', linewidth=1.5)
        ax.add_patch(rect)
        if classType == 0:
            label = f"#{i+1} (confidence: {confidence:.2f})"
        else:
            label = f"{classType} (confidence: {confidence:.2f})"
        ax.text(x1, y1-20, label, color='blue', fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.savefig(os.path.join(save_path, f'{image_name}_1_original_detections.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax

# Function to visualize the full segmentation
def visualize_full_segmentation(full_segmentation, save_path, image_name):
    decode_fn = Cityscapes.decode_target
    colorized_preds = decode_fn(full_segmentation).astype('uint8')
    colorized_preds = Image.fromarray(colorized_preds)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(colorized_preds)
    ax.set_title('Full Segmentation')
    ax.axis('off')
    
    plt.savefig(os.path.join(save_path, f'{image_name}_2_full_segmentation.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax, colorized_preds

# Function to visualize road segmentation
def visualize_road_segmentation(road_mask, save_path, image_name):
    road_vis = np.zeros((road_mask.shape[0], road_mask.shape[1], 3), dtype=np.uint8)
    road_color = np.array(cityscapes_classes[1][1], dtype=np.uint8)  # Road color
    road_vis[road_mask == 1] = road_color
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(road_vis)
    ax.set_title('Road Segmentation')
    ax.axis('off')
    
    plt.savefig(os.path.join(save_path, f'{image_name}_3_road_segmentation.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax, road_vis, road_color

# Function to visualize filtered detections
def visualize_filtered_detections(image, road_mask, filtered_detections, save_path, image_name):
    image_array = np.array(image)
    road_color = np.array(cityscapes_classes[1][1], dtype=np.uint8)  # Road color
    
    # Create overlay with road
    road_overlay = image_array.copy()
    overlay_mask = np.stack([road_mask, road_mask, road_mask], axis=2) * 0.3
    road_overlay = road_overlay * (1 - overlay_mask) + road_color * overlay_mask
    road_overlay = road_overlay.astype(np.uint8)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(road_overlay)
    ax.set_title('Potholes on Road')
    ax.axis('off')

    # Draw filtered pothole bounding boxes
    for i, (confidence, bbox, is_on_road, percentage) in enumerate(filtered_detections):
        x1, y1, x2, y2 = bbox
        
        if is_on_road:
            # Green for potholes on road
            color = 'green'
            label = f"#{i+1} On Road ({percentage:.2f})"
        else:
            # Red for potholes not on road
            color = 'red'
            label = f"#{i+1} Not on Road ({percentage:.2f})"
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x1, y1-20, label, color=color, fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.savefig(os.path.join(save_path, f'{image_name}_4_filtered_detections.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax, road_overlay

# Function to visualize depth results
def visualize_depth_results(depth_results, save_path, image_name):
    cropped_potholes = depth_results.get('cropped_potholes', [])
    depth_maps = depth_results.get('depth_maps', [])
    relative_depths = depth_results.get('relative_depths', [])
    normalized_depths = depth_results.get('normalized_depths', [])

    valid_indices = [i for i, crop in enumerate(cropped_potholes) if crop is not None] # Filter out None values (potholes not on road)
    potholes_on_road = len(valid_indices)
    
    if potholes_on_road == 0: # handle case where no potholes are on the road
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No potholes detected on road", fontsize=12, ha='center')
        ax.axis('off')
    else:
        # Create figure with 2 columns per pothole (image and depth map)
        fig, axes = plt.subplots(potholes_on_road, 2, figsize=(8, 3 * potholes_on_road))
        
        # Handle case where only one pothole is detected (axes won't be 2D)
        if potholes_on_road == 1:
            axes = np.array([axes])  # Make it 2D
            
        fig.suptitle('Pothole Crops and Depth Maps', fontsize=12)
        
        # Plot each pothole and its depth map
        for idx, i in enumerate(valid_indices):
            axes[idx, 0].imshow(cv2.cvtColor(cropped_potholes[i], cv2.COLOR_BGR2RGB))
            
            if i < len(relative_depths):
                relative_depth = relative_depths[i]
                normalized_depth = normalized_depths[i]
                axes[idx, 0].set_title(f"Pothole #{i+1}\nDepth: {relative_depth:.2f} (Normalized: {normalized_depth:.2f})")
            else:
                axes[idx, 0].set_title(f"Pothole #{i+1}")
            
            axes[idx, 0].axis('off')
            
            if i < len(depth_maps) and depth_maps[i] is not None:
                im = axes[idx, 1].imshow(depth_maps[i], cmap='viridis')
                axes[idx, 1].set_title(f"Depth Map #{i+1}")
                axes[idx, 1].axis('off')
                
                plt.colorbar(im, ax=axes[idx, 1], fraction=0.046, pad=0.04)
            else:
                axes[idx, 1].text(0.5, 0.5, "No depth map available", ha='center')
                axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    depth_save_path = os.path.join(save_path, f'{image_name}_5_depth_visualization.png')
    plt.savefig(depth_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig

# Function to visualize area, depth and categorization results
def visualize_area_depth_results(pothole_areas, depth_estimations, pothole_categorizations, filtered_detections, save_path, image_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    on_road_indices = [i for i, (_, _, is_on_road, _) in enumerate(filtered_detections) if is_on_road] # Filter for potholes on road only

    if not on_road_indices:
        ax.text(0.5, 0.5, "No potholes detected on road", fontsize=12, ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        areas = [pothole_areas[i] for i in on_road_indices]
        depths = [depth_estimations['normalized_depths'][i] for i in on_road_indices]
        categories = [pothole_categorizations['categories'][i] for i in on_road_indices]
        scores = [pothole_categorizations['scores'][i] for i in on_road_indices]
        
        scatter = ax.scatter(areas, depths, s=100, alpha=0.7, c=scores, cmap='viridis')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Categorization Score')
        
        for i, idx in enumerate(on_road_indices):
            ax.annotate(f"#{idx+1}: {categories[i]}", 
                        xy=(areas[i], depths[i]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        ax.set_xlabel('Pothole Area (normalized)')
        ax.set_ylabel('Pothole Depth (normalized)')
        ax.set_title('Pothole Area vs Depth with Categorization')
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        unique_categories = list(set(categories))
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=plt.cm.viridis(0.2 + i/len(unique_categories)), 
                              markersize=10, label=cat) 
                  for i, cat in enumerate(unique_categories)]
        ax.legend(handles=handles, title="Categories", loc="best")
    
    plt.tight_layout()
    
    area_depth_save_path = os.path.join(save_path, f'{image_name}_6_area_depth_visualization.png')
    plt.savefig(area_depth_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return fig

# TODO: NATHAN remove the visualize results part which combines the imgs into 1 figure?
def visualize_combined_results(pipeline_output, save_path):
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
    for i, (*bbox, confidence, classType) in enumerate(detections.xyxy[0].tolist()):
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box in blue for original detections
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor='blue', linewidth=1.5)
        ax[0, 1].add_patch(rect)
        if classType == 0:
            label = f"#{i+1} (confidence: {confidence:.2f})"
        else:
            label = f"{classType} (confidence: {confidence:.2f})"
        ax[0, 1].text(x1, y1-20, label, color='blue', fontsize=8,
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
    for i, (confidence, bbox, is_on_road, percentage) in enumerate(filtered_detections):
        x1, y1, x2, y2 = bbox
        
        if is_on_road:
            # Green for potholes on road
            color = 'green'
            label = f"#{i+1} On Road ({percentage:.2f})"
        else:
            # Red for potholes not on road
            color = 'red'
            label = f"#{i+1} Not on Road ({percentage:.2f})"
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                            edgecolor=color, linewidth=1.5)
        ax[1, 2].add_patch(rect)
        ax[1, 2].text(x1, y1-20, label, color=color, fontsize=8,
                  bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    

    combined_save_path = os.path.join(save_path, f'{os.path.splitext(os.path.basename(image_path))[0]}_combined.png')
    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')    
    plt.close()

def create_results_file(filtered_detections, pothole_areas, depth_estimations, pothole_categorizations, save_path, image_path):
    """
    Save the results of the pipeline as individual image files.
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
            f.write(f"Pothole #{i+1}: {status}\n")
            f.write(f"    Bounding box: {[round(coord, 2) for coord in bbox]}\n")
            f.write(f"    Detection Confidence: {confidence:.2f}\n")
            f.write(f"    % Pixels in box which are road: {percentage:.2f}\n")
            if is_on_road:
                f.write(f"    Estimated Area: {round(pothole_areas[i], 4)}\n")
            else:
                f.write(f"    Estimated Area: NA\n")
            if is_on_road:
                f.write(f"    Estimated Depth: {round(depth_estimations['normalized_depths'][i], 4)}\n")
            else:
                f.write(f"    Estimated Depth: NA\n")
            f.write(f"    Categorization: {pothole_categorizations['categories'][i]}\n")
            f.write(f"    Categorization Score: {round(pothole_categorizations['scores'][i], 4)}\n")



