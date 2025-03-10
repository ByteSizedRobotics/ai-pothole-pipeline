import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cv2
from PIL import Image
import os
import argparse

def add_pixel_coordinates(image_path, tick_spacing=50, grid=True, output_path=None):
    """
    Add pixel coordinate axes to an existing image.
    
    Parameters:
    ----------
    image_path : str
        Path to the input image. This parameter is required.
    tick_spacing : int
        Spacing between coordinate ticks in pixels.
    grid : bool
        Whether to show grid lines.
    output_path : str, optional
        Path to save the output image. If None, generates a name based on the input.
    
    Returns:
    -------
    str: Path to the saved output image.
    """
    # Validate input path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Generate output path if not provided
    if output_path is None:
        filename, ext = os.path.splitext(image_path)
        output_path = f"{filename}_with_coords{ext}"
    
    # Load the image
    try:
        # Try OpenCV first (better for most image formats)
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
        else:
            # Fall back to PIL if OpenCV fails
            img = np.array(Image.open(image_path))
            height, width = img.shape[:2]
    except Exception as e:
        raise IOError(f"Error loading image {image_path}: {e}")
    
    # Create figure with the exact size in pixels
    dpi = 100  # dots per inch
    figsize = (width/dpi, height/dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display the image
    ax.imshow(img, extent=[0, width, height, 0])  # Note: origin at top-left
    
    # Set axis limits to match pixel coordinates
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Inverted y-axis to match image coordinates
    
    # Set tick spacing
    ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
    
    # Add grid if requested
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Label axes
    ax.set_xlabel('X Pixel Coordinate')
    ax.set_ylabel('Y Pixel Coordinate')
    ax.set_title(f'Image with Pixel Coordinates - {os.path.basename(image_path)}')
    
    # Tight layout to maximize image size
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)  # Close the figure to free memory
    
    print(f"Image with pixel coordinates saved to: {output_path}")
    return output_path

def display_interactive_coordinates(image_path, tick_spacing=50, grid=True):
    """
    Launch an interactive matplotlib window showing an image with pixel coordinates when hovering.
    This requires running in an interactive Python environment.
    
    Parameters:
    ----------
    image_path : str
        Path to the input image.
    tick_spacing : int
        Spacing between coordinate ticks in pixels.
    grid : bool
        Whether to show grid lines.
    """
    # Validate input path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    # Load the image
    try:
        # Try OpenCV first
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
        else:
            # Fall back to PIL
            img = np.array(Image.open(image_path))
            height, width = img.shape[:2]
    except Exception as e:
        raise IOError(f"Error loading image {image_path}: {e}")
    
    # Create figure with the appropriate size
    dpi = 100
    figsize = (width/dpi, height/dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display the image
    ax.imshow(img, extent=[0, width, height, 0])
    
    # Set axis limits to match pixel coordinates
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    
    # Set tick spacing
    ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
    
    # Add grid if requested
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Label axes
    ax.set_xlabel('X Pixel Coordinate')
    ax.set_ylabel('Y Pixel Coordinate')
    ax.set_title(f'Interactive Pixel Coordinates - {os.path.basename(image_path)}')
    
    # Add coordinate display on hover
    coord_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    def update_coords(event):
        if event.inaxes:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            pixel_info = ''
            # Try to get pixel value if within bounds
            if 0 <= x < width and 0 <= y < height:
                if len(img.shape) == 3:  # Color image
                    pixel = img[y, x]
                    pixel_info = f" | RGB: {pixel[0]},{pixel[1]},{pixel[2]}"
                else:  # Grayscale
                    pixel_info = f" | Value: {img[y, x]}"
            
            coord_text.set_text(f'x={x}, y={y}{pixel_info}')
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', update_coords)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add pixel coordinates to an image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    add_pixel_coordinates(
        image_path=args.image_path,
        tick_spacing=50,
        grid=True
    )