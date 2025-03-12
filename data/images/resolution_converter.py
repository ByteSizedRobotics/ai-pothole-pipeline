import cv2
import os
import argparse
import glob

def resize_image(input_path, output_path, target_width=1280, target_height=720):
    """
    Resize an image to a specified dimensions)
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the resized image
        target_width (int): Target width in pixels (default: 1280)
        target_height (int): Target height in pixels (default: 720)
    """
    # Read the image
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return False
    
    # Get the original dimensions
    original_height, original_width = img.shape[:2]
    
    # Print original dimensions
    print(f"Original dimensions: {original_width}x{original_height}")
    
    # Resize the image
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    # Save the resized image
    cv2.imwrite(output_path, resized_img)
    
    print(f"Resized image saved to {output_path}")
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Resize images from 3280x2464 to 1280x720")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--output", "-o", help="Output directory (default: ./resized)")
    parser.add_argument("--width", "-w", type=int, default=1280, help="Target width (default: 1280)")
    parser.add_argument("--height", "-ht", type=int, default=720, help="Target height (default: 720)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output if args.output else "./resized"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process single file or directory
    if os.path.isfile(args.input):
        # Process single file
        filename = os.path.basename(args.input)
        output_path = os.path.join(output_dir, filename)
        resize_image(args.input, output_path, args.width, args.height)
    elif os.path.isdir(args.input):
        # Process all images in directory
        image_files = glob.glob(os.path.join(args.input, "*.jpg")) + \
                      glob.glob(os.path.join(args.input, "*.jpeg")) + \
                      glob.glob(os.path.join(args.input, "*.png"))
        
        if not image_files:
            print(f"No image files found in {args.input}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            filename = os.path.basename(image_file)
            output_path = os.path.join(output_dir, filename)
            resize_image(image_file, output_path, args.width, args.height)
        
        print(f"All images processed. Resized images saved to {output_dir}")
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()