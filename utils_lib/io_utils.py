import os
from glob import glob

def get_image_files(input_path):
    """
    Get a list of image files from a directory or a single file.
    
    Args:
        input_path: Path to an image or directory of images
    
    Returns:
        List of image file paths
    """
    image_files = []
    
    if os.path.isdir(input_path):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(input_path, '**/*.%s'%(ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(input_path):
        image_files.append(input_path)
    
    return image_files