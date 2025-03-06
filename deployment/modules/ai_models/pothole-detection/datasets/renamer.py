import os

def rename_images(folder_path, pattern):
    files = os.listdir(folder_path)
    
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Rename each image file
    for i, filename in enumerate(image_files):
        file_extension = os.path.splitext(filename)[1]
        
        new_filename = f"{pattern}{i+1}{file_extension}"
        
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        
        os.rename(old_file, new_file)
        print(f"Renamed {old_file} to {new_file}")

def create_empty_text_files_for_images(folder_path, pattern):
    files = os.listdir(folder_path)
    
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for i, filename in enumerate(image_files):
        file_name = os.path.splitext(filename)[0]
        # text_filename = f"{file_name}{i+1}.txt"

        text_filename = f"{pattern}{i+1}.txt"
        
        text_file_path = os.path.join(folder_path, text_filename)
        
        open(text_file_path, 'w').close()
        print(f"Created text file {text_file_path}")

folder_path = 'C:\\Users\\mike\\GitHub\\ai-pothole-models\\pothole-detection\\datasets\\custom\\CustomSet\\2.0Feb22\\images'
pattern = 'normal_'
# rename_images(folder_path, pattern)
create_empty_text_files_for_images(folder_path, pattern)