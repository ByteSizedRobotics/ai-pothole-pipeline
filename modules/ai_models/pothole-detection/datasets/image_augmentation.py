import cv2
import os
import albumentations as A

# Define augmentation pipeline with bounding box format (YOLO: [x_center, y_center, width, height] normalized)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.7),
    A.Rotate(limit=20, p=0.5),
    A.GaussNoise(p=0.4)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Input and output directories
input_img_dir = "custom\\1.1CombinedDataset\\raw\\customImagesNotAugmented\\images"
input_ann_dir = "custom\\1.1CombinedDataset\\raw\\customImagesNotAugmented\\labels"  # Directory containing .txt annotation files
output_img_dir = "custom\\1.1CombinedDataset\\raw\\customImagesAugmented\\images"
output_ann_dir = "custom\\1.1CombinedDataset\\raw\\customImagesAugmented\\labels"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_ann_dir, exist_ok=True)

# Loop through images
for img_name in os.listdir(input_img_dir):
    if not img_name.endswith(('.jpg', '.png')):  # Ensure it's an image
        continue

    img_path = os.path.join(input_img_dir, img_name)
    ann_path = os.path.join(input_ann_dir, os.path.splitext(img_name)[0] + ".txt")

    image = cv2.imread(img_path)
    height, width, _ = image.shape  # Image dimensions

    # Read bounding boxes from annotation file
    bboxes = []
    class_labels = []
    if os.path.exists(ann_path):
        with open(ann_path, "r") as file:
            for line in file.readlines():
                data = line.strip().split()
                class_id = int(data[0])  # Class label
                x_center, y_center, bbox_width, bbox_height = map(float, data[1:])
                bboxes.append([x_center, y_center, bbox_width, bbox_height])
                class_labels.append(class_id)

    for i in range(3):  # Generate 3 augmented versions
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']

        # Save augmented image
        aug_img_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
        cv2.imwrite(os.path.join(output_img_dir, aug_img_name), aug_img)

        # Save new annotation file
        aug_ann_name = f"{os.path.splitext(img_name)[0]}_aug{i}.txt"
        with open(os.path.join(output_ann_dir, aug_ann_name), "w") as f:
            for bbox, label in zip(aug_bboxes, class_labels):
                f.write(f"{label} {' '.join(map(str, bbox))}\n")

print("Data augmentation complete!")
