import shutil
from sklearn.model_selection import train_test_split
from os import getcwd, listdir, makedirs, path

# path to image and label folders
## CHANGE THIS TO OWN PATH
cwd = getcwd()
image_dir = path.join(cwd, "images")
label_dir = path.join(cwd, "labels")

images = sorted(listdir(image_dir))
labels = sorted(listdir(label_dir))

# split
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# create dirs
makedirs(path.join(cwd, "train/images"), exist_ok=True)
makedirs(path.join(cwd, "valid/images"), exist_ok=True)
makedirs(path.join(cwd, "train/labels"), exist_ok=True)
makedirs(path.join(cwd, "valid/labels"), exist_ok=True)

# Move the files
for img, lbl in zip(train_images, train_labels):
    shutil.copy(path.join(image_dir, img), path.join(cwd, "train/images"))
    shutil.copy(path.join(label_dir, lbl), path.join(cwd,"train/labels"))
for img, lbl in zip(val_images, val_labels):
    shutil.copy(path.join(image_dir, img), path.join(cwd,"valid/images"))
    shutil.copy(path.join(label_dir, lbl), path.join(cwd,"valid/labels"))
