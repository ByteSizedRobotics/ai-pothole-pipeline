Source: https://www.kaggle.com/datasets/andrewmvd/pothole-detection
Contains images of only potholes


dataset format: 
- image folder contains full sample image
- annotations folder contains .xml files for the images
- needed to convert PascalVOC annotations to XML using convert_voc_to_yolo.py script
  found https://gist.github.com/vdalv/321876a7076caaa771d47216f382cba5
  (make sure cwd is set to the right dir when using script)
  (also put pictures + annotations in same folder when u want to convert to yolo annotations)
- created new folder called labels using script to contain the yolo format annotations

train_val_splitter.py was created to split the dataset 