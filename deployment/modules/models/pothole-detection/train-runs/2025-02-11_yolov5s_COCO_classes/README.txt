Trained on 2025-02-11 with 2-potholes-only dataset

Classes: pothole, COCO (80 classes)
Training: using pretrained yolov5s weights

Training Command: python train.py --img 640 --batch 16 --epochs 50 --data dataset-coco-potholes2.yaml --weights yolov5s.pt --patience 10

Other Notes:
- Early stopping introduced
- 1st train-run keeping the COCO pretrained classes