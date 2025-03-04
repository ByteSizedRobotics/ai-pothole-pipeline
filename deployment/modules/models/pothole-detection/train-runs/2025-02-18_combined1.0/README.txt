Trained on 2025-02-18 with custom/1.0 Combined dataset

Classes: pothole
Training: using pretrained yolov5s weights BUT with early stopping
          attempting to use the set of hyperparams to optimize performance for small dataset
          using a 90/10 train/valid split

Command: python train.py --img 640 --batch 16 --epochs 100 --data data/dataset-potholes-custom.yaml --weights yolov5s.pt --patience 10 --hyp hyp.scratch-low.yaml

Results: best results observed at EPOCH 32, early stopping took place after 43 epochs

Model summary: 157 layers, 7012822 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95:
                   all        101        185      0.876      0.578      0.704      0.428