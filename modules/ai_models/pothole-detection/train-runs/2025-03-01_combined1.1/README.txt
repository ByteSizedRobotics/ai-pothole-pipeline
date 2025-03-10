Trained on 2025-03-01 with custom/1.1 Combined dataset

Classes: pothole
Training: using pretrained yolov5s weights BUT with early stopping
          attempting to use the set of hyperparams to optimize performance for small dataset
          using a 90/10 train/valid split
          dataset has custom taken images but each image is augmented and creates 3 versions of each custom image

Command: python train.py --img 640 --batch 16 --epochs 100 --data data/dataset-potholes-custom.yaml --weights yolov5s.pt --patience 25 --hyp hyp.scratch-low.yaml

Results: 100 epochs completed in 17.697 hours. 

Model summary: 157 layers, 7012822 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 6/6 [00:16<00:00,  2.74s/it]
                   all        181        387      0.867      0.758      0.843      0.514