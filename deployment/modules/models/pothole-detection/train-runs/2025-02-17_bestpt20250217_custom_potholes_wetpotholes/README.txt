Trained on 2025-02-17 with custom 1.0 Wetpotholes dataset

Classes: pothole
Training: using best.pt 2025-02-17_bestpt20250217_custom_potholes (this best.pt is trained on custom 1.0 Potholes and Wetpotholes)

Command:
python train.py --img 640 --batch 16 --epochs 50 --data data/dataset-potholes-custom.yaml --weights ..\pothole-detection\train-runs\2025-02-17_bestpt20250217_custom_potholes\weights\best.pt --patience 10