# models/road_segmenter.py
import os
import torch
import torch.nn as nn
from torchvision import transforms as T
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Adds 'modules/' to sys.path

import models.DeepLabV3Plus.network as network
import models.DeepLabV3Plus.utils as utils
from models.DeepLabV3Plus.datasets import VOCSegmentation, Cityscapes

class RoadSegmentation:
    def __init__(self, config):
        self.config = config
        
        # Set up the CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_ID
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up the model
        if config.DATASET.lower() == 'voc':
            self.num_classes = 21
            self.decode_fn = VOCSegmentation.decode_target
        elif config.DATASET.lower() == 'cityscapes':
            self.num_classes = 19
            self.decode_fn = Cityscapes.decode_target
        
        # Initialize the model
        self._init_model()
        
        # Initialize transformations
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    
    def _init_model(self):
        # Create the DeepLabV3+ model
        self.model = network.modeling.__dict__[self.config.DEEPLAB_MODEL](
            num_classes=self.num_classes, 
            output_stride=self.config.OUTPUT_STRIDE
        )
        utils.set_bn_momentum(self.model.backbone, momentum=0.01)
        
        # Load checkpoint if available
        if os.path.isfile(self.config.DEEPLAB_CHECKPOINT):
            checkpoint = torch.load(
                self.config.DEEPLAB_CHECKPOINT, 
                map_location=torch.device('cpu'),
                weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state"])
            print(f"Loaded segmentation model from {self.config.DEEPLAB_CHECKPOINT}")
            del checkpoint
        else:
            print("[!] Warning: No checkpoint found for segmentation model")
        
        # Wrap model with DataParallel and move to device
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()
    
    def segment_image(self, image):
        """
        Segment an image to get a road mask.
        
        Args:
            image: PIL Image
            
        Returns:
            road_mask: Binary numpy array with 1 for road pixels and 0 elsewhere
            full_segmentation: Complete segmentation output with all classes
        """
        with torch.no_grad():
            # Transform image to tensor
            img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            img_tensor = img_tensor.to(self.device)
            
            # Forward pass
            outputs = self.model(img_tensor)
            predictions = outputs.max(1)[1].cpu().numpy()[0]  # (H, W)
            
            # Create road mask (class index 1 is road in Cityscapes)
            road_mask = (predictions == 1).astype(np.uint8)
            
            return road_mask, predictions
    
    def get_colorized_segmentation(self, segmentation):
        """Convert segmentation indices to RGB visualization"""
        return self.decode_fn(segmentation).astype('uint8')