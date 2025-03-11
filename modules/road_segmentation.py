import os
import torch
import torch.nn as nn
from torchvision import transforms as T
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Adds 'modules/' to sys.path

import ai_models.DeepLabV3Plus.network as network
import ai_models.DeepLabV3Plus.utils as utils
from ai_models.DeepLabV3Plus.datasets import VOCSegmentation, Cityscapes

# THIS SCRIPT IS BASED OFF OF DEEPLABV3+ predict.py script in the DeepLabV3Plus repository
# https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/predict.py
class RoadSegmentation:
    def __init__(self, config):
        self.config = config
        
        os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_ID
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up the model
        if config.DATASET.lower() == 'voc':
            self.num_classes = 21
            self.decode_fn = VOCSegmentation.decode_target
        elif config.DATASET.lower() == 'cityscapes':
            self.num_classes = 19
            self.decode_fn = Cityscapes.decode_target
        
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
        
        # load checkpoint => using CITYSCAPES WEIGHTS for road segmentation
        if os.path.isfile(self.config.DEEPLAB_CHECKPOINT_FILE):
            checkpoint = torch.load(
                self.config.DEEPLAB_CHECKPOINT_FILE, 
                map_location=torch.device('cpu'),
                weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state"])
            print(f"Loaded segmentation model from {self.config.DEEPLAB_CHECKPOINT_FILE}")
            del checkpoint
        else:
            print("[!] Warning: No checkpoint found for segmentation model")
        
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()
    
    def segment_image(self, image):
        """
        Segment an image with Cityscapes classes
        """
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0)
            
            outputs = self.model(img_tensor)
            predictions = outputs.max(1)[1].cpu().numpy()[0]
            
            # Create road mask (class index 1 is road in Cityscapes but 0 is the one for road??)
            road_mask = (predictions == 0).astype(np.uint8)
            
            return road_mask, predictions
    
    def get_colorized_segmentation(self, segmentation):
        """Convert segmentation indices to RGB visualization"""
        return self.decode_fn(segmentation).astype('uint8')