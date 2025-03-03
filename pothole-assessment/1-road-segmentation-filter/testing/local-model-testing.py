import os
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DeepLabV3Plus-PyTorch')))

import utils
import network
import numpy as np
import torch
import torch.nn as nn
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from torch.utils.data import dataset
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.patches as mpatches

# Code adapted from DeepLabV#Plus-Pytorch/predict.py

# Define Cityscapes class colors and labels
# Labels based on https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py#L55
cityscapes_classes = {
    0: ('unlabeled', (0, 0, 0)),
    1: ('road', (128, 64, 128)),
    2: ('sidewalk', (244, 35, 232)),
    3: ('building', (70, 70, 70)),
    4: ('wall', (102, 102, 156)),
    5: ('fence', (190, 153, 153)),
    6: ('pole', (153, 153, 153)),
    7: ('traffic light', (250, 170, 30)),
    8: ('traffic sign', (220, 220, 0)),
    9: ('vegetation', (107, 142, 35)),
    10: ('terrain', (152, 251, 152)),
    11: ('sky', (70, 130, 180)),
    12: ('person', (220, 20, 60)),
    13: ('rider', (255, 0, 0)),
    14: ('car', (0, 0, 142)),
    15: ('truck', (0, 0, 70)),
    16: ('bus', (0, 60, 100)),
    17: ('train', (0, 80, 100)),
    18: ('motorcycle', (0, 0, 230)),
    19: ('bicycle', (119, 11, 32))
}

def visualize_segmentation(original_img, segmented_img, class_colors, class_names):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(original_img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(segmented_img)
    ax[1].set_title('Segmented Image')
    ax[1].axis('off')
    
    # Create a legend
    patches = [mpatches.Patch(color=np.array(color)/255.0, label=class_name) for class_name, color in zip(class_names, class_colors)]
    ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.show()

def main():
    input_path = "test_images" # ADD YOUR PATH TO IMAGE OR DIRECTORY
    dataset_name = 'cityscapes'
    model_name = 'deeplabv3plus_resnet101'
    separable_conv = False
    output_stride = 16
    save_val_results_to = 'test_results'
    crop_val = False
    val_batch_size = 4
    crop_size = 513
    ckpt = 'checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth'
    gpu_id = '0'

    if dataset_name.lower() == 'voc':
        num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif dataset_name.lower() == 'cityscapes':
        num_classes = 19
        decode_fn = Cityscapes.decode_target
        class_colors = [color for _, color in cityscapes_classes.values()]
        class_names = [name for name, _ in cityscapes_classes.values()]


    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(input_path):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(input_path, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(input_path):
        image_files.append(input_path)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)
    if separable_conv and 'plus' in model_name:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if ckpt is not None and os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    if crop_val:
        transform = T.Compose([
                T.Resize(crop_size),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if save_val_results_to is not None:
        os.makedirs(save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if save_val_results_to:
                colorized_preds.save(os.path.join(save_val_results_to, img_name+'.png'))
            
            # Visualize the segmentation
            visualize_segmentation(Image.open(img_path).convert('RGB'), colorized_preds, class_colors, class_names)

if __name__ == '__main__':
    main()