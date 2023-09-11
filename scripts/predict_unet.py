# -*- coding: utf-8 -*-
"""
@author: SSALAZAR
"""

import segmentation_models_pytorch as smp

import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold
import albumentations as album

from tqdm import tqdm


# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values 
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

class RoadsDataset(torch.utils.data.Dataset):

    """Roads Dataset. Read images, apply augmentation and preprocessing transformations.
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline (flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing (normalization, shape manipulation, etc."""

    def __init__(
            self, 
            images_dir, 
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,):
      
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)#ORIGINAL
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image#, mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor))
        
    return album.Compose(_transform)

def predict(CLASS_PREDICT = ['background', 'road'],SIZE=100, CLASS_RGB=[[0, 0, 0], [255, 255, 255]], CROP_PATH='./crop_images/',ENCODER = 'vgg16', ENCODER_WEIGHTS = 'imagenet', ACTIVATION = 'sigmoid',WEIGHT='./weight/Road_weight_k0_e25_vgg16.pth', PRED_PATH = './prediction/predictions/', CLASS_INDEX='road', clean_crop=False):

    
    height_crop=width_crop=(SIZE//32)*32
    height_pad=width_pad=((SIZE//32)*32)+32
    
    def get_training_augmentation():
        train_transform = [
            album.RandomCrop(height=height_crop, width=width_crop, always_apply=True),
            album.OneOf(
                [
                    album.HorizontalFlip(p=1),
                    album.VerticalFlip(p=1),
                    album.RandomRotate90(p=1),
                ],
                p=0.75,
            ),
        ]
        return album.Compose(train_transform)


    def get_validation_augmentation():   
        # Add sufficient padding to ensure image is divisible by 32
        test_transform = [
            album.PadIfNeeded(min_height=height_pad, min_width=width_pad, always_apply=True, border_mode=0),
        ]
        return album.Compose(test_transform)



    class_names = CLASS_PREDICT
    class_rgb_values =CLASS_RGB

    # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = CLASS_PREDICT
    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    #DATA SOURCE
    DATA_DIR =CROP_PATH
    x_test_dir = os.path.join(DATA_DIR, "images")
    #y_test_dir = os.path.join(DATA_DIR, "masks_prueba")

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,     encoder_weights=ENCODER_WEIGHTS, 
        classes=len(select_classes),     activation=ACTIVATION,)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # create test dataloader to be used with UNet model (with preprocessing operation: to_tensor(...))
    test_dataset = RoadsDataset(
        x_test_dir,     
        #y_test_dir, 
        augmentation=get_validation_augmentation(),     
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,)

    # Set device: `CUDA` or `CPU`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load best saved model checkpoint from the current run
    if os.path.exists(WEIGHT):
        best_model = torch.load(WEIGHT, map_location=DEVICE)
        print('Loaded UNet model from this run.')

    # Center crop padded image / mask to original image dims
    def crop_image(image, target_image_dims=[100,100,3]):
        target_size = target_image_dims[0]
        image_size = len(image)
        padding = (image_size - target_size) // 2

        if padding<0:
            return image

        return image[
            padding:image_size - padding,
            padding:image_size - padding, :, ]

    sample_preds_folder = PRED_PATH

    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)

    #for idx,name in tqdm(zip(range(len(test_dataset)),[i.split('\\', 1)[1] for i in test_dataset.image_paths])):
    for idx,name in tqdm(zip(range(len(test_dataset)),[i.split('\\')[-1] for i in test_dataset.image_paths])):
        image = test_dataset[idx]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # Predict test image
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask,(1,2,0))
        # Get prediction channel corresponding to road
        pred_road_heatmap = pred_mask[:,:,select_classes.index(CLASS_INDEX)]
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))
        # Convert gt_mask from `CHW` format to `HWC` format
        cv2.imwrite(os.path.join(sample_preds_folder, name), np.hstack([pred_mask]))
        #Borrar cortes despues de predecirlos
        if clean_crop==True:
            os.remove(os.path.join(x_test_dir, name))
