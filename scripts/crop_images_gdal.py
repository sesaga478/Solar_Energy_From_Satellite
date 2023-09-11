# -*- coding: utf-8 -*-
"""
@author: SSALAZAR
"""

from osgeo import gdal
import numpy as np
import os,glob
from tqdm import tqdm


def crop(ORI_PATH="./data/big_img/",SIZE=100,CROP_PATH='crop_images/images', EXT='tif'):

    original_path = ORI_PATH
    fol_crop = os.path.join(CROP_PATH)

    if not os.path.exists(CROP_PATH): # here '100' corresponds to size by default
        os.makedirs(CROP_PATH)

    list_imgs_original = glob.glob(original_path+'/*.tif')

    for im in tqdm(list_imgs_original):
        
        ds = gdal.Open(im)

        num_bands = ds.RasterCount
        data = []
        for i in range(num_bands):
            band = ds.GetRasterBand(i+1)
            band_data = band.ReadAsArray()
            data.append(band_data)
        data = np.array(data)
        img_np = np.transpose(data, (1, 2, 0))
        img_width, img_height, dimension = img_np.shape
        height_sizes = [SIZE]
        width_sizes =  [SIZE]
        print(img_np.shape)
        for height in height_sizes:
            width=height
            k = 0
            for i in range(0, img_height, height):
                for j in range(0, img_width, width):
                    try:
                        imagen_test_name = os.path.join(fol_crop,im.split('/')[-1].replace("."+EXT, '') + '_{}_{}_{}_{}_{}_1_AnalyticMS_rgb_scaled.{}'.format(i, j, k, height, width,EXT))
                        print(imagen_test_name)
                        if not os.path.exists(imagen_test_name):
                            gdal.Translate(imagen_test_name, im,options='-srcwin {} {} {} {}'.format(j, i, width, height))
                    except:
                        pass
                    k+=1

