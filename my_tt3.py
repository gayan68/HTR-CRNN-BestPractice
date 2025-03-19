import argparse
from omegaconf import OmegaConf

import sys
import os

import skimage.io as img_io
import skimage.color as img_color
from skimage.transform import resize
import numpy as np

from skimage import img_as_ubyte


def load_image(image_path):

    # read the image
    image = img_io.imread(image_path)

    # convert to grayscale skimage
    if len(image.shape) == 3:
        image = img_color.rgb2gray(image)
    
    # normalize the image
    image = 1 - image / 255.

    return image


def preprocess(config, img, input_size, border_size=8):
    
    h_target, w_target = input_size

    n_height = min(h_target - 2 * border_size, img.shape[0])
    
    #print(f"img.shape[0]: {img.shape[0]}")
    scale = n_height / img.shape[0]
    #print(f"scale: {scale}")
    n_width = min(w_target - 2 * border_size, int(scale * img.shape[1]))

    img = resize(image=img, output_shape=(n_height, n_width)).astype(np.float32)

    # right pad image to input_size
    if config.preprocess.mode == "constant":
        if config.preprocess.values == "median":
            img  = np.pad(img, ((border_size, h_target - n_height - border_size), (border_size, w_target - n_width - border_size),),
                                                    mode='constant', constant_values=np.median(img))
        else:
            img  = np.pad(img, ((border_size, h_target - n_height - border_size), (border_size, w_target - n_width - border_size),),
                                                    mode='constant', constant_values=0)
    else:
        img  = np.pad(img, ((border_size, h_target - n_height - border_size), (border_size, w_target - n_width - border_size),),
                                                    mode='median')

    return img

def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf


#######################################################################################################################

subset='test'

config = parse_args()
img = load_image("data/IAM/processed_words/train/a06-025-03-01.png")

fixed_size = (config.preproc.image_height, config.preproc.image_width)
fheight, fwidth = fixed_size[0], fixed_size[1]

if subset == 'train':
    nwidth = int(np.random.uniform(.75, 1.25) * img.shape[1])
    nheight = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nwidth)
    nwidth = img.shape[1] if nwidth == 0 else nwidth
    nheight = img.shape[0] if nheight == 0 else nheight
    img = resize(image=img, output_shape=(nheight, nwidth)).astype(np.float32)

print(img.shape)

img = preprocess(config, img, (fheight, fwidth))

print(img.shape)


output_path = "image.png"  # Specify the output file path

img_to_save = img_as_ubyte(img)  # Convert the float32 image (assumed normalized) to uint8
img_io.imsave(output_path, img_to_save)

print(f"Image saved to {output_path}")




