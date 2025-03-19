# basic preprocessing for loading a text image.
import skimage.io as img_io
import skimage.color as img_color
from skimage.transform import resize
import cv2
import numpy as np

def fft_filter(image):
    factor = 2

    f_transform = np.fft.fft2(image)  # Compute the 2D Fourier Transform
    f_shift = np.fft.fftshift(f_transform)  # Shift the zero frequency to the center
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # Center coordinates
    
    axesLength = (cols//factor, rows//factor ) 
    mask = np.zeros((rows, cols))
    mask = cv2.ellipse(mask, (ccol, crow), axesLength, 0, 0, 360, 1, -1) 
    
    f_shift_filtered = f_shift * mask # Apply the mask to the shifted Fourier Transform
    f_ishift = np.fft.ifftshift(f_shift_filtered)  # Shift back the zero frequency
    img_filtered = np.fft.ifft2(f_ishift)  # Compute the inverse FFT
    img_filtered = np.abs(img_filtered)  # Take the magnitude
    #img_filtered = cv2.normalize(img_filtered, None, 0, 1, cv2.NORM_MINMAX)

    return img_filtered


def load_image(image_path):

    # read the image
    image = img_io.imread(image_path)

    # convert to grayscale skimage
    if len(image.shape) == 3:
        image = img_color.rgb2gray(image)
    
    # normalize the image
    image = 1 - image / 255.

    #image = fft_filter(image)  #comment this if you want to remove FFT low-pass filter

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



