import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import skimage.io as img_io

def load_image(image_path):
    # read the image
    image = img_io.imread(image_path)
    # convert to grayscale skimage
    if len(image.shape) == 3:
        image = img_color.rgb2gray(image)
    # normalize the image
    image = 1 - image / 255.
    return image


# Load the image in grayscale
#image = cv2.imread('../DATASETS/IAM/processed_words/test/d01-052-02-01.png', cv2.IMREAD_GRAYSCALE)
image = load_image('../DATASETS/IAM/processed_words/test/n04-009-03-09.png')

# Step 1: Apply Fourier Transform
f_transform = np.fft.fft2(image)  # Compute the 2D Fourier Transform
f_shift = np.fft.fftshift(f_transform)  # Shift the zero frequency to the center

# Step 2: Create a low-pass filter mask
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2  # Center coordinates
radius = rows//2  # Radius for the low-pass filter

# Create a circular mask
# mask = np.zeros((rows, cols), np.uint8)
# mask = cv2.circle(mask, (ccol, crow), radius, 1, -1)

factor = 2
axesLength = (cols//factor, rows//factor ) 
mask = np.zeros((rows, cols))
mask = cv2.ellipse(mask, (ccol, crow), axesLength, 0, 0, 360, 1, -1) 

# Apply the mask to the shifted Fourier Transform
f_shift_filtered = f_shift * mask




# Step 3: Inverse Fourier Transform
f_ishift = np.fft.ifftshift(f_shift_filtered)  # Shift back the zero frequency

#Display FFT
f_shift_disp = np.log(np.abs(f_shift))
f_shift_disp = cv2.normalize(f_shift_disp, None, 0, 1, cv2.NORM_MINMAX)

#Display FFT filtered
f_shift_disp2 = f_shift_disp * mask

img_filtered = np.fft.ifft2(f_ishift)  # Compute the inverse FFT
img_filtered = np.abs(img_filtered)  # Take the magnitude
#img_filtered = cv2.normalize(img_filtered, None, np.min(image), np.max(image), cv2.NORM_MINMAX)


#f_shift_disp2 = np.log(np.abs(img_filtered))
#f_shift_disp2 = cv2.normalize(f_shift_disp2, None, 0, 1, cv2.NORM_MINMAX)

max_min = img_filtered
print(f"{np.min(max_min)} - {np.max(max_min)}")

img_io.imsave("image1.png", img_as_ubyte(image))
img_io.imsave("image2.png", img_as_ubyte(mask))
img_io.imsave("image3.png", img_as_ubyte(f_shift_disp))
img_io.imsave("image4.png", img_as_ubyte(f_shift_disp2))
img_io.imsave("image5.png", img_as_ubyte(img_filtered))



#cv2.imwrite('image1.png', image)  # Save the original image
# cv2.imwrite('image2.png', mask.astype(np.uint8))
# cv2.imwrite('image3.png', img_filtered.astype(np.uint8))
