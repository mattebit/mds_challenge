import numpy as np
import os
import cv2
import pywt
import scipy.fftpack as fftpack
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
from math import sqrt

def wpsnr(img1, img2):
  img1 = np.float32(img1)/255.0
  img2 = np.float32(img2)/255.0
  difference = img1-img2
  same = not np.any(difference)
  if same is True:
      return 9999999
  w = np.genfromtxt('csf.csv', delimiter=',')
  ew = convolve2d(difference, np.rot90(w,2), mode='valid')
  decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
  return decibels

def embed_redundant_watermark(image, watermark, alpha=0.1):
    # Step 1: Apply DWT to the original image
    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
    
    # Step 2: Apply DCT to the high-frequency subbands (LH, HL, HH)
    dct_HL = fftpack.dct(fftpack.dct(HL.T, norm='ortho').T, norm='ortho')
    dct_HH = fftpack.dct(fftpack.dct(HH.T, norm='ortho').T, norm='ortho')
    
    wm_resized = np.reshape(watermark, (dct_HL.shape[1], dct_HL.shape[0]))
    
    # Step 4: Embed different parts of the watermark into HL and HH DCT coefficients
    dct_HL_watermarked = dct_HL + alpha * wm_resized
    dct_HH_watermarked = dct_HH + alpha * wm_resized
    
    # Step 5: Apply inverse DCT to get modified subbands
    HL_modified = fftpack.idct(fftpack.idct(dct_HL_watermarked.T, norm='ortho').T, norm='ortho')
    HH_modified = fftpack.idct(fftpack.idct(dct_HH_watermarked.T, norm='ortho').T, norm='ortho')
    
    # Step 6: Reconstruct the watermarked image using inverse DWT
    #watermarked_image = idwt2_image(LL, (LH_modified, HL_modified, HH_modified))
    watermarked_image = np.uint8(pywt.idwt2((LL, (LH, HL_modified, HH_modified)), 'haar'))
    
    return watermarked_image

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

# Load the original image and watermark
original_image = cv2.imread('lena_grey.bmp', 0)
watermark = np.load('findbrivateknowledge.npy')

# Embed the watermark using redundant embedding in DWT and DCT
watermarked_image = embed_redundant_watermark(original_image, watermark)

# Save and display the watermarked image
cv2.imwrite('watermarked_image_redundant.png', watermarked_image)
cv2.imshow('Watermarked Image', watermarked_image)
print(f'WPSNR: {wpsnr(original_image, watermarked_image)}')
cv2.waitKey(0)
cv2.destroyAllWindows()