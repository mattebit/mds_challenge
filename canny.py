import cv2
import numpy as np
import pywt
from scipy.linalg import svd
from scipy.signal import convolve2d
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

def similarity(X,X_star):
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def embed_watermark_svd(subband, watermark, alpha=5.0):
    U, S, V = svd(subband)
    Uw, Sw, Vw = svd(watermark)

    S_watermarked = S + alpha * Sw[:len(S)]
    
    watermarked_subband = np.dot(U, np.dot(np.diag(S_watermarked), V))
    return watermarked_subband

def select_best_regions(edge_map, block_size, threshold):
    h, w = edge_map.shape
    regions = []
    
    # Loop through the image in blocks of size block_size
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = edge_map[i:i+block_size, j:j+block_size]
            edge_density = np.mean(block)
            
            # Select the block if the edge density exceeds the threshold
            if edge_density > threshold:
                regions.append((i, j, block_size, block_size))
    
    return regions

def extraction(image_wm, original, U_wm, V_wm, alpha=5.0):
    edges = cv2.Canny(original, low_threshold, high_threshold)
    regions = select_best_regions(edges, block_size=64, threshold=66)

    watermarks = []
    for region in regions:
        x, y, block_h, block_w = region
        block_wm = image_wm[x:x+block_h, y:y+block_w]

        LL_w, (LH_w, HL_w, HH_w) = pywt.dwt2(block_wm, 'haar')
        U_w, S_w, V_w = svd(LL_w)

        LL, (LH, HL, HH) = pywt.dwt2(original[x:x+block_h, y:y+block_w], 'haar')
        U, S, V = svd(LL)

        S_extracted = np.zeros(32)
        for i in range(32):
            S_extracted[i] = (S_w[i] - S[i]) / alpha
        
        watermark = np.dot(U_wm, np.dot(np.diag(S_extracted), V_wm))
        watermarks.append(watermark)

    return watermarks


image = cv2.imread('lena_grey.bmp', 0)
watermark = np.load('findbrivateknowledge.npy')
watermark = cv2.resize(watermark, (32, 32))
low_threshold = 50
high_threshold = 150
if __name__ == "__main__":
    image = cv2.imread('lena_grey.bmp', 0)
    watermark = np.load('findbrivateknowledge.npy')
    watermark = cv2.resize(watermark, (32, 32))

    edges = cv2.Canny(image, low_threshold, high_threshold)

    regions = select_best_regions(edges, block_size=64, threshold=66)

    watermarked_image = image.copy()
    for region in regions:
        x, y, block_h, block_w = region
        block = image[x:x+block_h, y:y+block_w]

        LL, (LH, HL, HH) = pywt.dwt2(block, 'haar')
        LL_prime = embed_watermark_svd(LL, watermark)
        
        block_prime = pywt.idwt2((LL_prime, (LH, HL, HH)), 'haar')
        
        watermarked_image[x:x+block_h, y:y+block_w] = block_prime

    U_wm, S_wm, V_wm = svd(watermark)
    watermarks = extraction(watermarked_image, image, U_wm, V_wm)

    # Save or display the watermarked image
    # cv2.imwrite('watermarked_image.png', np.uint8(watermarked_image))

    #cv2.imshow('Edges', edges)
    # cv2.imshow('Watermarked Image', np.uint8(watermarked_image))
    cv2.imwrite('watermarked_image.bmp', watermarked_image)
    print(f'WPSNR: {wpsnr(image, watermarked_image)}')

    for i, w in enumerate(watermarks):
        print("Sim", i, ":", similarity(watermark, w))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
