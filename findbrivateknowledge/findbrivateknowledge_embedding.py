from math import sqrt

import cv2
import numpy as np
import pywt
from numpy.linalg import svd
from scipy.signal import convolve2d

from mds_challenge.findbrivateknowledge.findbrivateknowledge_detection import watermark_to_bytes

BLOCK_SIZE = 64
LOW_THRESHOLD = 100
HIGH_THRESHOLD = 150
ALPHA = .86


def wpsnr(img1, img2):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0
    difference = img1 - img2
    same = not np.any(difference)
    if same is True:
        return 9999999
    w = np.genfromtxt('csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(w, 2), mode='valid')
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels


def similarity(X, X_star):
    norm_X = np.sqrt(np.sum(np.multiply(X, X)))
    norm_X_star = np.sqrt(np.sum(np.multiply(X_star, X_star)))

    if norm_X == 0 or norm_X_star == 0:
        return 0.0

    s = np.sum(np.multiply(X, X_star)) / (norm_X * norm_X_star)
    return s


def embed_watermark_svd(subband, watermark):
    U, S, V = svd(subband)
    w_b = watermark_to_bytes(watermark)

    S_watermarked = S.copy()

    for i in range(128):
        S_watermarked[i] = S[i] + (ALPHA * w_b[i])
    #for i in range(128,256):
    #    S_watermarked[i] = S[i] + (ALPHA * w_b[i-128])

    #print(w_b)

    watermarked_subband = np.dot(U, np.dot(np.diag(S_watermarked), V))
    return watermarked_subband


def select_best_regions(edge_map):
    h, w = edge_map.shape
    regions = []

    for i in range(1, h - BLOCK_SIZE + 1, 8):
        for j in range(1, w - BLOCK_SIZE + 1, 8):
            block = edge_map[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE]
            edge_density = np.mean(block)

            regions.append((i, j, edge_density))

    regions = sorted(regions, key=lambda x: x[2], reverse=True)

    selected_regions = []
    selected_regions.append(regions[0])
    selected_regions.append(regions[len(regions) - 1])
    selected_regions.append(regions[int(len(regions) // 2)])

    return [(i, j) for i, j, _ in selected_regions]


def embedding(input1, input2):
    """
    input1: original image file name
    input2: watermark file name
    """
    image = cv2.imread(input1, 0)
    watermark = np.load(input2)

    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
    LL_prime = embed_watermark_svd(LL, watermark)

    watermarked_image = pywt.idwt2((LL_prime, (LH, HL, HH)), 'haar')

    return watermarked_image


if __name__ == "__main__":
    watermarked_image = embedding('lena_grey.bmp', 'findbrivateknowledge.npy')
    cv2.imwrite('findbrivateknowledge_embedded.bmp', np.uint8(watermarked_image))
