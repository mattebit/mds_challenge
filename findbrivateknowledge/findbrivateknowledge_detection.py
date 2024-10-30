from math import sqrt

import cv2
import numpy as np
import pywt
from numpy.linalg import svd
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

BLOCK_SIZE = 64
THRESHOLD_W = 70
LOW_THRESHOLD = 100
HIGH_THRESHOLD = 150
ALPHA = .4
THRESHOLD = 0.83


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


def watermark_to_bytes(watermark: np.ndarray) -> np.ndarray:
    was_matrix = False
    if len(watermark) < 1023:
        was_matrix = True
        size = watermark.shape
        watermark = watermark.flatten()

    c = 1
    act_b = ""
    w_b = []
    for b in watermark:
        act_b += str(b)
        if c == 8:
            w_b.append(int(act_b, 2))
            c = 0
            act_b = ""
        c += 1

    w_b = np.array(w_b)

    if was_matrix:
        w_b = np.resize(np.array(w_b), (12, 12))

    return w_b


def extraction(image_wm, original):
    LL_w, (LH_w, HL_w, HH_w) = pywt.dwt2(image_wm, 'haar')
    U_w, S_w, V_w = svd(LL_w)

    LL, (LH, HL, HH) = pywt.dwt2(original, 'haar')
    U, S, V = svd(LL)

    w_ex = np.zeros(128)

    key = [(0, i) for i in range(128)]
    # for i in range(128):
    #    w_ex[i] = (LL_w[i][i] - LL[i][i]) / ALPHA
    c = 0
    for k in key:
        x, y = k
        w_ex[c] = (LL_w[x][y] - LL[x][y]) / ALPHA
        # LL_prime[x][y] += watermark[c] * ALPHA
        c += 1

    w_ex_2 = np.zeros(128)
    key = [(1, i) for i in range(128)]
    # for i in range(128):
    #    w_ex[i] = (LL_w[i][i] - LL[i][i]) / ALPHA
    c = 0
    for k in key:
        x, y = k
        w_ex_2[c] = (LL_w[x][y] - LL[x][y]) / ALPHA
        # LL_prime[x][y] += watermark[c] * ALPHA
        c += 1

    w_ex_3 = np.zeros(128)
    key = [(i+2, 0) for i in range(128)]
    # for i in range(128):
    #    w_ex[i] = (LL_w[i][i] - LL[i][i]) / ALPHA
    c = 0
    for k in key:
        x, y = k
        w_ex_3[c] = (LL_w[x][y] - LL[x][y]) / ALPHA
        # LL_prime[x][y] += watermark[c] * ALPHA
        c += 1

    w_ex_4 = np.zeros(128)
    key = [(i+2, 1) for i in range(128)]
    # for i in range(128):
    #    w_ex[i] = (LL_w[i][i] - LL[i][i]) / ALPHA
    c = 0
    for k in key:
        x, y = k
        w_ex_4[c] = (LL_w[x][y] - LL[x][y]) / ALPHA
        # LL_prime[x][y] += watermark[c] * ALPHA
        c += 1

    ex_avg = np.mean(np.array([w_ex, w_ex_2, w_ex_3, w_ex_4]), axis=0)
    return ex_avg


def detection(input1, input2, input3):
    """
    input1: original image
    input2: watermarked image
    input3: attacked image
    """
    original_img = cv2.imread(input1, 0)
    watermarked_img = cv2.imread(input2, 0)
    attacked_img = cv2.imread(input3, 0)

    original_watermark_ex = extraction(watermarked_img, original_img)
    # original_watermark = np.load("findbrivateknowledge.npy")
    # original_watermark = watermark_to_bytes(original_watermark)
    original_watermark = original_watermark_ex

    attacked_watermarks = extraction(attacked_img, original_img)

    # print(f"similarity: {similarity(original_watermark, attacked_watermarks)}")
    # print(f"similarity_ex: {similarity(original_watermark, original_watermark_ex)}")

    sim = similarity(original_watermark, attacked_watermarks)
    wpsnr_res = wpsnr(watermarked_img, attacked_img)
    detected = 1 if sim > THRESHOLD else 0
    return detected, wpsnr_res


if __name__ == "__main__":
    original_image = 'lena_grey.bmp'
    watermarked_image = 'findbrivateknowledge_embedded.bmp'
    attacked_image = 'results/findbrivateknowledge_attacked0.bmp'

    a = cv2.imread(watermarked_image, 0)
    b = cv2.imread(original_image, 0)

    print("wpsnr original, watermarked: ", wpsnr(a, b))

    attacked_image_ = gaussian_filter(b, [.2, .2])
    cv2.imwrite(attacked_image, attacked_image_)

    detected, wpsnr_value = detection(
        original_image, watermarked_image, attacked_image)
    print("Detected:", detected, " with WPSNR:", wpsnr_value)
