import os
from math import sqrt

import cv2
import numpy as np
from scipy.signal import convolve2d

from findbrivateknowledge_attack import Attack, apply_attacks
from findbrivateknowledge_detection import detection
from findbrivateknowledge_embedding import embedding


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


if __name__ == "__main__":
    watermarked_image = embedding('lena_grey.bmp', 'findbrivateknowledge.npy')
    cv2.imwrite('findbrivateknowledge_embedded.bmp',
                np.uint8(watermarked_image))

    attacks = [
        [{"attack": Attack.MEDIAN, "params": {"kernel_size": 5}}],
        [{'attack': Attack.AWGN, 'params': {'std': 13, 'seed': 30, 'mean': 0.0}}]
    ]

    for i, attack in enumerate(attacks):
        if not os.path.isdir("results"):
            os.mkdir("results")
        res_attacked, res_contains_w, res_wpsnr, a = apply_attacks(
            "findbrivateknowledge_embedded.bmp", attack, detection)
        cv2.imwrite(
            f'results/findbrivateknowledge_attacked{i}.bmp', np.uint8(res_attacked))
        ATTACK_STRING = "Watermark present:" if res_contains_w else "Watermark removed:"
        print(ATTACK_STRING, res_wpsnr, a)
