from findbrivateknowledge_attack import apply_attacks, Attack
from findbrivateknowledge_embedding import embedding
from findbrivateknowledge_detection import detection
import cv2
import os
import numpy as np
from scipy.signal import convolve2d
from math import sqrt

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

if __name__ == "__main__":
    """ watermarked_image = embedding('lena_grey.bmp', 'findbrivateknowledge.npy')
    cv2.imwrite('findbrivateknowledge_embedded.bmp',
                np.uint8(watermarked_image))

    attacks = [
        [{"attack": Attack.BLUR, "params": {"sigma": [0.5, 0.5]}}],
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
        print(ATTACK_STRING, res_wpsnr, a) """
    
    lena = cv2.imread('lena_grey.bmp', 0)
    watermarked_image = embedding('lena_grey.bmp', 'findbrivateknowledge.npy')
    cv2.imwrite('findbrivateknowledge_embedded.bmp', np.uint8(watermarked_image))
    print(wpsnr(lena, watermarked_image))
    random_bits = np.random.uniform(0.0, 1.0, 1024)
    random_bits = np.uint8(np.rint(random_bits))
    np.save("random_bits.npy", random_bits)
    
    watermarked_fake = embedding('lena_grey.bmp', 'random_bits.npy')
    cv2.imwrite('fake_embedded.bmp', np.uint8(watermarked_image))

    print("1:", np.allclose(watermarked_image, watermarked_fake))

    det1, wpsnr1 = detection('lena_grey.bmp', 'findbrivateknowledge_embedded.bmp', 'fake_embedded.bmp')

    print("Det 1:", det1, " - wpsnr:", wpsnr1)
    
