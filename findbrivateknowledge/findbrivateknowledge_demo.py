from findbrivateknowledge_attack import apply_attacks, Attack
from findbrivateknowledge_embedding import embedding
from findbrivateknowledge_detection import detection
import cv2
import os
import numpy as np

if __name__ == "__main__":
    watermarked_image = embedding('lena_grey.bmp', 'findbrivateknowledge.npy')
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
        print(ATTACK_STRING, res_wpsnr, a)
