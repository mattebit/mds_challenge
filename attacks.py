import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from math import sqrt
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt, convolve2d
from skimage.transform import rescale

from canny import detection

w = np.genfromtxt('csf.csv', delimiter=',')

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = ".cache"
CACHE_PATH = os.path.join(SCRIPT_PATH, CACHE_DIR)
ORIGINAL_IMG_PATH = "lena_grey.bmp"


def show(*args: np.ndarray):
    """
    Plots the given images
    """
    col = len(args)
    # Show images side by side
    disp = "1" + str(col)

    for i in range(col):
        plt.subplot(int(disp + str(i + 1)))
        plt.title(str(i + 1))
        plt.imshow(args[i], cmap='gray')

    plt.show()


class Attack(Enum):
    BLUR = 0
    AWGN = 1
    SHARPEN = 2
    MEDIAN = 3
    RESIZING = 4
    JPEG = 5


def wpsnr(img1, img2):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0
    difference = img1 - img2
    same = not np.any(difference)
    if same is True:
        return 9999999
    ew = convolve2d(difference, np.rot90(w, 2), mode='valid')
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels


def attack_blur(img: np.ndarray, sigma) -> np.ndarray:
    """
    attack_blur(im, [2,2])
    """
    attacked = gaussian_filter(img, sigma)
    return attacked


def attack_AWGN(img: np.ndarray, std: int, seed: float, mean: float = 0.0) -> np.ndarray:
    mean = mean  # some constant
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked


def attack_sharpen(img: np.ndarray, sigma: int, alpha: int) -> np.ndarray:
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked


def attack_median(img: np.ndarray, kernel_size: int) -> np.ndarray:
    attacked = medfilt(img, kernel_size)
    return attacked


def attack_resizing(img: np.ndarray, scale: float) -> np.ndarray:
    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1 / scale)
    attacked = attacked[:x, :y]
    if attacked.shape[0] != x or attacked.shape[1] != y:
        attacked = cv2.resize(attacked, (x, y))
    return attacked


def attack_jpeg(img: np.ndarray, QF: int):
    img = Image.fromarray(img)
    img.save('tmp.jpg', "JPEG", quality=QF)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked, dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked


a_map = {
    Attack.BLUR: attack_blur,
    Attack.MEDIAN: attack_median,
    Attack.JPEG: attack_jpeg,
    Attack.AWGN: attack_AWGN,
    Attack.SHARPEN: attack_sharpen,
    Attack.RESIZING: attack_resizing
}

demo_attacks = [
    {
        "attack": Attack.BLUR,
        "params": {
            "sigma": [1, 1]
        }
    },
    {
        "attack": Attack.JPEG,
        "params": {
            "QF": 20
        }
    }
]


def apply_attack_queue(im: str,
                       l: list,
                       detection_function=None) -> Tuple[np.ndarray, int, float]:
    """
    Apply the given list of attacks to the given image and return the attacked image.
    The list of attacks should be of this form:
    demo_attacks = [
    {
        "attack": Attack.BLUR,  # Specify which attack to apply
        "params": {
            "sigma": [1,1]  # Specify the parameters
        }
    }
    ]
    """
    attacking: np.ndarray = cv2.imread(im, 0)
    contains_w, wpsnr = None, None

    for a in l:
        a_type: Attack = a["attack"]
        a_params: dict[any] = a["params"]
        if a_type == Attack.SHARPEN:
            attacking = attack_sharpen(
                attacking, a_params["sigma"], a_params["alpha"])
        elif a_type == Attack.BLUR:
            attacking = attack_blur(attacking, a_params["sigma"])
        elif a_type == Attack.AWGN:
            attacking = attack_AWGN(
                attacking, a_params["std"], a_params["seed"], a_params["mean"])
        elif a_type == Attack.JPEG:
            attacking = attack_jpeg(attacking, a_params["QF"])
        elif a_type == Attack.MEDIAN:
            attacking = attack_median(attacking, a_params["kernel_size"])
        elif a_type == Attack.RESIZING:
            attacking = attack_resizing(attacking, a_params["scale"])
        else:
            raise KeyError(f"invalid attack in dict: {a_type}")
    print("Attack applied: ", l)
    if detection_function is not None:
        detection_attacked_path = os.path.join(CACHE_PATH, f"attacked_{uuid.uuid4()}.bmp")
        cv2.imwrite(detection_attacked_path, attacking)
        contains_w, wpsnr = detection_function(ORIGINAL_IMG_PATH, im, detection_attacked_path)
    return attacking, contains_w, wpsnr


if __name__ == "__main__":
    attacks = [[
        {
            "attack": Attack.BLUR,
            "params": {
                "sigma": [1, 1]
            }
        },
    ], [
        {
            "attack": Attack.JPEG,
            "params": {
                "QF": 20
            }
        }
    ], [
        {
            "attack": Attack.AWGN,
            "params": {
                "std": 10,
                "seed": 0,
                "mean": 0
            }
        }
    ], [
        {
            "attack": Attack.MEDIAN,
            "params": {
                "kernel_size": 3
            }
        }
    ], [
        {
            "attack": Attack.RESIZING,
            "params": {
                "scale": 0.2
            }
        }
    ],
        [
            {
                "attack": Attack.SHARPEN,
                "params": {
                    "sigma": 1,
                    "alpha": 1
                }
            }

        ]]

    watermarked_img_path = 'watermarked_image.bmp'
    watermarked_img = cv2.imread(watermarked_img_path, 0)
    original_img = cv2.imread(ORIGINAL_IMG_PATH, 0)
    watermark = np.load('findbrivateknowledge.npy')
    watermark = cv2.resize(watermark, (32, 32))

    # Execute attacks
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(apply_attack_queue, watermarked_img_path, attack, detection)
            for i, attack in enumerate(attacks)
        ]

    results = []
    for future in futures:
        results.append(future.result())

    U_wm, S_wm, V_wm = svd(watermark)
    # np.savetxt("U_wm", U_wm)
    # np.savetxt("S_wm", S_wm)
    # np.savetxt("V_wm", V_wm)

    for i, result in enumerate(results):
        res_attacked, res_contains_w, res_wpsnr = result
        print(f"Contains w?: {res_contains_w}, WPSNR: {res_wpsnr}")
        # watermarks = extraction(res_attacked, original_img, U_wm, V_wm)
        # watermarks_values = [similarity(watermark, wm) for wm in watermarks]
        # print(watermarks_values, wpsnr(watermarked_img, res_attacked), attacks[i][0]["attack"])
