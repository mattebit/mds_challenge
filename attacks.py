import datetime
import os
from enum import Enum
from math import sqrt
from queue import Queue

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt, convolve2d
from skimage.transform import rescale


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


w = np.genfromtxt('csf.csv', delimiter=',')


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
    from scipy.ndimage.filters import gaussian_filter
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


def apply_attack_queue(im: np.ndarray, l: list, res_queue: Queue, id=None) -> np.ndarray:
    """
    Apply the given list of attacks to the given image and return the attacked image.
    demo_attacks = [
    {
        "attack": Attack.BLUR,
        "params": {
            "sigma": [1,1]
        }
    }
    ]
    """
    attacked: np.ndarray = im

    for a in l:
        a_type: Attack = a["attack"]
        a_params: dict[any] = a["params"]
        if a_type == Attack.SHARPEN:
            attacked = attack_sharpen(attacked, a_params["sigma"], a_params["alpha"])
        elif a_type == Attack.BLUR:
            attacked = attack_blur(attacked, a_params["sigma"])
        elif a_type == Attack.AWGN:
            attacked = attack_AWGN(attacked, a_params["std"], a_params["seed"], a_params["mean"])
        elif a_type == Attack.JPEG:
            attacked = attack_jpeg(attacked, a_params["QF"])
        elif a_type == Attack.MEDIAN:
            attacked = attack_median(attacked, a_params["kernel_size"])
        elif a_type == Attack.RESIZING:
            attacked = attack_resizing(attacked, a_params["scale"])
        else:
            raise KeyError(f"invalid attack in dict: {a_type}")

    if res_queue is not None:
        t_start = datetime.datetime.now()
        w = wpsnr(im, attacked)
        # TODO add similarity
        t_end = datetime.datetime.now()
        print((t_end - t_start).seconds)
        res_queue.put({id, w})

    return attacked
