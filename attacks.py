import datetime
import io
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


def attack_sharpen(img: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked


def attack_median(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    best value kernel_size = 3, higher values cause lower wpsnr linearly
    """
    attacked = medfilt(img, kernel_size)
    return attacked


def attack_resizing(img: np.ndarray, scale: float) -> np.ndarray:
    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1 / scale)
    attacked = attacked[:x, :y]
    return attacked


def attack_jpeg(img: np.ndarray, QF: int):
    img = Image.fromarray(img, mode="L")
    bytes_io = io.BytesIO()
    img.save(bytes_io, "JPEG", quality=QF)
    attacked = Image.open(bytes_io)
    attacked = np.asarray(attacked, dtype=np.uint8)
    return attacked


def apply_attack_queue(im: str,
                       l: list,
                       detection_function=None) -> Tuple[np.ndarray, int, float, list]:
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
    If the detection function is passed, it is ran and the output will be returned by this function
    Returns:
        A tuple containing:
        - The attacked image
        - An integer indicating if the watermark has been found by the detection function
        - The wpsnr value, if the detection function is passed
        - The attack
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

    if detection_function is not None:
        detection_attacked_path = os.path.join(CACHE_PATH, f"attacked_{uuid.uuid4()}.bmp")
        cv2.imwrite(detection_attacked_path, attacking)
        contains_w, wpsnr = detection_function(ORIGINAL_IMG_PATH, im, detection_attacked_path)
        os.remove(detection_attacked_path)
        if contains_w == 0:  # TODO remove
            print(f"Contains w?: {contains_w}, WPSNR: {wpsnr}. Attack: {l}")
    else:
        print("Attack applied: ", l)

    return attacking, contains_w, wpsnr, l


def prepare_attacks():
    blur = lambda x, y: {"attack": Attack.BLUR, "params": {"sigma": [x, y]}}
    median = lambda x: {"attack": Attack.MEDIAN, "params": {"kernel_size": x}}
    jpeg = lambda x: {"attack": Attack.JPEG, "params": {"QF": x}}
    sharpen = lambda x, y: {"attack": Attack.SHARPEN, "params": {"sigma": x, "alpha": y}}
    awgn = lambda x, y, z: {"attack": Attack.AWGN, "params": {"std": x, "seed": y, "mean": z}}
    resizing = lambda x: {"attack": Attack.RESIZING, "params": {"scale": x}}

    res = []

    if False:
        for i in range(10, 150, 10):
            res.append([blur(i / 100, i / 100)])

    if False:
        for i in range(1, 6, 2):
            res.append([median(i)])

    if False:
        for i in range(1, 40, 1):
            res.append([jpeg(i)])

    # Most time-consuming attacks:
    if False:
        for i in range(50, 100, 10):
            for j in range(1, 6, 1):
                res.append([sharpen(i / 100, j)])  # do not convert j to float

    if False:
        for i in range(13, 16):
            for j in range(30, 120, 10):
                res.append([awgn(i, j, 0.0)])

    # Probably broken
    if False:  # NOT USE
        for i in range(1, 100):
            res.append([resizing(i / 100)])

    return res


def prepare_joint_attacks():
    """
    Used to prepare a list of attacks,
    Returns:

    """
    blur = lambda x, y: {"attack": Attack.BLUR, "params": {"sigma": [x, y]}}
    median = lambda x: {"attack": Attack.MEDIAN, "params": {"kernel_size": x}}
    jpeg = lambda x: {"attack": Attack.JPEG, "params": {"QF": x}}
    sharpen = lambda x, y: {"attack": Attack.SHARPEN, "params": {"sigma": x, "alpha": y}}
    awgn = lambda x, y, z: {"attack": Attack.AWGN, "params": {"std": x, "seed": y, "mean": z}}

    res = []

    for i in range(1, 50, 5):
        for j in range(95, 101):
            act = []
            # act.append(sharpen(i / 100, 1))
            # act.append(awgn(i, 1, 0.0))
            if False:
                act.append(blur((i + 10) / 100, (i + 10) / 100))
                act.append(jpeg(j))
            elif True:
                act.append(jpeg(j))
                act.append(blur((i) / 100, (i) / 100))
            res.append(act)

    return res


if __name__ == "__main__":
    # Best starting value for attacks
    if not os.path.isdir(CACHE_PATH):
        os.mkdir(CACHE_PATH)
    attacks = [
        [{"attack": Attack.BLUR, "params": {"sigma": [0.5, 0.5]}}, ],
        [{"attack": Attack.JPEG, "params": {"QF": 5}}],
        [{"attack": Attack.AWGN, "params": {"std": 14, "seed": 101, "mean": 0}}],
        [{"attack": Attack.MEDIAN, "params": {"kernel_size": 3}}],
        [{"attack": Attack.RESIZING, "params": {"scale": 1}}],
        [{'attack': Attack.SHARPEN, 'params': {'sigma': 1, 'alpha': 1}}],
        [{'attack': Attack.SHARPEN, 'params': {'sigma': 0.71, 'alpha': 1}}],
        [{"attack": Attack.AWGN, "params": {"std": 14, "seed": 101, "mean": 0}},
         {"attack": Attack.JPEG, "params": {"QF": 5}}],
    ]

    # attacks = prepare_attacks()
    attacks = prepare_joint_attacks()

    watermarked_img_path = 'watermarked_image.bmp'
    watermarked_img = cv2.imread(watermarked_img_path, 0)
    original_img = cv2.imread(ORIGINAL_IMG_PATH, 0)
    watermark = np.load('findbrivateknowledge.npy')
    watermark = np.reshape(watermark, (32, 32))

    time_start = datetime.datetime.now()

    # Execute attacks
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(apply_attack_queue, watermarked_img_path, attack, detection)
            for i, attack in enumerate(attacks)
        ]

    results = []
    for future in futures:
        results.append(future.result())

    # To beat max wpsnr: 51.86934243636088, attack: [{'attack': <Attack.BLUR: 0>, 'params': {'sigma': [0.5, 0.5]}}]

    U_wm, S_wm, V_wm = svd(watermark)
    # np.savetxt("U_wm", U_wm)
    # np.savetxt("S_wm", S_wm)
    # np.savetxt("V_wm", V_wm)

    f = []

    max_wpsnr = 0
    max_a = None

    for i, result in enumerate(results):
        res_attacked, res_contains_w, res_wpsnr, a = result

        if res_contains_w == 0:
            if max_wpsnr != 9999999:
                if res_wpsnr > max_wpsnr:
                    max_wpsnr = res_wpsnr
                    max_a = a
        # if res_contains_w == 0:
        #    print(f"Success w?: {res_contains_w}, WPSNR: {res_wpsnr}")
        # watermarks = extraction(res_attacked, original_img, U_wm, V_wm)
        # watermarks_values = [similarity(watermark, wm) for wm in watermarks]
        # print(watermarks_values, wpsnr(watermarked_img, res_attacked), attacks[i][0]["attack"])

    time_end = datetime.datetime.now()

    print(f"Time taken: {(time_end - time_start).seconds} seconds")
    print(f"max wpsnr: {max_wpsnr}, attack: {max_a}")
