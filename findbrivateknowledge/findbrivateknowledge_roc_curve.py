import io
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy.linalg import svd
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from sklearn.metrics import roc_curve, auc

import findbrivateknowledge_detection
import findbrivateknowledge_embedding


def similarity(X, X_star):
    norm_X = np.sqrt(np.sum(np.multiply(X, X)))
    norm_X_star = np.sqrt(np.sum(np.multiply(X_star, X_star)))

    if norm_X == 0 or norm_X_star == 0:
        return 0.0

    s = np.sum(np.multiply(X, X_star)) / (norm_X * norm_X_star)
    return s


def blur(img: np.ndarray, sigma) -> np.ndarray:
    """
    attack_blur(im, [2,2])
    """
    attacked = gaussian_filter(img, sigma)
    return attacked


def awgn(img: np.ndarray, std: int, seed: float, mean: float = 0.0) -> np.ndarray:
    mean = mean  # some constant
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked


def sharpening(img: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked


def median(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    best value kernel_size = 3, higher values cause lower wpsnr linearly
    """
    attacked = medfilt(img, kernel_size)
    return attacked


def resizing(img: np.ndarray, scale: float) -> np.ndarray:
    x, y = img.shape
    attacked = rescale(img, scale, preserve_range=True)
    attacked = rescale(attacked, 1 / scale, preserve_range=True)
    attacked = cv2.resize(attacked, (x, y))
    return attacked


def jpeg_compression(img: np.ndarray, QF: int):
    img = Image.fromarray(img, mode="L")
    bytes_io = io.BytesIO()
    img.save(bytes_io, "JPEG", quality=QF)
    attacked = Image.open(bytes_io)
    attacked = np.asarray(attacked, dtype=np.uint8)
    return attacked


def random_attack(img):
    i = random.randint(1, 7)
    if i == 1:
        attacked = awgn(img, 3.0, 123)
    elif i == 2:
        attacked = blur(img, [.5, .5])
    elif i == 3:
        attacked = sharpening(img, 1, 1)
    elif i == 4:
        attacked = median(img, [3, 3])
    elif i == 5:
        attacked = resizing(img, 0.8)
    elif i == 6:
        attacked = jpeg_compression(img, 50)
    elif i == 7:
        attacked = img
    return attacked, i


def step(watermark, watermarked_image, original_image):
    fakemark = np.random.uniform(0.0, 1.0, 1024)
    fakemark = np.uint8(np.rint(fakemark))
    # fakemark = np.reshape(fakemark, (12, 12))
    fakemark = findbrivateknowledge_detection.watermark_to_bytes(fakemark)
    fakemark = np.resize(fakemark, (12, 12))

    attacked_image, atk = random_attack(watermarked_image)
    wm_atk = findbrivateknowledge_detection.extraction(attacked_image, original_image)

    max_sim = None
    max_wm = None
    for w in wm_atk:
        act_sim = similarity(watermark, w)
        if max_sim is None or act_sim > max_sim:
            max_sim = act_sim
            max_wm = w

    return [(max_sim, 1), (similarity(fakemark, max_wm), 0)]


def step_custom(watermark, watermarked_image, original_image):
    fakemark = np.random.uniform(0.0, 1.0, 1024)
    fakemark = np.uint8(np.rint(fakemark))
    fakemark = findbrivateknowledge_detection.watermark_to_bytes(fakemark)
    #fakemark_path = "fakemark.npy"
    #np.save(fakemark_path, fakemark)
    #fakemark = findbrivateknowledge_detection.extraction(fake_image, original_image)

    attacked_image, atk = random_attack(watermarked_image)
    wm_atk = findbrivateknowledge_detection.extraction(attacked_image, original_image)

    w_sim = similarity(watermark, wm_atk)
    w_fake_sim = similarity(wm_atk, fakemark)

    #print(f"Real sim: {w_sim}")
    #print(f"fake sim: {w_fake_sim}")

    return [[w_sim, 1], [w_fake_sim, 0]]


def compute_roc_curve():
    start = time.time()

    sample_images = []
    for filename in os.listdir('sample_images'):
        path_tmp = os.path.join('sample_images', filename)
        sample_images.append(path_tmp)
    sample_images.sort()

    watermark_path = 'findbrivateknowledge.npy'

    scores = []
    labels = []

    for i in range(len(sample_images)):
        original_image = sample_images[i]
        watermarked_image = findbrivateknowledge_embedding.embedding(original_image, watermark_path)
        #fake_image = findbrivateknowledge_embedding.embedding(original_image, fakemark_path)

        watermark = findbrivateknowledge_detection.extraction(watermarked_image, cv2.imread(original_image, 0))
        print(sample_images[i])

        original_image_read = cv2.imread(original_image,0)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(step_custom, watermark, watermarked_image, original_image_read)
                for i in range(10)
            ]

        results = []
        for future in futures:
            res = future.result()
            labels.append(res[0][1])
            scores.append(res[0][0])
            labels.append(res[1][1])
            scores.append(res[1][0])

    # print('Scores:', scores)
    # print('Labels:', labels)

    fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=lw,
             label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    idx_tpr = np.where((fpr - 0.05) == min(i for i in (fpr - 0.05) if i > 0))
    print('For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])

    end = time.time()
    print('[COMPUTE ROC] Time: %0.2f seconds' % (end - start))


if __name__ == "__main__":
    compute_roc_curve()
