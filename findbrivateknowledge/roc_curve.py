import cv2
import numpy as np
import pywt
import time
import os
import random
from scipy.linalg import svd
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from math import sqrt
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from sklearn.metrics import roc_curve, auc
import embedding, detection

def similarity(X, X_star):
    norm_X = np.sqrt(np.sum(np.multiply(X, X)))
    norm_X_star = np.sqrt(np.sum(np.multiply(X_star, X_star)))

    if norm_X == 0 or norm_X_star == 0:
        return 0.0

    s = np.sum(np.multiply(X, X_star)) / (norm_X * norm_X_star)
    return s

def jpeg_compression(img, QF):
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    attacked = cv2.imread('tmp.jpg', 0)
    os.remove('tmp.jpg')
    return attacked

def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked

def awgn(img, std, seed):
    mean = 0.0
    # np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked

def median(img, kernel_size):
    attacked = medfilt(img, kernel_size)
    return attacked

def resizing(img, scale):
  from skimage.transform import rescale
  x, y = img.shape
  attacked = rescale(img, scale, preserve_range=True)
  attacked = rescale(attacked, 1 / scale, preserve_range=True)
  attacked = cv2.resize(attacked, (x, y))
  return attacked

def random_attack(img):
  i = random.randint(1,7)
  if i==1:
    attacked = awgn(img, 3.0, 123)
  elif i==2:
    attacked = blur(img, [3, 3])
  elif i==3:
    attacked = sharpening(img, 1, 1)
  elif i==4:
    attacked = median(img, [3, 3])
  elif i==5:
    attacked = resizing(img, 0.8)
  elif i==6:
    attacked = jpeg_compression(img, 75)
  elif i ==7:
     attacked = img
  return attacked, i

def compute_roc_curve():
    start = time.time()
    
    sample_images = []
    for filename in os.listdir('sample_images'):
        path_tmp = os.path.join('sample_images', filename)
        sample_images.append(path_tmp)
    sample_images.sort()

    watermark_path = 'findbrivateknowledge.npy'
    watermark = np.load(watermark_path)
    watermark = cv2.resize(watermark, (32, 32))

    scores = []
    labels = []

    for i in range(len(sample_images)):
        original_image = sample_images[i]
        watermarked_image = embedding.embedding(original_image, watermark_path)

        original_image = cv2.imread(original_image, 0)
        print(sample_images[i])

        sample = 0
        while sample < 10:
            fakemark = np.random.uniform(0.0, 1.0, 1024)
            fakemark = np.uint8(np.rint(fakemark))
            fakemark = cv2.resize(fakemark, (32, 32))

            attacked_image, atk = random_attack(watermarked_image)
            wm_atk = detection.extraction(attacked_image, original_image)
            
            max_sim = None
            max_wm = None
            for w in wm_atk:
                act_sim = similarity(watermark, w)
                if max_sim is None or act_sim > max_sim:
                    max_sim = act_sim
                    max_wm = w

            scores.append(max_sim)
            labels.append(1)
            
            scores.append(similarity(fakemark, max_wm))
            labels.append(0)
            sample += 1

    #print('Scores:', scores)
    #print('Labels:', labels)

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

compute_roc_curve()