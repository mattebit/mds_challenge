import os
from math import sqrt

import cv2
import numpy as np
from scipy.signal import convolve2d

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


def jpeg_compression(img, QF):
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    attacked = cv2.imread('tmp.jpg', 0)
    os.remove('tmp.jpg')
    return attacked


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

    lena = cv2.imread('sample_images/lena_grey.bmp', 0)
    if False:
        w = np.load("findbrivateknowledge.npy")
        w = watermark_to_bytes(w)
        w = np.resize(w, (32, 32))

        embedded = embedding('lena_grey.bmp', "findbrivateknowledge.npy")
        # cv2.imwrite('embedded.bmp', embedded)
        wms = extraction(embedded, lena)

        for wm in wms:
            print(similarity(w, wm))

        x = 345
        y = 105
        block = lena[x:x + 64, y:y + 64]
        LL, (LH, HL, HH) = pywt.dwt2(block, 'haar')

        U, S, V = np.linalg.svd(LL)
        Uw, Sw, Vw = np.linalg.svd(w)
        # np.savetxt("Uw", Uw)
        # np.savetxt("Sw", Sw)
        # np.savetxt("Vw", Vw)

        # Embed
        S_embedded = S + Sw
        watermark_embedded = np.dot(U, np.dot(np.diag(S_embedded), V))
        block_prime = pywt.idwt2((watermark_embedded, (LH, HL, HH)), 'haar')
        lena[x:x + 64, y:y + 64] = block_prime

        # print(np.alltrue(lena==embedded))
        print(f"lena and watermarked: {similarity(lena, embedded)}")
        # SONO UGUALI DIO

        # Extract
        LL, (LH, HL, HH) = pywt.dwt2(lena[x:x + 64, y:y + 64], 'haar')
        Uwm, Swm, Vwm = np.linalg.svd(LL)
        # Uwm, Swm, Vwm = np.linalg.svd(watermark_embedded)
        S_extracted = abs(Swm - S)

        w_reconstructed = np.dot(Uw, np.dot(np.diag(S_extracted), Vw))
        w_reconstructed_2 = extraction(embedded, cv2.imread('sample_images/lena_grey.bmp', 0))[0]

        print(f"sim wr e wr2: {similarity(w_reconstructed, w_reconstructed_2)}")

        print(f"Extracted sim: {similarity(w_reconstructed, w)}")

        print(np.alltrue(w == w_reconstructed))
        w = w.flatten()
        w_reconstructed = w_reconstructed.flatten()
        for i in range(len(w)):
            if w[i] != w_reconstructed[i]:
                pass  # print(w[i], w_reconstructed[i])

        exit()

        w_reconstructed = np.dot(Uw, np.dot(np.diag(Sw), Vw))

        print(f"asd: {similarity(w, w_reconstructed)}")
        w = w.flatten()
        w_reconstructed = w_reconstructed.flatten()

        print(np.alltrue(w == w_reconstructed))
        for i in range(len(w)):
            if w[i] != w_reconstructed[i]:
                pass
                # print(w[i], w_reconstructed[i])
        exit()

    while 1:
        ORIGINAL_IMAGE_PATH = "lena_grey.bmp"
        WATERMARKED_IMAGE_PATH = "watermarked_image.bmp"
        FAKE_IMAGE_PATH = "fake_embedded.bmp"
        WATERMARK_PATH = "findbrivateknowledge.npy"
        RANDOM_WATERMARK_PATH = "random_watermark.npy"
        ATTACKED_IMAGE_PATH = "attacked.bmp"

        watermarked_image = embedding(ORIGINAL_IMAGE_PATH, WATERMARK_PATH)
        cv2.imwrite(WATERMARKED_IMAGE_PATH, watermarked_image)

        random_watermark = np.random.uniform(0.0, 1.0, 1024)
        random_watermark = np.uint8(np.rint(random_watermark))
        np.save(RANDOM_WATERMARK_PATH, random_watermark)

        fake_image = embedding(ORIGINAL_IMAGE_PATH, RANDOM_WATERMARK_PATH)
        cv2.imwrite(FAKE_IMAGE_PATH, fake_image)

        det1, wpsnr1 = detection(ORIGINAL_IMAGE_PATH, WATERMARKED_IMAGE_PATH, FAKE_IMAGE_PATH)
        det2, wpsnr2 = detection(ORIGINAL_IMAGE_PATH, WATERMARKED_IMAGE_PATH, WATERMARKED_IMAGE_PATH)
        attacked_image = jpeg_compression(watermarked_image, 50)
        cv2.imwrite(ATTACKED_IMAGE_PATH, attacked_image)
        det3, wpsnr3 = detection(ORIGINAL_IMAGE_PATH, WATERMARKED_IMAGE_PATH, ATTACKED_IMAGE_PATH)
        print("Det 1:", det1, " - wpsnr:", wpsnr1)
        print("Det 2:", det2, " - wpsnr:", wpsnr2)
        print("Det 3:", det3, " - wpsnr:", wpsnr3)
        if det1 == 1:
            break