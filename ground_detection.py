# detect the ground plane by using curve fitting on the depth info
# two curves will be used
# one: a * exp(b * y) + c * exp(d * y)
# two: h / cos(theta - arctan(y / f))


import numpy as np
import cv2
import scipy.signal as signal
import scipy.optimize as opt
from matplotlib import pyplot as plt
import time

XDIM, YDIM = 640, 360

def ground_truth(file):
    """
    convert the ground truth image to binary
    """
    im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return im < 250

def exp_func(y, a, b, c, d):
    """
    exponential approximation
    """
    return a * np.exp(b * y) + c * np.exp(d * y)

def cos_func(y, h, f, theta):
    """
    approximation using camera height, focal length and the angle
    """
    return h / (np.cos(theta - np.arctan(y / f )))

def ground_plane_detection(depth, threshold=2, kernel=9):
    """
    detect the ground plane only based on the depth info, assuming there is no holes and dips in the ground
    based on the assumption, the ground should be the furthest depth
    return a binarized map where 1 indicates the ground, 0 for everything else.
    """
    # Take the max value along each row
    D = np.max(depth, axis=1)
    D = signal.medfilt(D, kernel)

    # The y value should be exponentially decaying, scan D from right to left
    ys, ds = [], []
    curr_d = 1
    for i in np.arange(len(D) - 1, 0, -1):
        if D[i] >= curr_d:
            curr_d = D[i]
            ys.append(i)
            ds.append(D[i])

    # Compute the reference ground plane curve
    a, b, c, d = opt.curve_fit(exp_func, ys, ds, p0=[1, 1, 1, -1])[0]
    # print(a, b, c, d)
    # h, f, theta = opt.curve_fit(cos_func, ys, ds)[0]

    col = np.arange(YDIM).reshape((YDIM, 1))
    col = exp_func(col, a, b, c, d)
    # col = cos_func(col, h, f, theta)
    mask = np.tile(col, XDIM)

    # threshold the depth map
    return np.abs(depth - mask) < threshold

def accuracy(true_ground, detected_ground):
    """
    compute the accuracy by comparing the true_ground and detected_ground
    """
    return np.sum(true_ground == detected_ground) * 100 / (XDIM * YDIM)


if __name__ == "__main__":
    scene = './images/812_cam-image_array_.jpg'
    scene = cv2.imread(scene, cv2.IMREAD_GRAYSCALE)
    # print(scene)

    true_ground = './images/812_true_ground.jpg'
    true_ground = ground_truth(true_ground)

    depth_file = './images/812_cam-depth_array_.png'
    depth = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)
    start = time.time()
    detected_ground = ground_plane_detection(depth, threshold=1)
    print("time elapsed: ", time.time() - start)

    print(accuracy(true_ground, detected_ground))
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(scene, cmap='gray')
    ax[0, 1].imshow(depth, cmap='gray')
    ax[1, 0].imshow(true_ground, cmap='gray')
    ax[1, 1].imshow(detected_ground, cmap='gray')
    plt.show()
