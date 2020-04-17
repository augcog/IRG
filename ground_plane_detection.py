import cv2
import numpy as np
import os
from sklearn.linear_model import LinearRegression, Ridge
import scipy.optimize as optimization
import time

NUM_IMAGES = 1353
START_NUM = 500
folder_prefix = os.getcwd() + '/tub_2_20-04-08/'
depth_suffix = '_cam-depth_array_.png'
image_suffix = '_cam-image_array_.jpg'

def exponential_func(x, a, b, c, d):
    return a*np.exp(b*x) + c*np.exp(d*x)

def ground_plane_subtraction(depth_array, image_array):
    # TODO: GROUND PLANE DETECTION AND SUBTRACTION (Returns image_array for now, will be changed later)
    
    tol =0.04
    a_s = []
    b_s = []
    c_s = []
    d_s = []
    train_orig_time = time.time()
    error_count=0
    for j in range(len(depth_array[0])):
        x = []
        y = []
        for i in range(len(depth_array)):
            if (depth_array[i][j] != 0):
                x.append(len(depth_array)-i)
                y.append(depth_array[i][j])
        if len(x) > 5:
            try:
                a,b,c,d = optimization.curve_fit(exponential_func, x, y, p0 = [1,0,1,0], maxfev=200)[0]
                a_s.append(a)
                b_s.append(b)
                c_s.append(c)
                d_s.append(d)
            except RuntimeError:
                error_count += 1
                continue
    a = np.median(a_s)
    b = np.median(b_s)
    c = np.median(c_s)
    d = np.median(d_s)
    orig_time = time.time()
    print("Fit Errors:", error_count)
    print("train time:", orig_time-train_orig_time)
    for i in range(len(depth_array)):
        for j in range(len(depth_array[0])):
            if (depth_array[i][j] != 0):
                if abs(exponential_func(len(depth_array)-i, a, b, c, d) - (depth_array[i][j])) < tol*(len(depth_array)-i):
                    image_array[i][j] = 0
                else:
                    #print(exponential_func(len(depth_array)-i, a, b, c, d) - depth_array[i][j])
                    pass
    inference_time = time.time() - orig_time
    print("inference time:", inference_time)
    return image_array

if __name__ == "__main__":
    for i in range(NUM_IMAGES):
        depth_name = folder_prefix + str(START_NUM+i) + depth_suffix
        image_name = folder_prefix + str(START_NUM+i) + image_suffix
        depth_array = cv2.imread(depth_name, cv2.IMREAD_GRAYSCALE)
        image_array = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE) #Currently a grayscale image :/
        cv2.imshow("Depth", depth_array)
        #print(depth_array.shape)
        cv2.imshow("Image", image_array)
        #print(image_array.shape)
        subtracted_array = ground_plane_subtraction(depth_array, image_array)
        cv2.imshow("Subtracted", subtracted_array)
        cv2.waitKey(15)
