import cv2
import numpy as np
import os
from sklearn.linear_model import LinearRegression, Ridge

NUM_IMAGES = 1353
START_NUM = 1
folder_prefix = os.getcwd() + '/tub_2_20-04-08/'
depth_suffix = '_cam-depth_array_.png'
image_suffix = '_cam-image_array_.jpg'


def ground_plane_subtraction(depth_array, image_array):
    # TODO: GROUND PLANE DETECTION AND SUBTRACTION (Returns image_array for now, will be changed later)
    X = []
    y = []
    calib = 0.6
    linreg = Ridge()
    for i in range(len(depth_array)):
        for j in range(len(depth_array[0])):
            if (depth_array[i][j] != 0):
                X.append([1, i])
                y.append(np.log(depth_array[i][j]))
    linreg.fit(X=X,y=y)
    results = linreg.predict(X)
    counter = 0
    for i in range(len(depth_array)):
        for j in range(len(depth_array[0])):
            if (depth_array[i][j] != 0):
                if abs(results[counter] - np.log(depth_array[i][j])) < calib:
                    image_array[i][j] = 0
                counter += 1
            else:
                #image_array[i][j] = 0
                pass
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
