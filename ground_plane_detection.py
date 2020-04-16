import cv2
import numpy as np
import os

NUM_IMAGES = 1353
START_NUM = 1
folder_prefix = os.getcwd() + '/tub_2_20-04-08/'
depth_suffix = '_cam-depth_array_.png'
image_suffix = '_cam-image_array_.jpg'

def ground_plane_subtraction(depth_array, image_array):
    # TODO: GROUND PLANE DETECTION AND SUBTRACTION (Returns image_array for now, will be changed later)
    return image_array

if __name__ == "__main__":
    for i in range(NUM_IMAGES):
        depth_name = folder_prefix + str(START_NUM+i) + depth_suffix
        image_name = folder_prefix + str(START_NUM+i) + image_suffix
        depth_array = cv2.imread(depth_name, cv2.IMREAD_GRAYSCALE)
        image_array = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE) #Currently a grayscale image :/
        subtracted_array = ground_plane_subtraction(depth_array, image_array)
        cv2.imshow("Depth", depth_array)
        cv2.imshow("Image", image_array)
        cv2.imshow("Subtracted", subtracted_array)
        cv2.waitKey(15)
