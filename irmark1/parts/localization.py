import json 
import time
import math
import os
from cv2 import aruco
import numpy as np
import cv2

class Localization(object):
    '''
    allow for the car to locate its global positioning
    '''
    
    def __init__(self, camera_matrix, distortion_coeffs, json_in):
        print("initiated Localization")

        self.json_in_path = json_in
        print(os.path.abspath(json_in))
        content = open(self.json_in_path)
        self.json_in =  json.loads(content.read()) 
        self.map = self.json_in["Segments"]
        self.ar_tags = self.json_in["AR tags"]

        #Calibration for Camera Matrix/Distortion Coefficients
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.prev_x = None
        self.prev_y= None
        self.prev_theta=None
        self.prev_gray=None
        self.cur_gray=None
        self.prev_depth=None
        self.cur_depth = None

    def get_global_xy_position(self, ar_id, rel_x, rel_y, angle):
        curr_ar = 0
        for i in self.ar_tags:
            if ar_id == i["Id"]:
                curr_ar = i
                break

        if curr_ar == 0:
            return False, False, False

        ar_loc = curr_ar["Location"] 
 
        offset = 90 #ar tag x,y is actually y,x on track with angle 0
        yaw = np.deg2rad(ar_loc[2] + offset)
        cob = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0,0,1]])

        curr_loc = np.array([rel_x, rel_y, angle])

        relative_loc = cob.dot(curr_loc)
        global_loc = ar_loc + relative_loc 

        return global_loc
        
    def angle(self, rvec, tvec):
        # yaw = 0.1*180*math.atan2(tvec[0][0], tvec[0][1])
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat,tvec.T))
        affine_P = np.insert(P,len(P),np.array([0]*len(P)+[1]),0)
        P = np.linalg.inv(affine_P)
        P = np.delete(P,len(P)-1, 0)
        eul = -cv2.decomposeProjectionMatrix(P)[6]
        yaw = eul[1,0] #rotational angle
        return yaw

    def distance(self,rvec,tvec):
        tvec = tvec[0]
        return math.sqrt(tvec[0]**2 + tvec[2]**2)

    def update(self):
        return

    def get_position_list(self, gray=None):
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        corners, ids, rej = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=parameters)
        if np.all(ids!= None):
            valid_ids = True
            rvecs, tvecs ,_ = aruco.estimatePoseSingleMarkers(corners, 0.2032, self.camera_matrix, self.distortion_coeffs)

            xs, ys, thetas, dists = [], [], [], []
            for i, val in enumerate(ids):
                curr_time = time.ctime(time.time())
                curr_id = val[0]
                print("curr_id",curr_id)

                rvec = rvecs[i]
                tvec = tvecs[i]
                dist = self.distance(rvec,tvec)
                theta = self.angle(rvec, tvec)
                
                angle = 0.1*180*math.atan2(tvec[0][0], tvec[0][1])
                rel_x = tvec[0][0]
                rel_y = tvec[0][2]
                x,y, theta= self.get_global_xy_position(curr_id, rel_x, rel_y, theta)
                if x == False and y == False and theta == False:
                    print("Invalid AR Id")
                    return None

                xs.append(x)
                xs.append(x)
                ys.append(y)
                thetas.append(theta)
                dists.append(dist)

            #TODO: If multiple QR codes, take the average relative to distance of those QR codes
            norm, total_x, total_y, total_theta = 0, 0, 0, 0

            positions_list = zip(xs, ys, thetas, dists)
            return positions_list
        return None

    def run_threaded(self, img_arr=None, depth_arr=None):
        start_time = time.time()
        if type(img_arr) == np.ndarray:
            if not img_arr.size:
                return 0,0,0, False
        else:
            return 0,0,0, False
        
        #finds all the QR codes seen in the current frame
        img_arr = img_arr.copy()
        valid_ids = False
        if len(img_arr)  == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_arr
        self.prev_gray, self.prev_depth = self.cur_gray, self.cur_depth
        self.cur_gray, self.cur_depth = gray, depth_arr
        
        positions_list = self.get_position_list(self.cur_gray)
        if (positions_list!=None):
            avg_x, avg_y, avg_theta = self.avg_positions(positions_list)
            self.prev_x = avg_x
            self.prev_y = avg_y
            self.prev_theta = avg_theta
            print("runtime:", time.time()-start_time)
            return avg_x, avg_y, avg_theta, True #Spherical Interp
        else:
            #TODO: Keyframe Matching for previous image
            return self.prev_x, self.prev_y, self.prev_theta, False

    def avg_positions(self, positions):
        norm = 0
        avg_x, avg_y, avg_theta = 0, 0, 0
        for x,y,theta,dist in positions:
            avg_x += x/dist
            avg_y += y/dist
            avg_theta += theta/dist 
            norm += 1/dist
        return avg_x/norm, avg_y/norm, avg_theta/norm

    def shutdown(self):
        pass