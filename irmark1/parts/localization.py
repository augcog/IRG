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

    def get_global_position(self, ar_id, rel_x, rel_y, rel_z, rel_roll, rel_pitch, rel_yaw):
        curr_ar = 0
        for i in self.ar_tags:
            if ar_id == i["Id"]:
                curr_ar = i
                break

        if curr_ar == 0:
            return False, False, False

        ar_xyz = curr_ar["Location"][:3]
        ar_rpy = curr_ar["Location"][3:] 
        #offset = 0 #ar tag x,y is actually y,x on track with angle 0
        #yaw = np.deg2rad(ar_loc[2] + offset)
        #roll, pitch, yaw = ar_rpy + [rel_roll, rel_pitch, rel_yaw]
        #roll, pitch, yaw = rel_roll, rel_pitch, rel_yaw
        roll, pitch, yaw = np.deg2rad(ar_rpy[0]), np.deg2rad(ar_rpy[1]), np.deg2rad(ar_rpy[2])
        
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0,0,1]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch),0,math.cos(pitch)]])
        Rx = np.array([[1,0,0], [0, math.cos(roll), -math.sin(roll)], [0,math.sin(roll), math.cos(roll)]])
        curr_loc = np.array([rel_x, rel_y, rel_z])
        relative_loc = np.dot(Rz, np.dot(Ry, np.dot(Rx, curr_loc)))
        #relative_loc = cob.dot(curr_loc)
        global_loc =ar_xyz + relative_loc 
        global_ang = [rel_roll +ar_rpy[0], rel_pitch + ar_rpy[1], rel_yaw + ar_rpy[2]]
        return list(np.append(global_loc, global_ang))
        
    def angles(self, rvec, tvec):
        # yaw = 0.1*180*math.atan2(tvec[0][0], tvec[0][1])
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat,tvec.T))
        affine_P = np.insert(P,len(P),np.array([0]*len(P)+[1]),0)
        P = np.linalg.inv(affine_P)
        P = np.delete(P,len(P)-1, 0)
        pitch, yaw, roll = -cv2.decomposeProjectionMatrix(P)[-1]
        pitch_t = np.sign(pitch)*(180-abs(pitch))
        return roll, pitch_t, yaw

    def distance(self,rvec,tvec):
        tvec = tvec[0]
        return math.sqrt(tvec[0]**2 + tvec[2]**2+ tvec[1]**2)

    def update(self):
        return

    def get_position_list(self, gray=None):
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        corners, ids, rej = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=parameters)
        if np.all(ids!= None):
            valid_ids = True
            rvecs, tvecs ,_ = aruco.estimatePoseSingleMarkers(corners, 0.2032, self.camera_matrix, self.distortion_coeffs)

            xs, ys, zs, rolls, pitchs, yaws, dists = [], [], [], [],[], [], []
            for i, val in enumerate(ids):
                curr_time = time.ctime(time.time())
                curr_id = val[0]
                print("curr_id",curr_id)

                rvec = rvecs[i]
                tvec = tvecs[i]
                dist = self.distance(rvec,tvec)
                roll, pitch, yaw = self.angles(rvec, tvec)
                
                #angle = 0.1*180*math.atan2(tvec[0][0], tvec[0][1])
                rel_x = tvec[0][0]
                rel_y = tvec[0][2]
                rel_z = tvec[0][1]
                x,y,z, roll, pitch, yaw= self.get_global_position(curr_id, rel_x, rel_y, rel_z, roll, pitch, yaw)
                if x == False:
                    print("Invalid AR Id")
                    return None

                xs.append(x)
                ys.append(y)
                zs.append(z)
                rolls.append(roll)
                yaws.append(yaw)
                pitchs.append(pitch)
                dists.append(dist)

            #TODO: If multiple QR codes, take the average relative to distance of those QR codes
            norm, total_x, total_y, total_theta = 0, 0, 0, 0

            positions_list = list(zip(xs, ys, zs, rolls, pitchs, yaws, dists))
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
            avg_x, avg_y, avg_z, avg_roll, avg_pitch, avg_yaw = self.avg_positions(positions_list)
            self.prev_x, self.prev_y, self.prev_z, self.prev_roll, self.prev_pitch, self.prev_yaw = avg_x, avg_y, avg_z, avg_roll, avg_pitch, avg_yaw

            print("runtime:", time.time()-start_time)
            return avg_x, avg_y, avg_z, avg_roll, avg_pitch, avg_yaw, True #Spherical Interp
        elif self.prev_x == None:
            return 0, 0, 0, 0,0,0, False
        else: #KEYFRAME MATCHING
            return  self.prev_x, self.prev_y, self.prev_z, self.prev_roll, self.prev_pitch, self.prev_yaw, False
        # else:
        #     R, T = self.keyframe_matching()
        #     angle = np.arctan2(R[1, 0], R[0, 0])
        #     pos = R * np.array([[self.prev_x], [self.prev_y], [1]]) + T

        #     self.prev_x = pos[0]
        #     self.prev_y = pos[1]
        #     self.prev_theta += angle
        #     return self.prev_x, self.prev_y, self.prev_theta, False

    def avg_positions(self, positions):
        norm = 0
        avg_x, avg_y, avg_z, avg_roll, avg_pitch, avg_yaw= 0, 0, 0,0,0,0
        print(len(positions))
        for x,y,z, roll, pitch, yaw,dist in positions:
            print(dist)
            avg_x += x/dist
            avg_y += y/dist
            avg_z += z/dist
            avg_roll += roll/dist
            avg_pitch += pitch/dist
            avg_yaw += yaw/dist 
            norm += 1/dist
        return avg_x/norm, avg_y/norm, avg_z/norm, avg_roll/norm, avg_pitch/norm, avg_yaw/norm

    # def keyframe_matching(self):
    #     """
    #     Using keyframe matching to find the transformation between self.prev_gray and self.cur_gray.
    #     :return: the transformation info, rotation and translation
    #     """
    #     # using BRIEF to find the feature descriptors
    #     prev_kp, prev_des, cur_kp, cur_des = self.feature_extraction()

    #     # feature matching using brute-force and hamming distance
    #     # prev_gray is the train image and cur_gray is the query image
    #     matches = self.BFmatching(cur_des, prev_des)

    #     # compute local 3D points
    #     cur_p3d, prev_p3d = self.compute3D(matches, cur_kp, prev_kp)

    #     # estimate the rigid transformation
    #     R, t = self.estimate_rigid_transformation(cur_p3d, prev_p3d)

    #     return R, t

    # def feature_extraction(self):
    #     """
    #     Extract the features of self.prev_gray and self.cur_gray using CenSurE detector and BRIEF
    #     :return: key points and descriptors of self.prev_gray and self.cur_gray
    #     """
    #     # Initiate STAR(CenSurE) detector and BRIEF extractor
    #     star = cv2.xfeatures2d.StarDetector_create()
    #     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    #     # find the keypoints with STAR and descriptors of self.prev_gray
    #     kp1 = star.detect(self.prev_gray, None)
    #     kp1, des1 = brief.compute(self.prev_gray, kp1)

    #     # find the keypoints with STAR and descriptors of self.cur_gray
    #     kp2 = star.detect(self.cur_gray, None)
    #     kp2, des2 = brief.compute(self.cur_gray, kp2)

    #     return kp1, des1, kp2, des2

    # def BFmatching(self, des1, des2):
    #     """
    #     Matching des1 and des2 using brute-force matching and hamming distance
    #     :param des1: query descriptors of self.cur_gray
    #     :param des2: train descriptors of self.prev_gray
    #     :return: sorted matches between des1 and des2
    #     """
    #     # print("type of des1:", type(des1))
    #     # print(type(des2))
    #     # create BFMatcher object
    #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #     matches = bf.match(des1, des2)

    #     # Sort them in the order of their distance.
    #     return sorted(matches, key=lambda x: x.distance)

    # def compute3D(self, matches, cur_kp, prev_kp):
    #     """
    #     compute the local 3d points
    #     :param matches: the feature matched using BFmatching
    #     :param prev_kp: the keypoints in self.prev_gray
    #     :param cur_kp: the keypoints in self.cur_gray
    #     :return: ndarray of points in 3D
    #     """
    #     cur_3d = []
    #     prev_3d = []
    #     for match in matches:
    #         # x is the column number and y is the row number
    #         (x1, y1) = cur_kp[match.queryIdx].pt
    #         (x2, y2) = prev_kp[match.trainIdx].pt

    #         cur_3d.append(self.get3D(x1, y1, self.cur_depth))
    #         prev_3d.append(self.get3D(x2, y2, self.prev_depth))

    #     return np.asarray(cur_3d), np.asarray(prev_3d)

    # def get3D(self, x, y, depth):
    #     """
    #     compute the 3D point corresponding to (x, y)
    #     :param x: x coordinate in 2d
    #     :param y: y coordinate in 2d
    #     :param depth: depth map for the whole image
    #     :return: 3d coordinate
    #     """
    #     # print("x: ", x)
    #     # print("y: ", y)
    #     d = depth[int(y), int(x)]
    #     x3d = (x - self.camera_matrix[0, 2]) * d / self.camera_matrix[0, 0]
    #     y3d = (y - self.camera_matrix[1, 2]) * d / self.camera_matrix[1, 1]
    #     return np.array([x3d, y3d, d])

    # def get_rigid_transformation3d(self, A, B):
    #     """
    #     Find rigid body transformation between A and B
    #     :param A: 3 X n matrix of points
    #     :param B: 3 x n matrix of points
    #     :return: rigid body transformation between A and B
    #     code is from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    #     solve the problem RA + t = B
    #     """
    #     assert len(A) == len(B)
    #     num_rows, num_cols = A.shape;
    #     if num_rows != 3:
    #         raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))
    #     [num_rows, num_cols] = B.shape;
    #     if num_rows != 3:
    #         raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    #     # find mean column wise
    #     centroid_A = np.mean(A, axis=1).reshape((3, 1))
    #     centroid_B = np.mean(B, axis=1).reshape((3, 1))

    #     # subtract mean
    #     # Am = A - np.tile(centroid_A, (1, num_cols))
    #     # Bm = B - np.tile(centroid_B, (1, num_cols))
    #     Am = A - centroid_A
    #     Bm = B - centroid_B
    #     H = Am * np.transpose(Bm)

    #     # find rotation
    #     U, S, Vt = np.linalg.svd(H)
    #     R = Vt.T * U.T

    #     # special reflection case
    #     if np.linalg.det(R) < 0:
    #         print("det(R) < R, reflection detected!, correcting for it ...\n");
    #         Vt[2, :] *= -1
    #         R = Vt.T * U.T

    #     # t = -R * centroid_A + centroid_B
    #     t = centroid_B - np.matmul(R, centroid_A)

    #     return R, t

    # def estimate_rigid_transformation(self, cur_p3d, prev_p3d, num = 10, delta = 0.1):
    #     """
    #     find the best R, T between cur_p3d and prev_p3d
    #     :param cur_p3d: n X 3 matrix of 3d points in the current gray image
    #     :param prev_p3d: n X 3 matrix of 3d points in the previous gray image
    #     :param num: max number of iterations
    #     :param delta: bound used to check the quality of transformation
    #     :return: best rigid transformation R, T
    #     """
    #     rotation = None
    #     translation = None
    #     accuracy = 0

    #     for i in range(num):
    #         prev = np.random.permutation(prev_p3d).T # make it 3xn matrix
    #         cur = np.random.permutation(cur_p3d).T
    #         A = prev[:, :3]
    #         B = cur[:, :3]
    #         A1 = prev[:, 3:]
    #         B1 = cur[:, 3:]

    #         R, t = self.get_rigid_transformation3d(A, B)
    #         # print("shape of R: ", R.shape)
    #         # print("shape of t: ", t.shape)
    #         # print("shape of A1, B1: ", A1.shape, B1.shape)
    #         # print("shape of RA1: ", (np.matmul(R, A1) - t).shape)
    #         error = np.linalg.norm(np.matmul(R, A1) + t - B1, axis = 1)
    #         acc = np.sum(error < delta)
    #         if acc > accuracy:
    #             accuracy = acc
    #             rotation = R
    #             translation = t
            
    #     return rotation, translation


    def shutdown(self):
        pass