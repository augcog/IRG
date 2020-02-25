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

        self.json_in = json_in
        print(os.path.abspath(json_in))
        content = open(self.json_in)
        self.map =  json.loads(content.read())
        self.qr_loc = self.map["QR codes"]

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

    def get_global_xy_position(self, qr_id, rel_x, rel_y, angle):
        qr_loc = self.qr_loc[qr_id]["Location"]
        line = self.qr_loc[qr_id]["Segment"]

        segment = self.map["Track"]["Lines"][line]
        (s_x, s_y) = segment["Start"]
        (e_x, e_y) = segment["End"]

        x, y = 0, 0
        if abs(e_x - s_x) > abs(e_y - s_y):
            #car is moving in the x direction
            if s_x < e_x:
                x = qr_loc[0] - rel_x
                y = qr_loc[1] - rel_y
            else: 
                x = qr_loc[0] + rel_x
                y = qr_loc[1] - rel_y
        else: 
            #car is moving in the y direction
            if s_y < e_y:
                x = qr_loc[0] - rel_x
                y = qr_loc[1] - rel_y
            else: 
                x = qr_loc[0] - rel_x
                y = qr_loc[1] + rel_y

        return x, y
        
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

                rvec = rvecs[i]
                tvec = tvecs[i]
                dist = self.distance(rvec,tvec)
                theta = self.angle(rvec, tvec)
                
                angle = 0.1*180*math.atan2(tvec[0][0], tvec[0][1])
                rel_x = tvec[0][0]
                rel_y = tvec[0][2]
                x,y= self.get_global_xy_position(curr_id, rel_x, rel_y, theta)
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
                return 0,0,self.mode,self.recording
        else:
            return 0,0,self.mode,self.recording
        
        #finds all the QR codes seen in the current frame
        img_arr = img_arr.copy()
        valid_ids = False
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
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
            R, T = self.keyframe_matching()
            angle = self.angle(R, T)
            pos = R * np.array([[self.prev_x], [self.prev_y], [1]]) + T

            self.prev_x = pos[0]
            self.prev_y = pos[1]
            self.prev_theta += angle
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

    def keyframe_matching(self):
        """
        Using keyframe matching to find the transformation between self.prev_gray and self.cur_gray.
        :return: the transformation info, rotation and translation
        """
        # using BRIEF to find the feature descriptors
        prev_kp, prev_des, cur_kp, cur_des = self.feature_extraction()

        # feature matching using brute-force and hamming distance
        # prev_gray is the train image and cur_gray is the query image
        matches = self.BFmatching(cur_des, prev_des)

        # compute local 3D points
        cur_p3d, prev_p3d = self.compute3D(matches, cur_kp, prev_kp)

        # estimate the rigid transformation
        R, t = self.estimate_rigid_transformation(cur_p3d, prev_p3d)

        return R, t

    def feature_extraction(self):
        """
        Extract the features of self.prev_gray and self.cur_gray using CenSurE detector and BRIEF
        :return: key points and descriptors of self.prev_gray and self.cur_gray
        """
        # Initiate STAR(CenSurE) detector and BRIEF extractor
        star = cv2.FeatureDetector_create("STAR")
        brief = cv2.DescriptorExtractor_create("BRIEF")

        # find the keypoints with STAR and descriptors of self.prev_gray
        kp1 = star.detect(self.prev_gray, None)
        kp1, des1 = brief.compute(self.prev_gray, kp1)

        # find the keypoints with STAR and descriptors of self.cur_gray
        kp2 = star.detect(self.cur_gray, None)
        kp2, des2 = brief.compute(self.cur_gray, kp2)

        return kp1, des1, kp2, des2

    def BFmatching(self, des1, des2):
        """
        Matching des1 and des2 using brute-force matching and hamming distance
        :param des1: query descriptors of self.cur_gray
        :param des2: train descriptors of self.prev_gray
        :return: sorted matches between des1 and des2
        """
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        return sorted(matches, key=lambda x: x.distance)

    def compute3D(self, matches, cur_kp, prev_kp):
        """
        compute the local 3d points
        :param matches: the feature matched using BFmatching
        :param prev_kp: the keypoints in self.prev_gray
        :param cur_kp: the keypoints in self.cur_gray
        :return: ndarray of points in 3D
        """
        cur_3d = []
        prev_3d = []
        for match in matches:
            (x1, y1) = cur_kp[match.queryIdx].pt
            (x2, y2) = prev_kp[match.trainIdx].pt

            cur_3d.append(self.get3D(x1, y1, self.cur_depth))
            prev_3d.append(self.get3D(x2, y2, self.prev_depth))

        return np.asarray(cur_3d), np.asarray(prev_3d)

    def get3D(self, x, y, depth):
        """
        compute the 3D point corresponding to (x, y)
        :param x: x coordinate in 2d
        :param y: y coordinate in 2d
        :param depth: depth map for the whole image
        :return: 3d coordinate
        """
        d = depth[x, y]
        x3d = (x - self.camera_matrix[0, 2]) * d / self.camera_matrix[0, 0]
        y3d = (y - self.camera_matrix[1, 2]) * d / self.camera_matrix[1, 1]
        return np.array([x3d, y3d, d])

    def get_rigid_transformation3d(self, A, B):
        """
        Find rigid body transformation between A and B
        :param A: 3 X n matrix of points
        :param B: 3 x n matrix of points
        :return: rigid body transformation between A and B
        code is from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
        solve the problem RA + t = B
        """
        assert len(A) == len(B)
        num_rows, num_cols = A.shape;
        if num_rows != 3:
            raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))
        [num_rows, num_cols] = B.shape;
        if num_rows != 3:
            raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # subtract mean
        Am = A - np.tile(centroid_A, (1, num_cols))
        Bm = B - np.tile(centroid_B, (1, num_cols))
        H = Am * np.transpose(Bm)

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T * U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            print("det(R) < R, reflection detected!, correcting for it ...\n");
            Vt[2, :] *= -1
            R = Vt.T * U.T

        t = -R * centroid_A + centroid_B

        return R, t

    def estimate_rigid_transformation(self, cur_p3d, prev_p3d, num = 10, delta = 0.1):
        """
        find the best R, T between cur_p3d and prev_p3d
        :param cur_p3d: n X 3 matrix of 3d points in the current gray image
        :param prev_p3d: n X 3 matrix of 3d points in the previous gray image
        :param num: max number of iterations
        :param delta: bound used to check the quality of transformation
        :return: best rigid transformation R, T
        """
        rotation = None
        translation = None
        accuracy = 0

        for i in range(num):
            prev = np.random.permutation(prev_p3d).T # make it 3xn matrix
            cur = np.random.permutation(cur_p3d).T
            A = prev[:, :3]
            B = cur[:, :3]
            A1 = prev[:, 3:]
            B1 = cur[:, 3:]

            R, t = self.get_rigid_transformation3d(A, B)
            error = np.linalg.norm(R * A1 + t - B1, axis = 1)
            acc = np.sum(error < delta)
            if acc > accuracy:
                accuracy = acc
                rotation = R
                translation = t

        return rotation, translation

    def shutdown(self):
        pass