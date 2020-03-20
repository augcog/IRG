import json
import time
import math
import os
from cv2 import aruco
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class Localization(object):
    '''
    allow for the car to locate its global positioning
    '''

    def __init__(self, camera_matrix, distortion_coeffs, json_in):
        print("initiated Localization")

        #Parse the json to get the map
        self.json_in_path = json_in
        print(os.path.abspath(json_in))
        content = open(self.json_in_path)
        self.json_in =  json.loads(content.read())
        self.map = self.json_in["Segments"]
        self.ar_tags = self.json_in["AR tags"]
        self.ar_configs = {}

        #Set the Camera Matrix/Distortion Coefficients, initiallize other variables
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

        self.fast = cv2.FastFeatureDetector_create()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.camera_matrix_inv = np.linalg.pinv(self.camera_matrix)

        self.ARglobal2local = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.prev_config = None
        self.config_AR()

    def config_AR(self):
        """
        compute the configuration of the local coordinates of ar tags with respect to world frame
        """
        for i in self.ar_tags:
            t = i["Location"][:3]
            ar_rpy = i["Location"][3:]
            roll, pitch, yaw = np.deg2rad(ar_rpy[0]), np.deg2rad(ar_rpy[1]), np.deg2rad(ar_rpy[2])
            Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0,0,1]])
            Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch),0,math.cos(pitch)]])
            Rx = np.array([[1,0,0], [0, math.cos(roll), -math.sin(roll)], [0,math.sin(roll), math.cos(roll)]])

            R = np.dot(Rz, np.dot(Ry, Rx))
            R = np.dot(R, self.ARglobal2local)

            # Do rotation first, then translation. after rotation, the coordinates of translation changed.
            t = np.mat(t).T

            config = np.hstack((R, t))
            config = np.vstack((config, np.mat([0,0,0,1])))
            self.ar_configs[i["Id"]]= config


    def get_global_position(self, ar_id, rel_x, rel_y, rel_z, rel_roll, rel_pitch, rel_yaw): #Return global position taking into account relative positive and where ar is on map
        curr_ar = 0
        for i in self.ar_tags:
            if ar_id == i["Id"]:
                curr_ar = i
                break

        if curr_ar == 0:
            return False, False, False

        ar_xyz = curr_ar["Location"][:3]
        ar_rpy = curr_ar["Location"][3:]

        #Get roll pitch and yaw in radians
        roll, pitch, yaw = np.deg2rad(ar_rpy[0]), np.deg2rad(ar_rpy[1]), np.deg2rad(ar_rpy[2])

        #Apply rotation matrix and translation vector to get new global xyz and global angle
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0,0,1]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch),0,math.cos(pitch)]])
        Rx = np.array([[1,0,0], [0, math.cos(roll), -math.sin(roll)], [0,math.sin(roll), math.cos(roll)]])
        curr_loc = np.array([rel_x, rel_y, rel_z])
        relative_loc = np.dot(Rz, np.dot(Ry, np.dot(Rx, curr_loc)))
        #relative_loc = cob.dot(curr_loc)
        global_loc =ar_xyz + relative_loc
        global_ang = [rel_roll +ar_rpy[0], rel_pitch + ar_rpy[1], rel_yaw + ar_rpy[2]]
        return list(np.append(global_loc, global_ang))



    def angles(self, rvec, tvec): #Get roll pitch and yaw from rotation vector and translation vector
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat,tvec.T))
        affine_P = np.insert(P,len(P),np.array([0]*len(P)+[1]),0)
        P = np.linalg.inv(affine_P)
        P = np.delete(P,len(P)-1, 0)
        pitch, yaw, roll = -cv2.decomposeProjectionMatrix(P)[-1] #Use projection matrix to get pitch yaw roll
        pitch_t = np.sign(pitch)*(180-abs(pitch))
        return roll, pitch_t, yaw

    def distance(self,rvec,tvec): #Get distance in terms of relative translation (Just an l-2 norm of tvec)
        tvec = tvec[0]
        return math.sqrt(tvec[0]**2 + tvec[2]**2+ tvec[1]**2)

    def update(self):
        return

    def get_position_list(self, gray=None): #Get list of all global positions based on each ar tag in the frame
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        #Detect the ids
        corners, ids, rej = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=parameters)
        if np.all(ids!= None):
            valid_ids = True
            #Get rvec and tvec of each id relative to detected ar tags
            rvecs, tvecs ,_ = aruco.estimatePoseSingleMarkers(corners, 0.2032, self.camera_matrix, self.distortion_coeffs)

            xs, ys, zs, rolls, pitchs, yaws, dists = [], [], [], [],[], [], []
            config_list = []
            for i, val in enumerate(ids):
                curr_time = time.ctime(time.time())
                curr_id = val[0]
                print("curr_id",curr_id)

                #Get the distances and angles
                rvec = rvecs[i]
                tvec = tvecs[i]

                #rmatrix from rvec
                rmat = cv2.Rodrigues(rvec)[0]
                P = np.hstack((rmat,tvec.T))
                P = np.vstack((P, [0,0,0,1]))
                ARlocal2Car = np.linalg.inv(P)

                World2ARlocal = self.ar_configs[curr_id]
                World2Car = np.dot(World2ARlocal, ARlocal2Car)

                config_list.append(World2Car)
                dist = self.distance(rvec,tvec)
                dists.append(dist)

            #Format positions list correctly
            positions_list = list(zip(config_list, dists))
            return positions_list
        return None

    def run_threaded(self, img_arr=None, depth_arr=None):
        start_time = time.time()
        if type(img_arr) == np.ndarray:
            if not img_arr.size:
                return 0,0,0, False
        else:
            return 0,0,0, False

        #Copy image then convert it to grayscale
        img_arr = img_arr.copy()
        if len(img_arr)  == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_arr

        if self.cur_gray is not None:
            self.prev_gray, self.prev_depth = self.cur_gray, self.cur_depth

        # self.cur_gray, self.cur_depth = gray, depth_arr

        #Get the position list from the gray scale image, if there are ar tags
        # positions_list = self.get_position_list(self.cur_gray)
        positions_list = self.get_position_list(gray)

        if (positions_list!=None):
            #Average out the ar tag positions based on a weighted average
            self.prev_config = np.array(self.avg_RT(positions_list))
            #self.prev_x, self.prev_y, self.prev_z, self.prev_roll, self.prev_pitch, self.prev_yaw = avg_x, avg_y, avg_z, avg_roll, avg_pitch, avg_yaw

            print("runtime:", time.time()-start_time)
            self.cur_gray, self.cur_depth = gray, depth_arr

            return self.prev_config, True


        elif self.prev_gray is not None: #KEYFRAME MATCHING
            if self.prev_config is None:
                return None, False
            #Run keyframe matching function
            R, T = self.keyframe_matching(gray, depth_arr)
            if T is None or R is None:
                self.cur_gray, self.cur_depth = None, None
                return None, False
            #Normalize millimeters to meters
            T =T/1000
            g = np.hstack((R, T))
            g = np.vstack((g, [0, 0, 0, 1]))

            self.prev_config = np.dot(self.prev_config, g)
            self.cur_gray, self.cur_depth = gray, depth_arr

            return self.prev_config,False

        return None, False

    def avg_RT(self, positions):
        #Return weighted average of config matrices
        if (not positions):
            return positions
        if (len(positions) == 1):
            return positions[0][0]
        #Initialize weights = 1/distance for the interpolation
        lastWeight = 1/positions[0][1]
        lastR = positions[0][0][0:3, 0:3]
        lastT = positions[0][0][0:3, 3]

        #Iteratively recompute weighted average
        for i in range(1, len(positions)):
            newWeight = 1/ positions[i][1]
            newT = positions[i][0][0:3, 3]
            newR = positions[i][0][0:3, 0:3]
            #Linearly interpolate on T
            lastT = (lastWeight*lastT +newWeight*newT)/(lastWeight+newWeight)
            #Spherical interpolation on R
            inp = R.from_matrix(np.array([lastR, newR]))
            slerp = Slerp([0,1], inp)
            lastR = slerp([(newWeight)/(lastWeight+newWeight)]).as_matrix()[0]
            #Change weight
            lastWeight += newWeight
        tmp = np.hstack((lastR, lastT))
        ret = np.vstack((tmp, [0,0,0,1]))
        return ret

    def avg_positions(self, positions): #Obtain average position based on distance
        norm = 0
        avg_x, avg_y, avg_z, avg_roll, avg_pitch, avg_yaw= 0, 0, 0, 0, 0, 0
        for x,y,z, roll, pitch, yaw,dist in positions:
            avg_x += x/dist
            avg_y += y/dist
            avg_z += z/dist
            avg_roll += roll/dist
            avg_pitch += pitch/dist
            avg_yaw += yaw/dist
            norm += 1/dist
        #Normalize based on sum of distance inverses and return
        return avg_x/norm, avg_y/norm, avg_z/norm, avg_roll/norm, avg_pitch/norm, avg_yaw/norm

    def keyframe_matching(self, cur_gray, cur_depth):
        """
        Using keyframe matching to find the transformation between self.prev_gray and self.cur_gray.
        :return: the transformation info, rotation and translation
        """
        # using BRIEF to find the feature descriptors
        prev_kp, prev_des, cur_kp, cur_des = self.feature_extraction(cur_gray)

        # feature matching using brute-force and hamming distance
        # prev_gray is the train image and cur_gray is the query image
        matches = self.bf.match(cur_des, prev_des)

        # compute local 3D points
        cur_p3d, prev_p3d = self.compute3D(matches, cur_kp, prev_kp, cur_depth)

        # estimate the rigid transformation
        R, t = self.estimate_rigid_transformation(cur_p3d, prev_p3d)

        return R, t

    def feature_extraction(self, cur_gray):
        """
        Extract the features of self.prev_gray and cur_gray using FAST detector and BRIEF
        :return: key points and descriptors of self.prev_gray and cur_gray
        """

        # find the keypoints with STAR and descriptors of self.prev_gray
        kp1 = self.fast.detect(self.prev_gray, None)
        kp1, des1 = self.brief.compute(self.prev_gray, kp1[:200])

        # find the keypoints with STAR and descriptors of self.cur_gray
        kp2 = self.fast.detect(cur_gray, None)
        kp2, des2 = self.brief.compute(cur_gray, kp2[:200])

        return kp1, des1, kp2, des2

    def compute3D(self, matches, cur_kp, prev_kp, cur_depth):
        """
        compute the local 3d points
        :param matches: the feature matched using BFmatching
        :param prev_kp: the keypoints in self.prev_gray
        :param cur_kp: the keypoints in self.cur_gray
        :return: ndarray of points in 3D
        """
        # keypoints of current image, previous image in homogeneous coordinates  3 x n
        cur_2d = np.asarray([[cur_kp[match.queryIdx].pt[0], cur_kp[match.queryIdx].pt[1], 1] for match in matches]).T
        prev_2d =  np.asarray([[prev_kp[match.trainIdx].pt[0], prev_kp[match.trainIdx].pt[1], 1] for match in matches]).T

        # get the depth info of each point, n x n
        current_depth = np.diag(cur_depth[cur_2d[1].astype(int), cur_2d[0].astype(int)])
        previous_depth = np.diag(self.prev_depth[prev_2d[1].astype(int), prev_2d[0].astype(int)])

        return (self.camera_matrix_inv @ (cur_2d) @ current_depth).T, (self.camera_matrix_inv @ (prev_2d) @ previous_depth).T


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
        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))
        [num_rows, num_cols] = B.shape
        if num_rows != 3:
            raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # subtract mean
        Am = A - np.tile(centroid_A, (1, num_cols))
        Bm = B - np.tile(centroid_B, (1, num_cols))

        # compute covariance matrix
        H = Am * np.transpose(Bm)

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T * U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            # print("det(R) < R, reflection detected!, correcting for it ...\n");
            Vt[2, :] *= -1
            R = Vt.T * U.T

        t = -R * centroid_A + centroid_B

        return R, t

    def estimate_rigid_transformation(self, cur_p3d, prev_p3d, num = 10, delta = 30, inlier_bound = 10):
        """
        find the best R, T between cur_p3d and prev_p3d
        :param cur_p3d: n X 3 matrix of 3d points in the current gray image
        :param prev_p3d: n X 3 matrix of 3d points in the previous gray image
        :param num: max number of iterations
        :param delta: bound used to check the quality of tranformation
        :param accuracy: the minimum number of inliers required
        :return: best rigid transformation R, T
        """

        rotation = None
        translation = None
        assert len(cur_p3d) == len(prev_p3d)

        accuracy = 0
        best_inliers = None
        for i in range(num):
            # create 3 random indices of the correspondence points
            idx = np.random.randint(len(cur_p3d), size=3)

            # make sure that A, B are wrapped with mat
            A = np.mat(prev_p3d[idx].T)
            B = np.mat(cur_p3d[idx].T)

            R, t = self.get_rigid_transformation3d(A, B)
            error = np.linalg.norm(np.matmul(R, prev_p3d.T) + t - cur_p3d.T, axis = 0)

            inliers = error < delta
            acc = np.sum(inliers)

            if acc > accuracy and acc >= inlier_bound:
                accuracy = acc
                best_inliers = inliers
                rotation = R
                translation = t

        if best_inliers is None:
            return None, None
        else:
            idx = np.where(best_inliers > 0)[0]
            rotation, translation = self.get_rigid_transformation3d(np.mat(prev_p3d.T[:, idx]), np.mat(cur_p3d.T[:, idx]))

        return rotation, translation


    def shutdown(self):
        pass
