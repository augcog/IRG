import os
import numpy as np
import irmark1 as m1
import cv2

#11000 for constant

class QRMap(object):
    def __init__(self):
        self.type = {}
        self.object = {}
    def addObject(self, id_, obj):
        id_ = obj.id
        self.object[id_] = obj
        self.type[id_] = type(obj)

class QRObject(object):
    def __init__(self, id_, next_id):
        self.id = id_
        self.next_id = next_id
    
    def run(self):
        return  

class QRTurnObject(QRObject):   
    pass

class QRStraightTurnObject(QRObject):
    pass

class QRStraightCheckObject(QRObject):
    pass

class QRLineFollowerPilot(object):

    def __init__(self, gain, base_throttle, camera_matrix, dist_coeffs):
        self.gain = gain
        self.base_throttle = base_throttle
        self.qrmap = QRMap()
        self.qrmap.addObject(QRStraightCheckObject(9,1))
        self.qrmap.addObject(QRStraightTurnObject(1,7))
        self.qrmap.addObject(QRTurnObject(7,6))
        self.qrmap.addObject(QRStraightTurnbject(6,4))
        self.qrmap.addObject(QRTurnObject(4,2))
        self.qrmap.addObject(QRStraightCheckObject(2,8))
        self.qrmap.addObject(QRStraightTurnObject(8,5))
        self.qrmap.addObject(QRTurnObject(5,0))
        self.qrmap.addObject(QRStraightTurnObject(0,3))
        self.qrmap.addObject(QRTurnObject(3,9))
        self.currentObj = self.qrmap.object[9]
        self.nextId = self.currentObj.next_id

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # checkerboard of size (7 x 6) is used
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        # arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # iterating through all calibration images
        # in the folder
        images = glob.glob('calib_images/*.jpg')

        ovr = None
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ovr = gray
            # find the chess board (calibration pattern) corners
            ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

            # if calibration pattern is found, add object points,
            # image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                # Refine the corners of the detected corners
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ovr.shape[::-1],None,None)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        print("Calibrated Camera")

    def run(self, img_arr, gain = None, base_throttle = None):

        valid_ids = False
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        corners, ids, rej = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if np.all(ids!= None):
            valid_ids = True
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
            for i in ids:
                if id

        #BLOCK OUT SURROUNDINGS/TOP HALF

        for j in range(len(img_arr)):
            for i in range(len(img_arr[0])):
                if i < len(img_arr)//10 or i > 9*len(img_arr)//10 or j < len(img_arr)//2:
                    img_arr[j][i] = [0,0,0]
        
        # HSV Thresholding on off-white/white

        hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
        white_bottom = np.array([0,0,225], dtype=np.uint8)
        white_top = np.array([0,30,255], dtype=np.uint8)

        mask = cv2.inRange(hsv, white_bottom, white_top)
        res = cv2.bitwise_and(img_arr, img_arr, mask=mask)

        if cv2.countNonZero(image) == 0:


        #Get Difference

        M = cv2.moments(res)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        difference = (cX-x_center)//len(img_arr[0])

        #RETURN VALUES (apply gain)

        if (gain):
            retgain = gain*difference
        else:
            retgain = self.gain*difference
        
        if (base_throttle):
            return retgain, base_throttle
            
        return retgain, self.base_throttle