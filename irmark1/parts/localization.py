import json 
import time
import math

class Localization(object):
    '''
    allow for the car to locate its global positioning
    '''
    def __init__(self, camera_matrix, distortion_coeffs):
        self.json_in = "track/Track_0.json"
        content = open(self.json_in)
        self.map =  json.loads(content.read())
        self.qr_loc = self.map["QR codes"]

        #Calibration for Camera Matrix/Distortion Coefficients
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    
    def __init__(self, camera_matrix, distortion_coeffs, json_in):
        self.json_in = json_in
        content = open(self.json_in)
        self.map =  json.loads(content.read())
        self.qr_loc = self.map["QR codes"]

        #Calibration for Camera Matrix/Distortion Coefficients
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    def get_position(self, qr_id, dist, angle):
        qr_loc = self.qr_loc[qr_id]["Location"]
        line = self.qr_loc[qr_id]["Segment"]

        segment = self.map["Track"]["Lines"][line]
        (s_x, s_y) = segment["Start"]
        (e_x, e_y) = segment["End"]

        x, y = 0, 0
        if abs(e_x - s_x) > abs(e_y - s_y):
            #car is moving in the x direction
            if s_x < e_x:
                x = qr_loc[0] - dist * math.sin(angle)
                y = qr_loc[1] - dist * math.cos(angle)
            else: 
                x = qr_loc[0] + dist * math.sin(angle)
                y = qr_loc[1] - dist * math.cos(angle)
        else: 
            #car is moving in the y direction
            if s_y < e_y:
                x = qr_loc[0] - dist * math.sin(angle)
                y = qr_loc[1] - dist * math.cos(angle)
            else: 
                x = qr_loc[0] - dist * math.sin(angle)
                y = qr_loc[1] + dist * math.cos(angle)

        return x, y, angle
        
    def angle(self, rvec, tvec):
        yaw = 0.1*180*math.atan2(tvec[0][0], tvec[0][1])
        return yaw

    def distance(self,rvec,tvec):
        tvec = tvec[0]
        return math.sqrt(tvec[0]**2 + tvec[1]**2+tvec[2]**2)

    def run_threaded(self, img_arr=None, depth_arr=None):
        if type(img_arr) == np.ndarray:
            if not img_arr.size:
                return 0,0,self.mode,self.recording
        else:
            return 0,0,self.mode,self.recording
        
        #finds all the QR codes seen in the current frame
        img_arr = img_arr.copy()
        valid_ids = False
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        corners, ids, rej = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=parameters)

        if np.all(ids!= None):
            valid_ids = True
            rvecs, tvecs ,_ = aruco.estimatePoseSingleMarkers(corners, 0.2032, self.camera_matrix, self.distortion_coeffs)

            xs, ys, thetas, dists = [], [], [], []
            mintheta = None
            mindist = float('inf')
            for i, val in enumerate(ids):
                curr_time = time.ctime(time.time())
                curr_id = val[0]

                rvec = rvecs[i]
                tvec = tvecs[i]
                dist = self.distance(rvec,tvec)
                difference = self.angle(rvec, tvec)

                x,y,theta = self.get_position(curr_id, dist, difference)
                xs.append(x)
                ys.append(y)
                thetas.append(theta)
                dists.append(dist)
                if dist<mindist:
                    mintheta = theta
                    mindist = dist

            #TODO: If multiple QR codes, take the average relative to distance of those QR codes
            norm, total_x, total_y = 0, 0, 0

            for i, val in enumerate(dists):
                norm += 1/val
                total_x += 1/val * xs[i]
                total_y += 1/val * ys[i]
            avg_x = total_x/norm
            avg_y = total_y/norm

            return avg_x, avg_y, mintheta, True#Spherical Interp
        else:
            return None, None, None, False



    def shutdown(self):
        pass