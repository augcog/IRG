import time


class LocalizationCheck(object):

    def __init__(self):
        self.prev_time = time.time()
        return

    def shutdown(self):
        pass

    def update(self):
        pass

    def run_threaded(self, x, y, z, roll, pitch, yaw, ar_detected):
        print("Received: ", x, y, z, roll, pitch, yaw, ar_detected, 1/(time.time()-self.prev_time))
        self.prev_time = time.time()
