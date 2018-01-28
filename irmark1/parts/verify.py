import time


class LocalizationCheck(object):

    def __init__(self):
        return

    def shutdown(self):
        pass

    def update(self):
        pass

    def run_threaded(self, x, y, angle, qr_detected):
        print("Received: ", x, y, angle, qr_detected, time.time())
