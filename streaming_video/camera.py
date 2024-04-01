import cv2
from base_camera import BaseCamera
from tracking import tracker

class Camera(BaseCamera):
    @staticmethod
    def frames():
        camera = cv2.VideoCapture('/home/lucis/Downloads/scene_001/camera_0001/video.mp4')
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            img = tracker(img)


            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()