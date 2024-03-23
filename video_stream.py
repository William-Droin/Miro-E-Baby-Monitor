#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge

class ObjectDetector:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.frame = None
        rospy.Subscriber('/miro/sensors/camr', Image, self.callback)
        print("Subscriber initialized")

    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        ret, jpeg = cv2.imencode('.jpg', image)
        if ret:
            self.frame = jpeg.tobytes()
            print("Frame updated")

    def generate(self):
        while not rospy.is_shutdown():
            if self.frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.frame + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')


