#!/usr/bin/env python3

# Copied code for node from http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

from __future__ import print_function

import roslib
roslib.load_manifest('enph353_ros_lab')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist


class image_converter:

    def __init__(self):
        # we want to subscribe to the image that is published automatically by the camera
        # then we want to publish the velocity which is automatically heard by the robot
        # self.image_pub = rospy.Publisher("image_topic_2", Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.callback)

        self.publish = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Gets the velocity message from the determineVelocity function
        velocity = self.determineVelocity(cv_image)
        self.publish.publish(velocity)

    # determineVelocity function calculate the velocity for the robot based
    # on the position of the line in the image.   
    def determineVelocity(self, image):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayInverseImage = ~grayImage
        bw = cv2.threshold(grayInverseImage, 147, 255, cv2.THRESH_BINARY)[1]

        h, w = bw.shape[0:2]  # gets dimensions of image

        imageCentre = int(w/2)

        # finds where the line is on the bottom of the image
        left_x = -34  # random numbers that is supposed to be repalce with one when line is found
        right_x = -34
        for x in range(w):
            if (bw[h - 5, x] > 0):
                left_x = x
                break

        for x in range(w):
            if (bw[h - 5, w-x-1] > 0):
                right_x = w-x
                break

        lineCentre = int(left_x+right_x)/2

        lineBufferZone = 7
        straightZoneLeftBoundary = imageCentre - lineBufferZone
        straightZoneRightBoundary = imageCentre + lineBufferZone

        velocity = Twist()

        # goes through different options of turning
        if lineCentre < straightZoneLeftBoundary:
            # turn right Cop
            velocity.linear.x = 0
            velocity.angular.z = 0.3
        elif lineCentre > straightZoneRightBoundary:
            # turn left
            velocity.linear.x = 0
            velocity.angular.z = -0.3
        else:
            # go straight
            velocity.linear.x = 0.3
            velocity.angular.z = 0
        return velocity


# the main function is what is run
# calls on the image_converter class and initializes a node
def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()  # spin() keeps python from exiting until the node is stopped
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
