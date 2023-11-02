
import cv2
import gym
import math

from matplotlib import pyplot as plt
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)
        
        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

        self.bins = []
        self.lastCentroid = 0


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("raw", cv_image)
        cv2.waitKey(3)

        NUM_BINS = 3
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        stateUpper = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        done = False
        
        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #s
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        #split image into 10 even states
        #get bin sizes if not aleady initialised or if image width shape is different than the last 
        if len(self.bins) == 0:
            #print("print once : getingbins")
            self.getBins(cv_image)

        #get centroid in x of road in lower portion of image
        centroid, isRoad = self.getCentroid(cv_image[190:,:,2])
        if centroid == -1:
            centroid = self.lastCentroid
        self.lastCentroid = centroid
        if isRoad==False : self.timeout+=1
        if(self.timeout==30): 
            done = True
            print("30 frames out ")
        
        #indicate state of the road
        for i, endpoint in enumerate(self.bins):
            if(centroid<self.bins[i]):
                state[i]=1
                break

        #repeat process for top view of road
        #get centroid in x 
        centroidUpper, isUpperRoad= self.getUpperCentroid(cv_image[120:190,:,2])
        tempState= stateUpper.copy()
        #indicate state of the road
        for i, endpoint in enumerate(self.bins):
            if(centroidUpper<self.bins[i]):
                stateUpper[i]=1
                break

        if(isUpperRoad==False):
            stateUpper = tempState  
        
        #print("upper state:",stateUpper)
        ##returns the state of the lower road however this function also has the stateUpper which is the state of the upper road
        return state, done

    def getUpperCentroid(self,img):
        #take the red chanel sss
        blueImg= img
        isRoad = True
        cv2.imshow("blue Image U", blueImg)
        cv2.waitKey(1)
        # Using histogram to generate threshold from lowest intensity peak
        hist, edges = np.histogram(blueImg.flatten())
        # plt.bar(edges[:-1], hist, width=np.diff(edges), align='edge')
        # plt.show()
        # print(edges)
        # print(hist)

        #highest_3_bins = np.sort(hist)[:3]
        #lowest_high_bin = np.where(np.isin(hist, highest_3_bins))[0][0]
        #threshold = edges[lowest_high_bin + 1]
        #set manually
        threshold = 100
        # Applying threshold
        blueImg[blueImg > threshold] = 0

        # remove noise
        kernel_size = (3, 3)
        blueImg = cv2.morphologyEx(blueImg, cv2.MORPH_CLOSE, kernel_size)

        ##Check for road
        total = sum(hist)
        index = np.argwhere(edges>=threshold)[0][0]
        roadCount =sum(hist[:index+1])
        
        #print("upper percentage",roadCount/total)
        # print("total: ",total)
        # print(hist)
        # print("thresh", threshold)
        # print("percentage:",(max(hist)+hist[-2])/total)
        # print("percentage max only",max(hist)/total)
        if ( roadCount<.05*total):
            isRoad = False
            #print("no upper road!")



        cv2.imshow("blue ImageU edited", blueImg)
        cv2.waitKey(1)
        if(isRoad):
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(blueImg, connectivity = 4)
            sizes = np.sort(stats[1:, -1])
            #map the road to its correct label
            s = stats[1:, -1]
            size= sizes[-1:]
            idx=(np.argwhere(s==size)[0][0])
            #for each ledtter/num
            road= idx + 1
            xcentroid = centroids[road][0]
        else:
            xcentroid = -1

        return xcentroid, isRoad
    
    def getCentroid(self,img):
        #take the red chanel sss
        blueImg= img
        isRoad = True
        # cv2.imshow("blue Image", blueImg)
        # cv2.waitKey(1)
        # Using histogram to generate threshold from lowest intensity peak
        hist, edges = np.histogram(blueImg.flatten())
        # plt.bar(edges[:-1], hist, width=np.diff(edges), align='edge')
        # plt.show()
        # print(edges)
        # print(hist)

        #highest_3_bins = np.sort(hist)[:3]
        #lowest_high_bin = np.where(np.isin(hist, highest_3_bins))[0][0]
        #threshold = edges[lowest_high_bin + 1]
        #set manually
        threshold = 100
        # Applying threshold
        blueImg[blueImg > threshold] = 0

        # remove noise
        kernel_size = (3, 3)
        blueImg = cv2.morphologyEx(blueImg, cv2.MORPH_CLOSE, kernel_size)

        ##Check for road
        total = sum(hist)
        # print("total: ",total)
        # print(hist)
        # print("thresh", threshold)
        # print("percentage:",(max(hist)+hist[-2])/total)
        # print("percentage max only",max(hist)/total)

        if (max(hist)+ hist[-2] >.90*total):
            isRoad = False
            print("no road!")

        # cv2.imshow("blue Image edited", blueImg)
        # cv2.waitKey(1)
        if(isRoad):
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(blueImg, connectivity = 4)
            sizes = np.sort(stats[1:, -1])
            #map the road to its correct label
            s = stats[1:, -1]
            size= sizes[-1:]
            idx=(np.argwhere(s==size)[0][0])
            #for each ledtter/num
            road= idx + 1
            xcentroid = centroids[road][0]
        else:
            xcentroid = -1
        return xcentroid, isRoad

    
    def getBins(self, img):
        #save bins as array containing right end points of each state
        imgWidth = img.shape[1]
        binWidth = imgWidth/10
        tempBins = []
        for i in range (1,11):
            tempBins.append (i * binWidth)
        self.bins = tempBins
        #print(self.bins)
        return



    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 2
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
