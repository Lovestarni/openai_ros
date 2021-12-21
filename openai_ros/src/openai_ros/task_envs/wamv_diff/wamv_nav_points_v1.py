from openai_ros.robot_envs import wamv_diff_env
import rospy
import numpy as np
from gym import spaces
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from openai_ros.task_envs.wamv_diff.utils import trajectory_preprocess
# import math
import os


class WamvNavPointsEnv(wamv_diff_env.WamvDiffEnv):
    '''
    nav_points_wamv_v1:
    use ppo and modified reward function based on v0
    simulator factor 2
    control rate 5hz
    '''

    def __init__(self):
        """
        Make Wamv learn how to move straight from The starting point
        to a desired point follow with series waypoints.
        Without wind noise.

        * v1 changes: modifiy the reward function from negative reward to positive reward, use for route_0

        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/wamv/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="rl_wamv",
                    launch_file_name="world_1.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/wamv_diff/config",
                               yaml_file_name="wamv_nav_points_v1.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(WamvNavPointsEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here

        rospy.logdebug("Start WamvNavPointEnv INIT...")

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        # Actions and Observations
        self.propeller_high_speed = rospy.get_param(
            '/wamv/propeller_high_speed')
        # the distance for reaching the waypoint
        self.waypoint_distance_threshold = rospy.get_param(
            '/wamv/waypoint_distance_threshold')
        self.propeller_low_speed = rospy.get_param('/wamv/propeller_low_speed')
        self.max_distance_from_waypoint = rospy.get_param(
            '/wamv/max_distance_from_waypoint')
        self.max_velocity_threshold = rospy.get_param(
            'wamv/max_velocity_threshold')
        # Get desired velocity
        self.desired_velocity = rospy.get_param('wamv/desired_velocity')
        # get the target path, set the current waypoint
        self.dec_obs = rospy.get_param("/wamv/number_decimals_precision_obs")
        self.work_space_x_max = rospy.get_param("/wamv/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/wamv/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/wamv/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/wamv/work_space/y_min")
        predifined_trajectory_file_path = rospy.get_param(
            "/wamv/predifined_trajectory")
        self.predifined_trajectory = trajectory_preprocess.read_trajectory_to_points(
            predifined_trajectory_file_path)
        

        # We create the action space, the diff model, speed for double propellers
        # TODO: compare to semi-precision float binary16
        self.action_space = spaces.Box(
            low=np.array([-1*self.propeller_high_speed]*2), high=np.array([self.propeller_high_speed]*2))

        self.current_waypoint_index = 0

        # We place the Maximum and minimum values of observations
        # observation space is 
        #                      heading_error, [-3.14,3.14]
        #                      velocity_error, [-max_volocity_threshold, max_volocity_threshold]
        #                      distance_from_waypoints, [0, 1], current/max
        #                      current_velocity_left, [propeller_high_speed, propeller_high_speed]
        #                      current_velocity_right, [propeller_high_speed, propeller_high_speed]
        #                      current_heading, [-3.14, 3.14]
        #                      error between Last and current, [-3.14,3.14]

        high = np.array([3.14,
                         self.max_velocity_threshold,
                         1,
                         self.max_velocity_threshold,
                         self.max_velocity_threshold,
                         3.14,
                         3.14
                         ])

        low = np.array([-3.14,
                        -1 * self.max_velocity_threshold,
                        0,
                        -1 * self.max_velocity_threshold,
                        -1 * self.max_velocity_threshold,
                        -3.14,
                        -3.14
                        ])

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

        # Rewards

        self.heading_epsilon = rospy.get_param("/wamv/heading_epsilon")
        self.heading_reward = rospy.get_param("/wamv/heading_reward")
        self.velocity_epsilon = rospy.get_param("/wamv/velocity_epsilon")
        self.velocity_reward = rospy.get_param("/wamv/velocity_reward")
        self.distance_epsilon = rospy.get_param("/wamv/distance_epsilon")
        self.other_situation_reward = rospy.get_param(
            "/wamv/other_situation_reward")

        self.cumulated_steps = 0.0

        rospy.logdebug("END WamvNavPointEnv INIT...")

    def _set_init_pose(self):
        """
        Sets the two proppelers speed to 0.0 and waits for the time_sleep
        to allow the action to be executed
        """
        # TODO: 随机初始化位置，利用set_model_state service实现
        right_propeller_speed = 0.0
        left_propeller_speed = 0.0
        self.set_propellers_speed(
            right_propeller_speed, left_propeller_speed, time_sleep=0.1)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For waypoint Index
        # start point is the firt waypoint where is the wamv inital position
        self.current_waypoint_index = 1
        # For Info Purposes
        self.cumulated_reward = 0.0
        # We get the initial pose to mesure the distance from the desired point.
        odom = self.get_odom()
        current_position = Vector3()
        current_position.x = odom.pose.pose.position.x
        current_position.y = odom.pose.pose.position.y
        current_orientation_quat = odom.pose.pose.orientation
        _, _, current_yaw = self.get_orientation_euler(
            current_orientation_quat)
        current_volocity_linear = odom.twist.twist.linear
        current_volocity = (current_volocity_linear.x**2 + current_volocity_linear.y**2)**(1/2)

        # compute the distance between current point and waypoint
        self.previous_distance_from_waypoint = self.get_distance_from_waypoint(
            current_position)

        # compute the LOS angel
        self.previous_LOS_angel_from_waypoint = self.get_LOS_angel_from_waypoint(
            current_position)
        
        # copute the heading error
        self.previous_heading_error = self.previous_LOS_angel_from_waypoint - current_yaw
        
        # compute the velocity error
        self.previous_velocity_error = self.desired_velocity - current_volocity

    def _set_action(self, action):
        """
        It sets the joints of wamv based on the action integer given
        based on the action number given.
        :param action: The action integer that sets what movement to do next.
        """
        # We convert the actions to speed movements to send to the wamv, 不允许负向转动
        left_propeller_speed = (action[0] - (-1)) * (self.propeller_high_speed/(1-(-1)))
        right_propeller_speed = (action[1] - (-1)) * (self.propeller_high_speed/(1-(-1)))
        rospy.loginfo("Original velocity is "+ str(action))
        rospy.loginfo("Start Set Action ==>"+str(left_propeller_speed)+', '+str(right_propeller_speed))

        # We tell wamv the propeller speeds
        # TODO: 这个频率注释是说为了给出计算距离误差和角度误差的时间，运行位置又在pausesim和unpasesim之间，所以应该也反应了实际的控制频率，如果我想加速仿真，这里的时间如果要保持一致，需要改变为和加速频率相同的倍数
        self.set_propellers_speed(right_propeller_speed,
                                  left_propeller_speed,
                                  time_sleep=0.1)

        rospy.loginfo("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have access to, we need to read the
        WamvDiffEnv API DOCS.
        :return: observation
        """
        rospy.logdebug("Start Get Observation ==>")

        odom = self.get_odom()
        self.current_position = odom.pose.pose.position
        current_orientation_quat = odom.pose.pose.orientation
        _, _, current_yaw = self.get_orientation_euler(
            current_orientation_quat)
        current_volocity_linear = odom.twist.twist.linear
        self.current_volocity = (current_volocity_linear.x**2 + current_volocity_linear.y**2)**(1/2)
        
        # base_volocity_angular_yaw = odom.twist.twist.angular.z

        # Compute the relative value of the current position
        self.current_distance_from_waypoint = self.get_distance_from_waypoint(
            self.current_position)    
        self.current_LOS_angel_from_waypoint  = self.get_LOS_angel_from_waypoint(self.current_position)
        self.current_heading_error = self.current_LOS_angel_from_waypoint - current_yaw
        self.current_volocity_error =  self.desired_velocity - self.current_volocity
        # TODO: 检查heading和计算的LOS角是不是在一个坐标系
        observation = []
        observation.append(round(self.current_heading_error, self.dec_obs))
        observation.append(round(self.current_volocity_error, self.dec_obs))
        observation.append(round(self.current_distance_from_waypoint /
                           self.max_distance_from_waypoint, self.dec_obs))
        # TODO: 这里还是尝试使用x y轴分开的速度，而不是综合的速度，需要测试下综合速度
        observation.append(round(current_volocity_linear.x, self.dec_obs))
        observation.append(round(current_volocity_linear.y, self.dec_obs))
        observation.append(round(current_yaw, self.dec_obs))
        observation.append(round(self.previous_heading_error - self.current_heading_error, self.dec_obs))
        # observation.append(round(base_speed_angular_yaw, self.dec_obs))
        rospy.loginfo(observation)
        return np.array(observation)

    def _is_done(self, observations):
        """
        We consider the episode done if:
        1) The wamvs is ouside the workspace
        2) It got to the last waypoint
        """
        # TODO: observation保留住原始gym api格式，先留空
        _ = observations
        # if wamv outside the workspace
        is_inside_corridor = self.is_inside_workspace(self.current_position)
        # reach the waypoint, default is False
        has_reached_des_point = False
        if self.is_in_waypoint_position(
                self.current_position, self.waypoint_distance_threshold):
            if self.current_waypoint_index == (len(self.predifined_trajectory) - 1):
                # reach the destination
                has_reached_des_point = True
            else:
                # not reach the destination
                self.current_waypoint_index += 1

        exceed_maximum_distance = self.current_distance_from_waypoint > self.max_distance_from_waypoint

        # determine if the episode is done
        done = not(is_inside_corridor) or has_reached_des_point or exceed_maximum_distance

        return done

    def _compute_reward(self, observations, done):
        """
        We Base the rewards in if its done or not and we base it on
        if the distance to the desired point has increased or not
        :return:
        """

        # We only consider the plane, the fluctuation in z is due mainly to wave
        # TODO: 当前没有加入距离奖励，只有速度和角度奖励，后面可能要尝试加入距离奖励
        # If there has been a decrease in the distance to the desired point, we reward it
        rospy.loginfo(self.current_heading_error)
        rospy.loginfo('current heading error: ' + str(round(self.current_heading_error, self.dec_obs)))
        rospy.loginfo('current volocity error: ' + str(round(self.current_volocity_error, self.dec_obs)))
        rospy.loginfo('current distance from waypoint: ' + str(round(self.current_distance_from_waypoint, self.dec_obs)))
        rospy.loginfo('current LOS angel from waypoint: ' + str(round(self.current_LOS_angel_from_waypoint, self.dec_obs)))

        heading_weight = 1
        distance_weight = 1
        velocity_weight = 0.5
        
        # 1. time punishment
        reward = -0.01

        # 2. heading reward
        heading_reward = 1 if (abs(self.current_heading_error) <= self.heading_epsilon) or abs(self.current_heading_error) < abs(self.previous_heading_error) else 0
        # # 3. velocity reward
        # velocity_reward = 0
        velocity_reward = 1 if (abs(self.current_volocity - self.desired_velocity) <= self.velocity_epsilon) else 0
        # 4. distance reward, we want to minimize the distance to the desired point
        # 如果距离waypoint的距离增大，则被惩罚
        if done:
            distance_reward = 0
        else:
            distance_reward = -1 if (self.current_distance_from_waypoint - self.previous_distance_from_waypoint) >= self.distance_epsilon else 0
        
        reward += heading_weight * heading_reward + distance_weight * distance_reward + velocity_weight * velocity_reward

        # TODO: 如果达成完成条件，惩罚失败，奖励成功

        rospy.loginfo("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.loginfo("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.loginfo("Cumulated_steps=" + str(self.cumulated_steps))

        # if not done, save the current infomation
        if not done:  
            self.previous_distance_from_waypoint = self.current_distance_from_waypoint
            self.previous_LOS_angel_from_waypoint = self.current_LOS_angel_from_waypoint
            self.previous_heading_error = self.current_heading_error
            self.previous_velocity_error = self.current_volocity_error

        return reward

    # Internal TaskEnv Methods
    def get_waypoint_position(self):
        assert 0 <= self.current_waypoint_index < len(self.predifined_trajectory)
        return self.predifined_trajectory[self.current_waypoint_index]

    def is_in_waypoint_position(self, current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the waypoint poistion
        """

        is_in_waypoint_pos = False

        current_waypoint_position = self.get_waypoint_position()
        x_pos_plus = current_waypoint_position.x + epsilon
        x_pos_minus = current_waypoint_position.x - epsilon
        y_pos_plus = current_waypoint_position.y + epsilon
        y_pos_minus = current_waypoint_position.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (
            x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (
            y_current > y_pos_minus)

        is_in_waypoint_pos = x_pos_are_close and y_pos_are_close

        rospy.logdebug("###### IS waypoint POS ? ######")
        rospy.logdebug("current_position"+str(current_position))
        rospy.logdebug("x_pos_plus"+str(x_pos_plus) +
                       ",x_pos_minus="+str(x_pos_minus))
        rospy.logdebug("y_pos_plus"+str(y_pos_plus) +
                       ",y_pos_minus="+str(y_pos_minus))
        rospy.logdebug("x_pos_are_close"+str(x_pos_are_close))
        rospy.logdebug("y_pos_are_close"+str(y_pos_are_close))
        rospy.logdebug("is_in_waypoint_pos"+str(is_in_waypoint_pos))
        rospy.logdebug("############")

        return is_in_waypoint_pos

    def get_distance_from_waypoint(self, current_position):
        """
        Calculates the distance from the current position to the waypoint
        :param start_point:
        :return:
        """

        distance = self.get_distance_from_point(current_position,
                                                self.get_waypoint_position())

        return distance

    def get_distance_from_point(self, p_start, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((p_start.x, p_start.y, p_start.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        distance = np.linalg.norm(a - b)

        return distance

    def get_LOS_angel_from_waypoint(self, current_position):
        """
        Calculates the Line of sight angel from the current position to the waypoint
        :param start_point:
        :return:
        """
        los_angel = self.get_LOS_angel_from_point(current_position,
                                                  self.get_waypoint_position())

        return los_angel

    def get_LOS_angel_from_point(self, p_start, p_end):
        """
        Given a Vector3 Object, get Line of sight angel from current position
        :param p_end:
        :return:
        """
        a = np.array((p_start.x, p_start.y, p_start.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        los_angel = np.arctan2(b[1]-a[1], b[0]-a[0])

        return los_angel

    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

    def is_inside_workspace(self, current_position):
        """
        Check if the Wamv is inside the Workspace defined
        """
        is_inside = False

        # rospy.logdebug("##### INSIDE WORK SPACE? #######")
        rospy.loginfo("current waypoint \n"+str(self.get_waypoint_position()))
        rospy.loginfo("XYZ current_position \n"+str(current_position))
        # rospy.logdebug("work_space_x_max"+str(self.work_space_x_max) +
                    #   ",work_space_x_min="+str(self.work_space_x_min))
        # rospy.logdebug("work_space_y_max"+str(self.work_space_y_max) +
                    #   ",work_space_y_min="+str(self.work_space_y_min))
        # rospy.logdebug("############")

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                is_inside = True

        return is_inside
