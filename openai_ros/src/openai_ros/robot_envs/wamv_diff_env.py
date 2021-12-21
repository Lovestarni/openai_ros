import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from nav_msgs.msg import Odometry
from openai_ros.openai_ros_common import ROSLauncher
from std_msgs.msg import Float32


class WamvDiffEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all WamvEnv environments.
    """

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new WamvEnv environment.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /wamv/odom: Odometry of the Base of Wamv

        Actuators Topic List:
        * /left_cmd : publish the speed of the left propeller.
        * /right_cmd: publish the speed of the right propeller

        Services List:
        * gazebo/set_model_state for setting the initial pose of the model
        Args:
        """
        rospy.logdebug("Start WamvEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        ROSLauncher(rospackage_name="rl_wamv",
                    launch_file_name="put_wamv_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        from rl_wamv.msg import UsvDrive

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(WamvDiffEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=True,
                                            reset_world_or_sim="WORLD")



        rospy.logdebug("WamvEnv unpause1...")
        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()

        self._check_all_systems_ready()


        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/wamv/odom", Odometry, self._odom_callback)


        self.publishers_array = []

        self._cmd_drive_pub_left = rospy.Publisher('/wamv/thrusters/left_thrust_cmd', Float32, queue_size=1)
        self._cmd_drive_pub_right = rospy.Publisher('/wamv/thrusters/right_thrust_cmd', Float32, queue_size=1)

        self.publishers_array.append(self._cmd_drive_pub_left)
        self.publishers_array.append(self._cmd_drive_pub_right)

        self._check_all_publishers_ready()


        self.gazebo.pauseSim()

        rospy.logdebug("Finished WamvEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------


    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("WamvEnv check_all_systems_ready...")
        self._check_all_sensors_ready()
        rospy.logdebug("END WamvEnv _check_all_systems_ready...")
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        '''
        default setting will only check the wamv/odom
        '''
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        rospy.logdebug("ALL SENSORS READY")


    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /wamv/odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/wamv/odom", Odometry, timeout=1.0)
                rospy.logdebug("Current /wamv/odom READY=>")

            except:
                rospy.logerr("Current /wamv/odom not ready yet, retrying for getting odom")
        return self.odom



    def _odom_callback(self, data):
        self.odom = data


    def _check_all_publishers_ready(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rospy.logdebug("START CHECK ALL SENSORS READY")
        for publisher_object in self.publishers_array:
            self._check_pub_connection(publisher_object)
        rospy.logdebug("ALL SENSORS READY")

    def _check_pub_connection(self, publisher_object):

        rate = rospy.Rate(10)  # 10hz
        while publisher_object.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to publisher_object yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("publisher_object Publisher Connected")

        rospy.logdebug("All Publishers READY")


    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def set_model_state(self, ModelState):
        # We start all the ROS related Services caller
        # can set the inial pose of the robot
        service_name='/gazebo/set_model_state'
        rospy.logdebug("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        rospy.logdebug("Service " + str(service_name) + " exists")
        self.set_model_state = rospy.ServiceProxy(service_name, ModelState)
        rospy.logdebug("Calling service " + str(service_name) + "Success!")

    def set_propellers_speed(self, right_propeller_speed, left_propeller_speed, time_sleep=1.0):
        """
        It will set the speed of each of the two proppelers of wamv.
        """
        i = 0
        rospy.logdebug("left and right propeller speeds are >>"+str(left_propeller_speed)+", "+str(right_propeller_speed))

        self._cmd_drive_pub_right.publish(right_propeller_speed)
        self._cmd_drive_pub_left.publish(left_propeller_speed)


        i += 1
        self.wait_time_for_execute_movement(time_sleep)

    def wait_time_for_execute_movement(self, time_sleep):
        """
        Because this Wamv position is global, we really dont have
        a way to know if its moving in the direction desired, because it would need to evaluate the diference in position and speed on the local reference.
        """
        time.sleep(time_sleep)

    def get_odom(self):
        return self.odom

