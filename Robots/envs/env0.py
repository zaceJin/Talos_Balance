import gym
import numpy as np

from Robots.ressources.plane import Plane

ROBOT_LIST = ["talos", "solo"]
NAME_ROBOT = ROBOT_LIST[0] #talos
#NAME_ROBOT = ROBOT_LIST[1] #solo

if NAME_ROBOT=="talos":
    from Robots.ressources.talos import Talos as Robot
    HEIGHT_ROOT = 1.0   # Height of the robot when standing
    TRESHOLD_DEAD = [HEIGHT_ROOT-0.3, 1.3]  # Episode is over if the robot goes lower or higher
    TRESHOLD_DEADXY = [-3, 3]
elif NAME_ROBOT=="solo":
    from Robots.ressources.solo import Solo   as Robot
    HEIGHT_ROOT = 0.23  # Height of the robot when standing
    TRESHOLD_DEAD = [HEIGHT_ROOT-0.08, 0.6] # Episode is over if the robot goes lower or higher
else:
    input("Error, name of the robot not defined ...")

IS_POS_BASE_ONE_DIMENSION = True

class Env0(gym.Env):
    metadata = {'render.modes':['_']} # Not used

    def __init__(self, GUI = True):
        self.REAL_TIME = True and GUI
        self.robot = Robot(class_terrain=Plane, GUI=GUI)
        # State and Action space
        # Read : https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
        #        section : Tips and Tricks when creating a custom environment
        # Observation => All joints(joint Angles + joint Angle Velocity) + Base(base_pos+base_ori+base_lin_vel+base_ang_vel)
        cur_angle, cur_vel = self.robot.getJointsState()
        base_pos, base_ori = self.robot.getBasePosOri()
        base_lin_vel, base_ang_vel = self.robot.getBaseVel()
        self._len_cur_angle, self._len_cur_vel = len(cur_angle), len(cur_vel)
        self._len_base_pos, self._len_base_ori = len(base_pos), len(base_ori)
        if IS_POS_BASE_ONE_DIMENSION: self._len_base_pos = 1
        self._len_base_lin_vel, self._len_base_ang_vel = len(base_lin_vel), len(base_ang_vel)
        self.obs_dim = self._len_cur_angle + self._len_cur_vel + self._len_base_pos  # number of joints * 2
        self.obs_dim += self._len_base_ori + self._len_base_lin_vel + self._len_base_ang_vel  # 11 or 13, depending on IS_POS_BASE_ONE_DIMENSION
        self.observation_space = gym.spaces.box.Box(
            low=-1,
            high=1,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Actions => All joints torques
        self._len_controlled_joint_state = len(self.robot.getControlledJointsState()[0])
        self.action_dim = self._len_controlled_joint_state
        self.action_space = gym.spaces.box.Box(
            low=-1,
            high=1,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        self.bound_base_pos = [[-3, 3], [-3, 3], [0, HEIGHT_ROOT*2]]  # TO MODIFY
        self.bound_base_ori = [[-1, 1]] * 4  # Quaternion
        self.bound_base_lin_vel = [[-1, 1], [-1, 1], [-1, 1]]  # TO MODIFY
        self.bound_base_ang_vel = [[-4 * np.pi, 4 * np.pi]] * 3  # TO MODIFY
        self.counter = 0
        # Reset
        self.reset()
        pass

    def reset(self):
        self.robot.reset()
        self.counter = 0
        obs, obs_normalized = self.getObservation()
        return np.array(obs_normalized)

    def step(self, action):
        # Unnormalized action
        self.counter +=1
        action_unnormalized = self.unnormalizeAction(action.tolist())
        d_torques = action_unnormalized
        # Move Robot
        self.robot.moveRobot(np.array(d_torques), real_time=self.REAL_TIME)
        # Get updated Infor
        obs, obs_normalized = self.getObservation()
        # Get Rewards
        reward = self.getReward()
        # Check terminal condition
        done = self.checkDoneCondition()
        info = {}
        return np.array(obs_normalized), reward, done, info

    # ======================================================================================

    # This function returns an observation and a normalized observation of the robot.
    def getObservation(self):

        '''# Check if these values are correct for action_dim and obs_dim
        print("nb_controlled_joints : ", len(self.robot.getControlledJointsState()[0]))
        print("all_joints : ", len(self.robot.getJointsState()[0]))
        if IS_POS_BASE_ONE_DIMENSION:
            print("obs_dim should be    : all_joints*2 + 11     => ", self.obs_dim)
        else:
            print("obs_dim should be    : all_joints*2 + 13     => ", self.obs_dim)
        print("action_dim should be : nb_controlled_joints  => ", self.action_dim)
        '''


        obs, obs_normalized = [], []
        # Observation => All joints(joint Angles + joint Angle Velocity) + Base(base_pos+base_ori+base_lin_vel+base_ang_vel)
        cur_angle, cur_vol = self.robot.getJointsState()
        base_pos, base_ori = self.robot.getBasePosOri()
        if IS_POS_BASE_ONE_DIMENSION: base_pos = base_pos[2]
        base_lin_vel, base_ang_vel = self.robot.getBaseVel()
        # Current joints angles
        cur_angle_normalized = cur_angle[::]
        for i in range(0, self._len_cur_angle):
            cur_angle_normalized[i] = Env0._rescale(cur_angle_normalized[i], self.robot.joints_bound_pos_all[i], [-1,1])
        obs += cur_angle
        obs_normalized += cur_angle_normalized

        # Current joints volecity
        cur_vol_normalized = cur_vol[::]
        for i in range (0,self._len_cur_vel):
            cur_vol_normalized[i]  = Env0._rescale(cur_vol_normalized[i],self.robot.joints_bound_vel_all[i], [-1,1])
        obs += cur_vol

        obs_normalized += cur_vol_normalized

        # Base_Angle
        if IS_POS_BASE_ONE_DIMENSION:
            base_pos_normalized = Env0._rescale(base_pos, self.bound_base_pos[2], [-1, 1])
            obs += [base_pos]
            obs_normalized += [base_pos_normalized]
        else:
            base_pos_normalized = base_pos[::]
            for j in range(0, self._len_base_pos):
                base_pos_normalized[j] = Env0._rescale(base_pos_normalized[j], self.bound_base_pos[j], [-1, 1])
            obs += base_pos
            obs_normalized += base_pos_normalized

        # Base_Orientation
        base_ori_normalized = base_ori[::]
        for i in range(0, self._len_base_ori):
            base_ori_normalized[i] = Env0._rescale(base_ori_normalized[i], self.bound_base_ori[i], [-1, 1])
        obs += base_ori
        obs_normalized += base_ori_normalized
        # base_lin_vel
        base_lin_vel_normalized = base_lin_vel[::]
        for i in range(0, self._len_base_lin_vel):
            base_lin_vel_normalized[i] = Env0._rescale(base_lin_vel_normalized[i], self.bound_base_lin_vel[i], [-1, 1])
        obs += base_lin_vel
        obs_normalized += base_lin_vel_normalized

        # Base_ang_vel
        base_ang_vel_normalized = base_ang_vel[::]
        for i in range(0, self._len_base_ang_vel):
            base_ang_vel_normalized[i] = Env0._rescale(base_ang_vel_normalized[i], self.bound_base_ang_vel[i], [-1, 1])
        obs += base_ang_vel

        obs_normalized += base_ang_vel_normalized
        return obs, obs_normalized


    def getReward(self):
        reward = 0.
        # Keep the robot standing (fixed base position on Z)
        base_pos, _ = self.robot.getBasePosOri()
        # print("HEIGHT_ROOT: ",HEIGHT_ROOT," and z base: ",base_pos[2])
        reward = 1.0- abs (HEIGHT_ROOT- base_pos[2]) # Positive Rewards
        return reward

    def checkDoneCondition(self):
        done = False
        base_pos, _ = self.robot.getBasePosOri()
        if base_pos[2]<TRESHOLD_DEAD[0] or base_pos[2]>TRESHOLD_DEAD[1]:
            done = True
            print("Episode done, thresholdZ: ", TRESHOLD_DEAD, " and position z: ", base_pos[2])
        elif base_pos[0]<TRESHOLD_DEADXY[0] or base_pos[0]>TRESHOLD_DEADXY[1]:
            done = True
            print("Episode done, thresholdX: ", TRESHOLD_DEADXY, " and position X: ", base_pos[0])
        elif base_pos[1]<TRESHOLD_DEADXY[0] or base_pos[1]>TRESHOLD_DEADXY[1]:
            done = True
            print("Episode done, thresholdY: ", TRESHOLD_DEADXY, " and position Y: ", base_pos[1])
        return done

    def unnormalizeAction(self, action_normalized):
        action = action_normalized[::]
        number_controlled_joints = len(self.robot.controlled_joints)
        # unnormalize torques
        for i in range (0,number_controlled_joints):
            action[i] = Env0._rescale(action[i], [-1,1], self.robot.bounds_torques[i])
        return action

    @staticmethod
    def _rescale(value, input_bounds, output_bounds):
        delta1 = input_bounds[1] - input_bounds[0]
        delta2 = output_bounds[1] - output_bounds[0]
        if delta1 == 0:
            return output_bounds[0] + delta2 / 2.
        else:
            return (delta2 * (value - input_bounds[0]) / delta1) + output_bounds[0]

    @staticmethod
    def _run_test_env():
        env = Env0(GUI=True)
        action = np.array([0.01] * env.action_dim)
        while True:
            obs, reward, done, _ = env.step(action)
            if done:
                input("Press to restart ...")
                env.reset()
        return None

    @staticmethod
    def _run_min_test_talos():
        from Robots.ressources.talos import Talos
        Talos._run_min_torques_test()
        return None

    @staticmethod
    def _run_max_test_talos():
        from Robots.ressources.talos import Talos
        Talos._run_max_torques_test()
        return None

    @staticmethod
    def _run_test_solo():
        from Robots.ressources.solo import Solo
        Solo._run_test()
        return None

    # ======================================= OTHER TESTS

    @staticmethod
    def _run_test_reset_solo():
        from Robots.ressources.solo import Solo
        Solo._run_test_reset()
        return None

    @staticmethod
    def _run_test_joints_solo():
        from Robots.ressources.solo import Solo
        Solo._run_test_joints_limit()
        return None
