"""robot_supervisor_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
# from controller import Robot

from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
# from PPO_agent import PPOAgent, Transition

import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure, Figure
from stable_baselines3.common.callbacks import BaseCallback

from gym.spaces import Box, Discrete
import numpy as np

from math import atan2, pi

# https://github.com/aidudezzz/deepbots-tutorials/blob/master/robotSupervisorSchemeTutorial/README.md
class CartpoleRobot(RobotSupervisorEnv):
    def __init__(self):
        super().__init__(timestep=500)


        
        # FROM REINFORCE
        # General environment parameters
        self.max_speed = 1.5 # Maximum Angular speed in rad/s
        self.destination_coordinate = np.array([2.45, 0]) # Target (Goal) position
        self.reach_threshold = 0.06 # Distance threshold for considering the destination reached.
        obstacle_threshold = 0.1 # Threshold for considering proximity to obstacles.
        self.obstacle_threshold = 1 - obstacle_threshold
        self.floor_size = np.linalg.norm([8, 8])

        # Reference robot node, init sensor, get reference for endpoint, init motors
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        # self.position_sensor = self.getDevice("polePosSensor")
        # self.position_sensor.enable(self.timestep)
        # self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")



        # Activate Devices
        #~~ 1) Wheel Sensors
        self.left_motor = self.getDevice('left wheel')
        self.right_motor = self.getDevice('right wheel')

        #~~ 2) GPS Sensor
        sampling_period = 1 # in ms
        self.gps = self.getDevice("gps")
        self.gps.enable(sampling_period)
        
        #~~ 3) Enable Touch Sensor
        self.touch = self.getDevice("touch sensor")
        self.touch.enable(sampling_period)

        #~~ 4) Enable Intertial Unit
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(sampling_period)

        # List of all available sensors
        available_devices = list(self.devices.keys())
        
        
        # Filter sensors name that contain 'so'
        filtered_list = [item for item in available_devices if 'so' in item and any(char.isdigit() for char in item)]
        filtered_list = sorted(filtered_list, key=lambda x: int(''.join(filter(str.isdigit, x))))


        # Reset
        # self.simulationReset()
        # self.simulationResetPhysics()
        # super(Supervisor, self).step(int(self.getBasicTimeStep()))
        # self.robot.step(200) # take some dummy steps in environment for initialization
        

        # Create dictionary from all available distance sensors and keep min and max of from total values
        self.max_sensor = 0
        self.min_sensor = 0
        self.dist_sensors = {}
        for i in filtered_list:    
            self.dist_sensors[i] = self.getDevice(i)
            self.dist_sensors[i].enable(sampling_period)
            self.max_sensor = max(self.dist_sensors[i].max_value, self.max_sensor)    
            self.min_sensor = min(self.dist_sensors[i].min_value, self.min_sensor)
        # self.wheels = []
        # for wheel_name in ['left wheel', 'right wheel']:
        #     wheel = self.getDevice(wheel_name)  # Get the wheel handle
        #     wheel.setPosition(float('inf'))  # Set starting position
        #     wheel.setVelocity(0.0)  # Zero out starting velocity
        #     self.wheels.append(wheel)

        # Variables for training
        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved


        # Flags
        self.done = False
        self.solve = False

        # Consecutive turns are BAD
        self.consecutive_turn = 0


        # Inputs and Outputs of NN
        self.observation_space = Box(low=np.array([0, self.min_sensor, self.min_sensor, 0]),
                                     high=np.array([1, self.max_sensor, self.max_sensor, 1]),
                                     dtype=np.float64)
        self.action_space = Discrete(3)

    def normalizer(self, value, min_value, max_value):
        """
        From REINFORCE
        Performs min-max normalization on the given value.

        Returns:
        - float: Normalized value.
        """
        normalized_value = (value - min_value) / (max_value - min_value)        
        return normalized_value
    
    def get_distance_to_goal(self):
        """
        From REINFORCE
        Calculates and returns the normalized distance from the robot's current position to the goal.
        
        Returns:
        - numpy.ndarray: Normalized distance vector.
        """
        
        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)
        distance_to_goal = np.linalg.norm(self.destination_coordinate - current_coordinate)
        normalizied_coordinate_vector = self.normalizer(distance_to_goal, min_value=0, max_value=self.floor_size)
        
        return normalizied_coordinate_vector
    
    def get_sensor_data(self):
        """
        Retrieves and normalizes data from distance sensors.
        
        Returns:
        - numpy.ndarray: Normalized distance sensor data.
        """
        
        # Gather values of distance sensors.
        sensor_data = []
        for z in self.dist_sensors:
            sensor_data.append(self.dist_sensors[z].value)  
            
        sensor_data = np.array(sensor_data)
        normalized_sensor_data = self.normalizer(sensor_data, self.min_sensor, self.max_sensor)
        
        return normalized_sensor_data
    
    def get_heading_to_goal(self):
        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)

        goal_diff = self.destination_coordinate - current_coordinate
        goal_angle = atan2(goal_diff[1], goal_diff[0])

        imu_yaw = self.imu.getRollPitchYaw()[2]

        rel_orientation = goal_angle - imu_yaw
        # print(rel_orientation)
        normalizied_orient_vector = self.normalizer(rel_orientation, min_value=(-1 * pi), max_value=pi)
        return normalizied_orient_vector
        # return normalizied_coordinate_vector

    def get_observations(self):
        # # Position on x-axis
        # cart_position = normalize_to_range(self.robot.getPosition()[0], -0.4, 10.4, -1.0, 1.0)
        # # Linear velocity on x-axis
        # cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
        # # Pole angle off vertical
        # # pole_angle = normalize_to_range(self.position_sensor.getValue(), -0.23, 0.23, -1.0, 1.0, clip=True)
        # # Angular velocity y of endpoint
        # endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)

        # return [cart_position, cart_velocity, pole_angle, endpoint_velocity]
        # FROM REINFORCE
        normalized_sensor_data = np.array(self.get_sensor_data(), dtype=np.float32)
        normalized_current_coordinate = np.array([self.get_distance_to_goal()], dtype=np.float32)
        normalized_heading = np.array([self.get_heading_to_goal()], dtype=np.float32)
        state_vector = np.concatenate([normalized_current_coordinate, normalized_sensor_data, normalized_heading], dtype=np.float32)
        
        return state_vector

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        
        # return 1
        # FROM REINFORCE
        reward = 0
        
        normalized_sensor_data = self.get_sensor_data()
        normalized_current_distance = self.get_distance_to_goal()
        normalized_current_orientation = self.get_heading_to_goal()
        
        normalized_current_distance *= 100 # The value is between 0 and 1. Multiply by 100 will make the function work better
        reach_threshold = self.reach_threshold * 100
        
        # (1) Reward according to distance 
        if normalized_current_distance < 42:
            if normalized_current_distance < 10:
                growth_factor = 5
                A = 2.5
            elif normalized_current_distance < 25:
                growth_factor = 4
                A = 1.5
            elif normalized_current_distance < 37:
                growth_factor = 2.5
                A = 1.2
            else:
                growth_factor = 1.2
                A = 0.9
            reward += A * (1 - np.exp(-growth_factor * (1 / normalized_current_distance)))
        else: 
            reward += -normalized_current_distance / 100

        # normalized_current_orientation *= 100
        # normalized_current_orientation -= 50

        # if abs(normalized_current_orientation) > 25:
        #     reward -= .0001
            

        # (2) Reward or punishment based on failure or completion of task
        check_collision = self.touch.value
        if normalized_current_distance < reach_threshold:
            # Reward for finishing the task
            self.done = True
            self.solve = True
            reward += 25
            print('+++ SOlVED +++')
        elif check_collision:
            # Punish if Collision
            self.done = True
            reward -= 5
            
            
        # (3) Punish if close to obstacles
        # elif np.any(normalized_sensor_data[normalized_sensor_data > self.obstacle_threshold]):
        #     reward -= 0.0001
        
        # # 4 PUNISH CONSECUTIVE TURNS
        # reward -= .0001 * (self.consecutive_turn - 2)

        return reward

    def is_done(self):
        if self.done:
            self.done = False
            return True
        # if self.episode_score > 195.0:
        #     return True

        # pole_angle = round(self.position_sensor.getValue(), 2)
        # if abs(pole_angle) > 0.261799388:  # more than 15 degrees off vertical (defined in radians)
        #     return True

        # cart_position = round(self.robot.getPosition()[0], 2)  # Position on x-axis
        # if abs(cart_position) > 0.39:
        #     return True

        return False
    
    def solved(self):
        # if len(self.episode_score_list) > 100:  # Over 100 trials thus far
        #     if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
        if self.solve:
            self.solve = False
            return True
        return False
    
    def get_info(self):
        info = {
            "TimeLimit.truncated": False,
        }
        return info

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        """
        From REINFORCE
        Applies the specified action to the robot's motors.
        
        Returns:
        - None
        """
        # print(action)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        if action == 0: # move forward
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
            self.consecutive_turn -= 1
        elif action == 1: # turn right
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(-self.max_speed)
            self.consecutive_turn += 1
        elif action == 2: # turn left
            self.left_motor.setVelocity(-self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
            self.consecutive_turn += 1
        # elif action == 4: # stay
        #     self.left_motor.setVelocity(0)
        #     self.right_motor.setVelocity(0)
        if self.consecutive_turn < 2:
            self.consecutive_turn = 2
        if self.consecutive_turn > 10:
            self.consecutive_turn = 10


class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Plot values (here a random variable)
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True

env = CartpoleRobot()
# agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n)

timestep_limit = 5000

# new_logger = configure('.', ["stdout", "csv", "tensorboard"])

model = PPO("MlpPolicy", env, verbose=1, n_steps=64, tensorboard_log="./tlog/")
# model.set_logger(new_logger)

# model.load("ppo_cartpole")

# for i in range(10):
#     model.learn(total_timesteps=timestep_limit, tb_log_name='PPO')
#     model.save("ppo"+str(i))

# del model # remove to demonstrate saving and loading

model = PPO.load("ppo8")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print("DONE")
# solved = False

# episode_count = 0
# episode_limit = 2000

# # Run outer loop until the episodes limit is reached or the task is solved
# while not solved and episode_count < episode_limit:
#     observation = env.reset()  # Reset robot and get starting observation
#     env.episode_score = 0
#     for step in range(env.steps_per_episode):
#         # In training mode the agent samples from the probability distribution, naturally implementing exploration
#         selected_action, action_prob = agent.work(observation, type_="selectAction")
       
#         # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
#         # the done condition
#         new_observation, reward, done, info = env.step([selected_action])

#         # Save the current state transition in agent's memory
#         trans = Transition(observation, selected_action, action_prob, reward, new_observation)
#         agent.store_transition(trans)

#         if done:
#             # Save the episode's score
#             env.episode_score_list.append(env.episode_score)
#             agent.train_step(batch_size=step + 1)
#             solved = env.solved()  # Check whether the task is solved
#             break

#         env.episode_score += reward  # Accumulate episode reward
#         observation = new_observation  # observation for next step is current step's new_observation

#     print("Episode #", episode_count, "score:", env.episode_score)
#     episode_count += 1  # Increment episode counter

# if not solved:
#     print("Task is not solved, deploying agent for testing...")
# elif solved:
#     print("Task is solved, deploying agent for testing...")

# observation = env.reset()
# env.episode_score = 0.0
# while True:
#     selected_action, action_prob = agent.work(observation, type_="selectActionMax")
#     observation, _, done, _ = env.step([selected_action])
#     if done:
#         observation = env.reset()