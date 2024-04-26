"""robot_supervisor_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
# from controller import Robot

# from controller import Supervisor

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
import torch
from PIL import Image

# from PIL import Image

import requests
from transformers import AutoProcessor, AutoModelForCausalLM
import openai

from heapq import nlargest

import time
import os



# https://github.com/aidudezzz/deepbots-tutorials/blob/master/robotSupervisorSchemeTutorial/README.md
class GroundRobot(RobotSupervisorEnv):
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

        self.cam = self.getDevice("camera")
        self.cam.enable(sampling_period)

        self.uav_cam = self.getDevice("uav camera")
        self.uav_cam.enable(sampling_period)

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

    def set_destination(self, coordinate):
        self.destination_coordinate = np.array(coordinate)
        self.solve = False

    def normalizer(self, value, min_value, max_value):
        """
        From REINFORCE
        Performs min-max normalization on the given value.

        Returns:
        - float: Normalized value.
        """
        normalized_value = (value - min_value) / (max_value - min_value)        
        return normalized_value
    
    def get_camera_view(self, camera_name):
        camera = self.getDevice(camera_name)
        image = np.array(camera.getImageArray())

        detected_colors = []
        if camera_name == 'camera':
            camera.recognitionEnable(1)
            objects = camera.getRecognitionObjects()
            if len(objects) >= 1:
                for object in objects:
                    color_ar = object.getColors()
                    red_chan = color_ar[0]
                    green_chan = color_ar[1]
                    blue_chan = color_ar[2]
                    if red_chan==0 and green_chan==0 and blue_chan==1:
                        detected_colors.append('blue')
                    elif red_chan==1 and green_chan==0 and blue_chan==0:
                        detected_colors.append('red')
                    elif red_chan==0 and green_chan==1 and blue_chan==0:
                        detected_colors.append('green')
                    elif red_chan==1 and green_chan==1 and blue_chan==0:
                        detected_colors.append('yellow')
            # camera.recognitionDisable()

        return image, detected_colors
    
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

        rel_orientation = (goal_angle - imu_yaw) 
        if rel_orientation > pi:
            rel_orientation -= 2 * pi
        elif rel_orientation < -1 * pi:
            rel_orientation += 2 * pi
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
            self.left_motor.setPosition(0)
            self.right_motor.setPosition(0)
            self.left_motor.setVelocity(0)
            self.right_motor.setVelocity(0)
            return True
        return False

    def stop(self):
        self.left_motor.setPosition(0)
        self.right_motor.setPosition(0)
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
    
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
        # print("ACTION")
        else: # stay
            self.left_motor.setPosition(0)
            self.right_motor.setPosition(0)
            self.left_motor.setVelocity(0)
            self.right_motor.setVelocity(0)
        if self.consecutive_turn < 2:
            self.consecutive_turn = 2
        if self.consecutive_turn > 10:
            self.consecutive_turn = 10
    

    def start_recording(self, filename='hi.mp4'):
        self.movieStartRecording(filename, 400, 400, None, 95, 30, True)

    def stop_recording(self):
        self.movieStopRecording()



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

env = GroundRobot()
env.reset()

# model = PPO("MlpPolicy", env, verbose=1, n_steps=64)
model = PPO.load("ppo8")

# =================================================================================================================
# Actions
# =================================================================================================================

class Action():
    def __init__(self, coordinate, env, model):
        self.coordinate = coordinate
        self.env = env
        self.model = model
        self.affordance_func = self.model.policy.mlp_extractor.value_net

    def go(self, limit=1000):
        self.env.set_destination(self.coordinate)
        i = 0
        while i < limit:
            obs = self.env.get_observations()
            # print(obs)
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            if self.env.solved():
                print("SOLVED")
                i = limit
                obs, rewards, dones, info = self.env.step(4)
        # self.env.stop()
             
    def get_affordance(self):
        obs = self.env.get_observations()
        affordances = self.affordance_func(torch.tensor(obs))
        affordance = affordances.mean().item()
        return affordance
    
class CameraAction():
    def __init__(self, camera, env):
        self.env = env
        self.camera = camera
    
    def go(self):
        image, colors = self.env.get_camera_view(self.camera)
        return np.array(image), colors
    
    def get_affordance(self):
        return 1
    
class TerminationAction():
    def __init__(self, env, max_val=1):
      self.env = env
      self.max_val = max_val

    def go(self):
        self.env.stop()
   
    def get_affordance(self):
        return -1

# Sample Action Sets

action_set = {
    'go to the red square': Action([-2, -2], env, model),
    'go to the blue square': Action([2, -2], env, model),
    'go to the green square': Action([2, 2], env, model),
    'go to the yellow square': Action([-2, 2], env, model),
}

camera_action_set = {
    'get a description of the robot camera view': CameraAction('camera', env),
    'get a description of an overhead camera view': CameraAction('uav camera', env)
}

termination_action_set = {
    'done': TerminationAction(env)
}

# Experimental Action Sets

base_action_set = {
    'go to the red square': Action([-2, -2], env, model),
    'go to the blue square': Action([2, -2], env, model),
    'go to the green square': Action([2, 2], env, model),
    'go to the yellow square': Action([-2, 2], env, model),
    'done': TerminationAction(env)
}

# Sanity Check

img = None
for action in camera_action_set.keys():
    print(action)
    do_action = camera_action_set[action]
    print(do_action.get_affordance())
    img, _ = do_action.go()
    # plt.imshow(do_action.go())
    # plt.show()

# =================================================================================================================
# GIT CAPTIONING
# =================================================================================================================
cache_dir = "C:/Users/Sean/Projects/Classes/CS8803DLM/MultiSayCan/plan_d/Webots-reinforcement-navigation-master/controllers/PPO/model_cache"
model_dir = cache_dir + "/coco/snapshots/a"
from pathlib import Path

# path = Path(str_path)
model_path = Path(model_dir)

# git_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco", cache_dir=cache_dir, local_files_only=True)
# git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco", cache_dir=cache_dir, local_files_only=True)
git_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
git_model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

questions = [
   "the square colors are ",
   "furniture in the room includes "
]

def generate_caption(image, processor, model, question=""):
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg" #640x480
    # image = Image.open(requests.get(url, stream=True).raw)
    # image 

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # question = "what colors are the squares? red, yellow, green, or blue?"

    input_ids = processor(text=question, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    # generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_caption)
    return generated_caption

generate_caption(img, git_processor, git_model)

# =================================================================================================================
# SayCan
# =================================================================================================================

client = openai.OpenAI(
  base_url="http://localhost:8000/v1", # "http://<Your api-server IP>:port"
  api_key = "sk-no-key-required"
)

overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}


def gpt3_call(engine="itdontmatter", prompt="", max_tokens=128, temperature=0,
              logprobs=1, echo=False):
  full_query = ""
  for p in prompt:
    full_query += p
  id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
  if id in LLM_CACHE.keys():
    print('cache hit, returning')
    response = LLM_CACHE[id]
  else:
    # print(prompt, max_tokens, temperature, logprobs, echo)
    response = {}
    response["choices"] = []
    for p in prompt:
        r = client.completions.create(
                                    model="llama_cpp",
                                    prompt=p,
                                    max_tokens=1,
                                    temperature=temperature,
                                    logprobs=logprobs,
                                    echo=True)
        r_logprobs = {}
        r_logprobs['tokens'] = r.choices[0].logprobs.tokens[:-1]
        r_logprobs['token_logprobs'] = r.choices[0].logprobs.token_logprobs[:-1]
        r_choice = {}
        r_choice["logprobs"] = r_logprobs
        response["choices"].append(r_choice)
    LLM_CACHE[id] = response
  return response

def gpt3_scoring(query, options, engine="itdontmatter", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
  if limit_num_options:
    options = options[:limit_num_options]
  verbose and print("Scoring", len(options), "options")
  gpt3_prompt_options = [query + option for option in options]
  response = gpt3_call(
      engine=engine,
      prompt=gpt3_prompt_options,
      max_tokens=0,
      logprobs=1,
      temperature=0,
      echo=True,)

  scores = {}
  for option, choice in zip(options, response["choices"]):
    tokens = choice["logprobs"]["tokens"]
    token_logprobs = choice["logprobs"]["token_logprobs"]

    total_logprob = 0
    for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
      print_tokens and print(token, token_logprob)
      if option_start is None and not token in option:
        break
      if token == option_start:
        break
      total_logprob += token_logprob
    scores[option] = total_logprob

  for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
    verbose and print(option[1], "\t", option[0])
    if i >= 10:
      break

  return scores, response

# =================================================================================================================

def build_scene_description(image, processor, model, questions):
    scene_description = ""
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    for question in questions:
        input_ids = processor(text=question, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        scene_description += generated_caption + "; "
    return scene_description

def step_to_nlp(step):
  step = step.replace("robot.pick_and_place(", "")
  step = step.replace(")", "")
  pick, place = step.split(", ")
  return "Pick the " + pick + " and place it on the " + place + "."

def normalize_scores(scores):
  max_score = max(scores.values())
  normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
  return normed_scores

def plot_saycan(llm_scores, vfs, combined_scores, task, correct=True, show_top=None, save_path='.fig.png'):
  if show_top:
    top_options = nlargest(show_top, combined_scores, key = combined_scores.get)
    # add a few top llm options in if not already shown
    top_llm_options = nlargest(show_top // 2, llm_scores, key = llm_scores.get)
    for llm_option in top_llm_options:
      if not llm_option in top_options:
        top_options.append(llm_option)
    llm_scores = {option: llm_scores[option] for option in top_options}
    vfs = {option: vfs[option] for option in top_options}
    combined_scores = {option: combined_scores[option] for option in top_options}

  sorted_keys = dict(sorted(combined_scores.items()))
  keys = [key for key in sorted_keys]
  positions = np.arange(len(combined_scores.items()))
  width = 0.3

  fig = plt.figure(figsize=(12, 6))
  ax1 = fig.add_subplot(1,1,1)

  plot_llm_scores = normalize_scores({key: np.exp(llm_scores[key]) for key in sorted_keys})
  plot_llm_scores = np.asarray([plot_llm_scores[key] for key in sorted_keys])
  plot_affordance_scores = np.asarray([vfs[key] for key in sorted_keys])
  plot_combined_scores = np.asarray([combined_scores[key] for key in sorted_keys])

  ax1.bar(positions, plot_combined_scores, 3 * width, alpha=0.6, color="#93CE8E", label="combined")

  score_colors = ["#ea9999ff" for score in plot_affordance_scores]
  ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#ea9999ff", label="vfs")
  ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#a4c2f4ff", label="language")
  ax1.bar(positions - width / 2, np.abs(plot_affordance_scores), width, color=score_colors)

  plt.xticks(rotation="vertical")
  ax1.set_ylim(0.0, 1.0)

  ax1.grid(True, which="both")
  ax1.axis("on")

  ax1_llm = ax1.twinx()
  ax1_llm.bar(positions + width / 2, plot_llm_scores, width, color="#a4c2f4ff", label="language")
  ax1_llm.set_ylim(0.01, 1.0)
  plt.yscale("log")

  font = {"fontname":"Arial", "size":"16", "color":"k" if correct else "r"}
  plt.title(task, **font)
  key_strings = [key.replace("robot.pick_and_place", "").replace(", ", " to ").replace("(", "").replace(")","") for key in keys]
  plt.xticks(positions, key_strings, **font)
  ax1.legend()
#   plt.show()
  plt.savefig(save_path, bbox_inches='tight')


def affordance_scoring(actions, desc=""):
    colors = ["red", "blue", "green", "yellow"]
    affordance_scores = {}
    for action_desc in actions.keys():
        for color in colors:
            if color in action_desc and color in desc:  
                action = actions[action_desc]
                action_aff = action.get_affordance()
                affordance_scores[action_desc] = action_aff
            if color in action_desc and color not in desc:
                action = actions[action_desc]
                action_aff = 0 * action.get_affordance()
                affordance_scores[action_desc] = action_aff
            
    max_val = np.max(np.array(list(affordance_scores.values())))
    if max_val == 0:
       max_val = 1
    for action_desc in actions.keys():
        if action_desc not in affordance_scores.keys():
            if action_desc != 'done':
               affordance_scores[action_desc] = 2 * max_val
    affordance_scores['done'] = .2 * max_val
    for action_desc in affordance_scores.keys():
        affordance_scores[action_desc] *= 10
    return affordance_scores


# query = "To move to the red square, I should:\n"

# options = base_action_set.keys()
# llm_scores, _ = gpt3_scoring(query, options, verbose=True)

# affordance_scores = affordance_scoring(base_action_set)
# # print(affordance_scores)


# combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
# combined_scores = normalize_scores(combined_scores)
# selected_task = max(combined_scores, key=combined_scores.get)
# print("Selecting: ", selected_task)

# =================================================================================================================
# Experimentation
# =================================================================================================================

tasks = [
   "visit four different color squares",
   "find a chair",
   "find a chair near a green square",
#    "find a table near a red square",
   "visit the closest color square"
]

base_action_set = {
    'go to the red square': Action([-2, -2], env, model),
    'go to the blue square': Action([2, -2], env, model),
    'go to the green square': Action([2, 2], env, model),
    'go to the yellow square': Action([-2, 2], env, model),
    'done': TerminationAction(env)
}

vpsc_action_set = {
    'go to the red square': Action([-2, -2], env, model),
    'go to the blue square': Action([2, -2], env, model),
    'go to the green square': Action([2, 2], env, model),
    'go to the yellow square': Action([-2, 2], env, model),
    'get another camera view': CameraAction('uav camera', env),
    'done': TerminationAction(env)
}


class Experiment():
    def __init__(self, name, tasks, action_set):
        self.name = name
        self.tasks = tasks
        self.action_set = action_set
        self.task_idx = 0
        self.action_depth_limit = 9
        self.cam = CameraAction('camera', env)
        self.questions = [
            "furniture in the room includes "
        ]
        self.all_llm_scores = []
        self.all_affordance_scores = []
        self.all_combined_scores = []
        self.prompts = []
        selected_task = ""
        self.steps_text = []
        self.image = None
        self.colors = []
        self.plot = True
        self.uav = False
        self.last_action = None
        self.recording = False

        if not os.path.exists('./'+self.name):
            os.mkdir('./'+self.name)

    def create_prompt(self, task, desc):
        prompt = "You are creating a plan for a robot. "
        prompt += "Your task is " + task + ". \n"
        prompt += "You observe the following: " + desc
        prompt += "You can get another camera perspective if observations are not enough. \n"
        prompt += "Your plan so far looks like: \n"
        for step in self.steps_text:
           prompt += step + "\n"
        prompt += "Your next action is: \n"
        print(prompt)
        return prompt

    def run_task(self, task_idx):
        task_dir = './'+self.name+'/'+str(task_idx)
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        # env.start_recording(filename=task_dir+'/rec.mp4')
        task = self.tasks[task_idx]

        env.reset()
        
        i = 0
        self.image, self.colors = self.cam.go()
        while i < self.action_depth_limit:

            scene_description = build_scene_description(self.image, git_processor, git_model, self.questions)
            if self.uav == False:
                if len(self.colors) > 0:
                    detected_colors = "The following color squares are detected:"
                    for color in self.colors:
                        detected_colors += ' '
                        detected_colors += str(color)
                    detected_colors += ". "
                    scene_description += detected_colors
                else:
                    detected_colors = "No color squares are detected."
                    scene_description += detected_colors
            else:
                scene_description += "The following color squares are detected: red, green, blue, yellow. "


            affordance_scores = affordance_scoring(self.action_set, scene_description)
            
            prompt = self.create_prompt(task, scene_description)
            self.prompts.append(prompt)
            options = list(self.action_set.keys())
            if self.last_action is not None:
                options.remove(self.last_action)

            llm_scores, _ = gpt3_scoring(prompt, options, verbose=True, print_tokens=False)
            combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
            combined_scores = normalize_scores(combined_scores)
            selected_action = max(combined_scores, key=combined_scores.get)
            self.steps_text.append(selected_action)
            print(i, "Selecting: ", selected_action)
            self.last_action = selected_action

            self.all_llm_scores.append(llm_scores)
            self.all_affordance_scores.append(affordance_scores)
            self.all_combined_scores.append(combined_scores)

            if selected_action == 'done':
               print("DONE")
               i += self.action_depth_limit
               break

            env.start_recording(filename=task_dir+'/rec.mp4')
            self.recording = True

            if "camera" in selected_action:
               self.cam = CameraAction('uav camera', env)
               self.image, self.colors = self.cam.go()
               self.uav = True
            else:
               self.cam = CameraAction('camera', env)
               self.action_set[selected_action].go()
               self.image, self.colors = self.cam.go()
               self.uav = False

            i += 1
        if self.recording:
            env.stop_recording()
        if self.plot:
            with open(task_dir+"/summary.txt", "w") as summary_txt:
                j = 0
                for llm_scores, affordance_scores, combined_scores, step in zip(
                    self.all_llm_scores, self.all_affordance_scores, self.all_combined_scores, self.steps_text):
                    plot_saycan(llm_scores, affordance_scores, combined_scores, step, show_top=10, save_path=task_dir+'/plot_'+str(j))
                    
                    summary_txt.write(self.prompts[j])
                    summary_txt.write('\n')
                    summary_txt.write(str(llm_scores))
                    summary_txt.write('\n')
                    summary_txt.write(str(affordance_scores))
                    summary_txt.write('\n')
                    summary_txt.write(str(combined_scores))
                    summary_txt.write('\n')
                    summary_txt.write(str(step))
                    summary_txt.write('\n')
                    summary_txt.write('==============')
                    summary_txt.write('\n')
                    j += 1

    def run_tasks(self):
        for task_idx in range(len(self.tasks)):
            self.all_llm_scores = []
            self.all_affordance_scores = []
            self.all_combined_scores = []
            self.prompts = []
            selected_task = ""
            self.steps_text = []
            self.image = None
            self.colors = []
            self.uav = False
            self.last_action = None
            self.recording = False
            self.run_task(task_idx)
            

    

    
tasks = [
   "visit four different color squares",
#    "find a chair",
   "find a chair near a green square",
   "visit the closest color square"
]

# base_experiment = Experiment('base0', tasks, base_action_set)
# base_experiment.run_tasks()

# vpsc_experiment = Experiment('vpsc0', tasks, vpsc_action_set)
# vpsc_experiment.run_tasks()

# base_experiment = Experiment('base1', tasks, base_action_set)
# base_experiment.run_tasks()

# vpsc_experiment = Experiment('vpsc1', tasks, vpsc_action_set)
# vpsc_experiment.run_tasks()

# base_experiment = Experiment('base2', tasks, base_action_set)
# base_experiment.run_tasks()

# vpsc_experiment = Experiment('vpsc2', tasks, vpsc_action_set)
# vpsc_experiment.run_tasks()

# base_experiment = Experiment('base3', tasks, base_action_set)
# base_experiment.run_tasks()
# base_experiment = Experiment('base4', tasks, base_action_set)
# base_experiment.run_tasks()


# vpsc_experiment = Experiment('vpsc3', tasks, vpsc_action_set)
# vpsc_experiment.run_tasks()
vpsc_experiment = Experiment('vpsc4', tasks, vpsc_action_set)
vpsc_experiment.run_tasks()

# base_experiment = Experiment('base5', tasks, base_action_set)
# base_experiment.run_tasks()
# base_experiment = Experiment('base6', tasks, base_action_set)
# base_experiment.run_tasks()
# base_experiment = Experiment('base7', tasks, base_action_set)
# base_experiment.run_tasks()
# base_experiment = Experiment('base8', tasks, base_action_set)
# base_experiment.run_tasks()
# base_experiment = Experiment('base9', tasks, base_action_set)
# base_experiment.run_tasks()



# vpsc_experiment = Experiment('vpsc5', tasks, vpsc_action_set)
# vpsc_experiment.run_tasks()
# vpsc_experiment = Experiment('vpsc6', tasks, vpsc_action_set)
# vpsc_experiment.run_tasks()
# vpsc_experiment = Experiment('vpsc7', tasks, vpsc_action_set)
# vpsc_experiment.run_tasks()
# vpsc_experiment = Experiment('vpsc8', tasks, vpsc_action_set)
# vpsc_experiment.run_tasks()
# vpsc_experiment = Experiment('vpsc9', tasks, vpsc_action_set)
# vpsc_experiment.run_tasks()

# start = time.time()

# plot_on = True
# # max_tasks = 5


# found_objects = vild(image_path, category_name_string, vild_params, plot_on=False)
# scene_description = build_scene_description(found_objects)
# env_description = scene_description

# print(scene_description)

# gpt3_prompt = gpt3_context
# if use_environment_description:
#   gpt3_prompt += "\n" + env_description
# gpt3_prompt += "\n# " + raw_input + "\n"

# all_llm_scores = []
# all_affordance_scores = []
# all_combined_scores = []
# affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False)
# num_tasks = 0
# selected_task = ""
# steps_text = []
# while not selected_task == termination_string:
#   num_tasks += 1
#   if num_tasks > max_tasks:
#     break

#   llm_scores, _ = gpt3_scoring(gpt3_prompt, options, verbose=True, engine=ENGINE, print_tokens=False)
#   combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
#   combined_scores = normalize_scores(combined_scores)
#   selected_task = max(combined_scores, key=combined_scores.get)
#   steps_text.append(selected_task)
#   print(num_tasks, "Selecting: ", selected_task)
#   gpt3_prompt += selected_task + "\n"

#   all_llm_scores.append(llm_scores)
#   all_affordance_scores.append(affordance_scores)
#   all_combined_scores.append(combined_scores)

# end = time.time()
# print("TIME: ", end - start)

# if plot_on:
#   for llm_scores, affordance_scores, combined_scores, step in zip(
#       all_llm_scores, all_affordance_scores, all_combined_scores, steps_text):
#     plot_saycan(llm_scores, affordance_scores, combined_scores, step, show_top=10)

# print('**** Solution ****')
# print(env_description)
# print('# ' + raw_input)
# for i, step in enumerate(steps_text):
#   if step == '' or step == termination_string:
#     break
#   print('Step ' + str(i) + ': ' + step)
#   nlp_step = step_to_nlp(step)

# if not only_plan:
#   print('Initial state:')
#   plt.imshow(env.get_camera_image())

#   for i, step in enumerate(steps_text):
#     if step == '' or step == termination_string:
#       break
#     nlp_step = step_to_nlp(step)
#     print('GPT-3 says next step:', nlp_step)

#     obs = run_cliport(obs, nlp_step)

#   # Show camera image after task.
#   print('Final state:')
#   plt.imshow(env.get_camera_image())



# new_logger = configure('.', ["stdout", "csv", "tensorboard"])



# for i in range(10):
#     model.learn(total_timesteps=timestep_limit, tb_log_name='PPO')
#     model.save("ppo"+str(i))

# del model # remove to demonstrate saving and loading


# affordance_function = model.policy.mlp_extractor.value_net

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     affordance = affordance_function(torch.tensor(obs)).mean().item()
#     print(affordance)
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