import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser
import cv2
import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
import tensorflow as tf


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.num_of_episode = 0
        self.choosen_action = 0
        self.goal_pos = [
            np.array([190, 0, -3])
        ]
        self.taken_step = 0
        self.last_distance = 0
        self.reward = 0
        self.image_shape = image_shape
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "velocity": np.zeros(3),
            "prev_position": np.zeros(3),
            "prev_velocity": np.zeros(3),
            "prev_distance": np.zeros(3),
            "distance": np.zeros(3),
        }
        self.least_distance = 1500
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(5)
        self._setup_flight()
        self.image_request = airsim.ImageRequest(
            0, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        # Set home position and velocity
        self.taken_step = 0
        self.num_of_episode += 1
        self.drone.takeoffAsync().join()
        complete_state = self.drone.getMultirotorState()
        quad_state = complete_state.kinematics_estimated.position
        self.state["position"] = complete_state.kinematics_estimated.position
        self.state["velocity"] = complete_state.kinematics_estimated.linear_velocity
        self.state["prev_distance"] = self.state["distance"]
        self.state["prev_position"] = self.state["position"]
        self.state["prev_velocity"] = self.state["velocity"]
        self.least_distance = 1500
        self.last_distance = self.get_distance(quad_state)
        # self.drone.moveToPositionAsync(170, 0, -10, 10).join()
        # time.sleep(5)

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        # # img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img1d = img1d * 3.5 + 30
        img1d[img1d > 255] = 255
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        from PIL import Image
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))
        im_final = np.expand_dims(im_final, axis=0)
        im_final = np.expand_dims(im_final, axis=-1)
        return im_final.reshape((84, 84, 1))

    def get_state(self):
        return self.drone.getMultirotorState().kinematics_estimated

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        while responses is None:
            while responses[0] is None:
                responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()
        self.state["prev_distance"] = self.state["distance"]
        self.state["prev_position"] = self.state["position"]
        self.state["prev_velocity"] = self.state["velocity"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        collision = self.drone.simGetCollisionInfo()
        self.state["collision"] = collision.has_collided
        self.state["distance"] = np.array(
            [np.linalg.norm(self.state["position"].x_val - self.state["prev_position"].x_val),
             np.linalg.norm(self.state["position"].y_val - self.state["prev_position"].y_val),
             np.linalg.norm(self.state["position"].z_val - self.state["prev_position"].z_val)])

        velocity = np.array([self.state["velocity"].x_val, self.state["velocity"].y_val, self.state["velocity"].z_val])
        prev_velocity = np.array(
            [self.state["prev_velocity"].x_val, self.state["prev_velocity"].y_val, self.state["prev_velocity"].z_val])

        velocity = (velocity.reshape((1, 1, 3)))
        prev_velocity = (prev_velocity.reshape((1, 1, 3)))
        distance = (self.state["distance"].reshape((1, 1, 3)))
        prev_distance = (self.state["prev_distance"].reshape((1, 1, 3)))
        return ((image), (velocity), (prev_velocity),
                (distance), (prev_distance))

    def _do_action(self, action):
        self.choosen_action = action
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityZAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            -3,
            1,
        ).join()

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        pts = self.goal_pos
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - pts)
        return dist

    def _compute_reward(self):
        reward = 0
        quad_pt = self.state["position"]
        dist = self.get_distance(quad_pt)
        if dist < self.least_distance:
            self.least_distance = dist
        print(dist)

        if dist - self.least_distance > 100:
            reward = -50
            done = 1

        elif self.state['collision']:
            reward = -100
            done = 1

        else:
            done = 0
            diff = (self.last_distance) - dist
            reward += diff
            self.last_distance = dist

            if (dist < 10):
                reward += 500
                done = 1

            if self.choosen_action == 0:
                reward += 4
            if self.choosen_action == 2:
                reward += -1
            if self.choosen_action == 4:
                reward += -2

            if self.num_of_episode < 250:
                if self.taken_step > 150 and dist >= 150:
                    done = 1

            if 300 < self.num_of_episode < 480:
                if self.taken_step > 250 and dist >= 100:
                    done = 1

            if 500 < self.num_of_episode < 700:
                if self.taken_step > 300 and dist >= 100:
                    done = 1

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        self.taken_step += 1
        return obs, reward, done, self.state

    def reset(self):

        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 3:
            quad_offset = (0, -self.step_length, 0)
        else:
            quad_offset = (0, 0, 0)
        print("choosen action= ", action)
        return quad_offset
