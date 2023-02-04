import os
import random
import gym
import numpy
import pylab
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from collections import deque

import tensorflow.keras.losses
from tensorflow.keras.models import Model, load_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


import cv2
from PER import *
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# class dataSequence(tf.keras.utils.Sequence):
#
#     def __init__(self, x_set, y_set, batch_size):
#         self.x, self.y = x_set, y_set
#         self.batch_size = batch_size
#
#     def __len__(self):
#         return math.ceil(len(self.x) / self.batch_size)
#
#     def __getitem__(self, idx):
#         batch_x = self.x[idx * self.batch_size:(idx + 1) *
#         self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) *
#         self.batch_size]
#
#         return tuple(batch_x),tuple(batch_y)

def FinalModel(input_shape, action_space, dueling):
    image = layers.Input(input_shape)
    x = layers.Conv2D(32, kernel_size=7 ,  padding="same", activation="elu")(image)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, kernel_size=5,  activation="elu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(16, kernel_size=3,  activation="elu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(8, kernel_size=3, activation="elu")(x)
    x = layers.Flatten()(x)
    velocity_input=layers.Input((1,3))
    prev_velocity_input = layers.Input((1,3))
    distance_input = layers.Input((1,3))
    prev_distance_input = layers.Input((1,3))
    x = layers.Reshape((1,288))(x)
    vel = layers.Reshape((1,3))(velocity_input)
    prev_vel= layers.Reshape((1,3))(prev_velocity_input)
    distance = layers.Reshape((1,3))(distance_input)
    prev_dis = layers.Reshape((1,3))(prev_distance_input)
    dense = layers.Concatenate()([x,vel,prev_vel,distance,prev_dis])
    dense = layers.Dense(256, kernel_initializer='he_uniform',activation=keras.layers.LeakyReLU(alpha=0.2))(dense)
    dense = layers.Dense(256, kernel_initializer='he_uniform',activation=keras.layers.LeakyReLU(alpha=0.2))(dense)
    dense = layers.Dense(256, kernel_initializer='he_uniform', activation=keras.layers.LeakyReLU(alpha=0.2))(dense)

    state_value = layers.Dense(1, kernel_initializer='he_uniform')(dense)
    state_value = layers.Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)
    action_advantage = layers.Dense(action_space, kernel_initializer='he_uniform')(dense)
    action_advantage = layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(
        action_advantage)
    outputs = layers.Add()([state_value, action_advantage])

    model = Model(inputs=[image, velocity_input, prev_velocity_input, distance_input, prev_distance_input],
                  outputs=outputs)
    # model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.compile(optimizer=Adam(lr=0.00025), loss='mean_squared_error')
    # model.compile(optimizer=Adam(lr=0.00005), loss='mean_squared_error')
    model.summary()
    return model
def OurModel(input_shape, action_space, dueling):
    image = layers.Input(input_shape)
    x = layers.Conv2D(32, kernel_size=7 ,  padding="same", activation="elu")(image)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, kernel_size=5,  activation="elu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(16, kernel_size=3,  activation="elu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(8, kernel_size=3, activation="elu")(x)
    x = layers.Flatten()(x)
    velocity_input=layers.Input((1,3))
    prev_velocity_input = layers.Input((1,3))
    distance_input = layers.Input((1,3))
    prev_distance_input = layers.Input((1,3))

    x = layers.Reshape((1,288))(x)
    vel = layers.Reshape((1,3))(velocity_input)
    prev_vel= layers.Reshape((1,3))(prev_velocity_input)
    distance = layers.Reshape((1,3))(distance_input)
    prev_dis = layers.Reshape((1,3))(prev_distance_input)
    dense = layers.Concatenate()([x,vel,prev_vel,distance,prev_dis])
    dense = layers.Dense(256, activation="relu")(dense)
    dense = layers.Dense(256,activation="relu")(dense)

    state_value = layers.Dense(1, kernel_initializer='he_uniform')(dense)
    state_value = layers.Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)
    action_advantage = layers.Dense(action_space, kernel_initializer='he_uniform')(dense)
    action_advantage = layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(
        action_advantage)
    outputs = layers.Add()([state_value, action_advantage])


    model = Model(inputs=[image,velocity_input,prev_velocity_input,distance_input,prev_distance_input], outputs=outputs)
    # model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    # model.compile(optimizer=Adam(lr=0.00025), loss='mean_squared_error')
    model.compile(optimizer=Adam(lr=0.00005), loss='mean_squared_error')
    model.summary()
    return model

class DQNAgent:
    def __init__(self, env):
        self.env_name = str(env)
        self.env = env
        # self.env.seed(0)
        self.action_size = self.env.action_space.n
        self.EPISODES = 1200

        # Instantiate memory
        memory_size = 150000
        # self.memory = deque(maxlen=memory_size)
        self.MEMORY = Memory(memory_size)
        self.gamma = 0.99  # discount rate

        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.02  # minimum exploration probability
        self.epsilon_decay = 0.00009  # exponential decay rate for exploration prob
        self.batch_size = 32

        # defining model parameters
        self.ddqn = True  # use doudle deep q network
        self.dueling = True  # use dealing netowrk
        self.step_warmup = 50
        self.Current_step = 0
        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.scores, self.episodes, self.average = [], [], []

        self.Model_name = os.path.join(self.Save_Path, self.env_name + "_CNN.h5")
        self.Target_model_name = os.path.join(self.Save_Path,self.env_name + "_target_CNN.h5")
        self.ROWS = 84
        self.COLS = 84
        self.update_model_steps = 1000
        self.state_size = (self.ROWS, self.COLS,1)
        self.image_memory = np.zeros(self.state_size)
        self.USE_PER = True
        self.Soft_Update = True
        self.TAU = 0.1
        # create main model and target model
        self.model = FinalModel(input_shape=self.state_size, action_space=self.action_size, dueling=self.dueling)
        self.trained_model = OurModel(input_shape=self.state_size, action_space=self.action_size, dueling=self.dueling)
        self.target_model = FinalModel(input_shape=self.state_size, action_space=self.action_size, dueling=self.dueling)

    #  update the target model
    def update_target_model(self, game_steps):
        if not self.Soft_Update and game_steps % self.update_model_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
            return

        if self.Soft_Update :
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def reset(self):
        frame = self.env.reset()
        return frame

    def remember(self, state, action, reward, next_state, done):
        experience = [state, action, reward, next_state, done]
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append((experience))

    def act(self, state, decay_step):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
        explore_probability = self.epsilon

        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            return random.randrange(self.action_size), explore_probability
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            image, vel, prev_vel, dis , prev_dis = state
            image = np.reshape(image,(1,84,84,1))
            temp_state = [image,vel,prev_vel,dis,prev_dis]
            return np.argmax(self.model.predict([temp_state])), explore_probability

    def replay(self):
        if self.Current_step < self.step_warmup:
            return
        # Randomly sample minibatch from the deque memory
        tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        # else:
        #     if len(self.memory) > self.batch_size:
        #         minibatch = random.sample(self.memory, self.batch_size)
        #     else:
        #         return

        state_size = [84,84,1]
        data_size=[1,3]
        images = np.zeros([self.batch_size] + state_size)
        vels = np.zeros([self.batch_size]+ data_size)
        prev_velocity = np.zeros([self.batch_size] + data_size)
        prev_dis = np.zeros([self.batch_size] + data_size)
        dis = np.zeros([self.batch_size] + data_size)
        actions = np.zeros((self.batch_size),dtype=int)
        rewards = np.zeros((self.batch_size))
        next_images = np.zeros([self.batch_size] + state_size)
        next_vels = np.zeros([self.batch_size] + data_size)
        next_dis = np.zeros([self.batch_size] + data_size)
        prev_velocity_next = np.zeros([self.batch_size] + data_size)
        prev_dis_next = np.zeros([self.batch_size] + data_size)
        done = np.zeros((self.batch_size))

        for i, sample in enumerate(minibatch):
            images[i], vels[i] , prev_velocity[i] , dis[i] , prev_dis[i] = sample[0]
            actions[i] = sample[1]
            rewards[i] = sample[2]
            next_images[i], next_vels[i] , prev_velocity_next[i] , next_dis[i] , prev_dis_next[i] = sample[3]
            done[i] = sample[4]
        states = [images, vels , prev_velocity , dis,prev_dis]
        next_states = [next_images, next_vels , prev_velocity_next , next_dis,prev_dis_next]

        # predict Q-values for starting state using the main network
        target = self.model.predict([states])

        # predict best action in ending state using the main network
        target_next = self.model.predict([next_states])

        target_val = self.target_model.predict([next_states])
        target = np.reshape(target,(self.batch_size,self.action_size))
        target_old = np.array(target)
        target_val = np.reshape(target_val, (self.batch_size, self.action_size))
        target_next = np.reshape(target_next, (self.batch_size, self.action_size))
        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][actions[i]] = rewards[i]
            else:

                # selection of action is from model
                # update is from target model

                # current Q Network selects the action
                # a'_max = argmax_a' Q(s', a')
                a = int(np.argmax(target_next[i]))

                # target Q Network evaluates the action
                # Q_max = Q_target(s', a'_max)
                action_indices=(actions[i])
                target[i][action_indices] = rewards[i] + self.gamma * target_val[i][a]

        # Train the Neural Network with batches
        if self.USE_PER:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, actions]-target[indices, actions])
            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)
        self.model.fit(states,target, verbose=0)


    def load_model(self, model_name,target_name):
        self.model = load_model(model_name)
        self.target_model= load_model(target_name)

    def load_model_for_start(self,name):
        self.trained_model = load_model(name)

    def save(self, model_name,target_name,scores,average,episode):
        self.model.save(model_name)
        self.target_model.save(target_name)
        scores = numpy.array(scores)
        average = numpy.array(average)
        dir=os.path.join(self.Save_Path,str(episode),str(episode))
        np.savez_compressed(dir, score=scores, average=average)



    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        self.Model_name = os.path.join(self.Save_Path,str(episode),"main_model_dqn_dueling_CNN.h5")
        self.Target_model_name = os.path.join(self.Save_Path,str(episode),"target_dqn_dueling_CNN.h5")
        plt.figure(figsize=(18, 9))
        plt.plot(self.episodes, self.average, 'r')
        plt.plot(self.episodes, self.scores, 'b')
        plt.ylabel('Score', fontsize=18)
        plt.xlabel('Games', fontsize=18)
        dqn = 'DQN_'
        dueling = ''
        greedy = ''
        pylab
        dqn = '_DDQN'
        dueling = '_Dueling'
        greedy = '_Greedy'
        plt.savefig(dqn + dueling + greedy + "_CNN.png")

        return self.average[-1]

    def run(self):
        decay_step = 0
        max_average = -300
        self.load_model_for_start("Best_model/main_model_dqn_dueling_CNN.h5")
        counter=0
        # self.target_model.set_weights(target_model_theta)
        for layer in self.trained_model.layers:
            name = (layer.get_config().get("name"))
            if "flatten" in name:
                break
            self.model.layers[counter].set_weights(layer.get_weights())
            counter+=1

        for e in range(self.EPISODES):
            state = self.reset()
            done = False
            score = 0
            SAVING = ''
            while not done:
                decay_step += 1
                self.Current_step = decay_step
                action, explore_probability = self.act(state, decay_step)
                next_state, reward, done, _ = self.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                score += reward

                if done:
                    # every episode, plot the result
                    average = self.PlotModel(score, e)

                    # saving best models
                    if e % 100 == 0:
                        self.save(self.Model_name, self.Target_model_name, self.scores, self.average, e)
                    if average >= max_average:
                        max_average = average
                        self.save(self.Model_name,self.Target_model_name,self.scores,self.average,e)
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    print("episode: {}/{}, score: {}, e: {:.2f}, average: {:.2f} {}".format(e, self.EPISODES, score,
                                                                                            explore_probability,
                                                                                            average, SAVING))
                # update target model
                self.update_target_model(decay_step)
                # # train model
                self.replay()
        self.save(self.Model_name,self.Target_model_name,self.scores,self.average,e)
        # close environemnt when finish training

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def test(self):
        decay_step = 0
        max_average = -1000
        self.epsilon = 0
        self.load_model("Models/355/main_model_dqn_dueling_CNN.h5","Models/900/target_dqn_dueling_CNN.h5")
        for e in range(5):
            state = self.reset()
            done = False
            score = 0
            SAVING = ''
            while not done:
                decay_step += 1
                self.Current_step = decay_step
                action, explore_probability = self.act(state, decay_step)
                next_state, reward, done, _ = self.step(action)
                # self.remember(state, action, reward, next_state, done)
                state = next_state
                score += reward

                # if done:
                #     # every episode, plot the result
                #     average = self.PlotModel(score, e)
                #
                #     # saving best models
                #     if e % 100 == 0:
                #         self.save(self.Model_name, self.Target_model_name, self.scores, self.average, e)
                #     if average >= max_average:
                #         max_average = average
                #         self.save(self.Model_name, self.Target_model_name, self.scores, self.average, e)
                #         SAVING = "SAVING"
                #     else:
                #         SAVING = ""
                #     print("episode: {}/{}, score: {}, e: {:.2f}, average: {:.2f} {}".format(e, self.EPISODES, score,
                #                                                                             explore_probability,
                #                                                                             average, SAVING))
                # # update target model
                # self.update_target_model(decay_step)
                # #
                # # # train model
                # self.replay()

if __name__ == "__main__":
    env = gym.make(
        "airgym:airsim-drone-sample-v0",
        ip_address="127.0.0.1",
        step_length=1,
        image_shape=(84, 84, 1),
    )
    # np.random.seed(1234)
    # random.seed(1234)
    agent = DQNAgent(env)
    # agent.run()
    agent.test()