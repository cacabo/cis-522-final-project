from models.ModelInterface import ModelInterface
from model_utils.ReplayBuffer import ReplayBuffer
from actions import Action

from collections import deque
from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from skimage import transform

import matplotlib.pyplot as plt

REPLAY_BUF_CAPACITY = 10000
BATCH_SIZE = 64
DOWNSAMPLE_SIZE = (100, 100)

# CNN which takes in the game state as TODO and returns Q-values for each possible action
class CNN(nn.Module):
    def __init__(self, tau, input_dim, output_dim):
        super(CNN, self).__init__()
        self.tau = tau

        self.convnet = nn.Sequential(
            nn.Conv2d(self.tau, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_input_dim = self.calc_cnn_out_dim(input_dim)

        self.fcnet = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.reshape(-1, self.fc_input_dim)
        return self.fcnet(x)

    def calc_cnn_out_dim(self, input_dim):
        return self.convnet(torch.zeros(1, self.tau, *input_dim)).flatten().shape[0]

# tau is the number of frames to stack to create one "state"
class DeepCNNModel(ModelInterface):
    def __init__(self, camera_follow, tau=4, epsilon=0.0):
        super(DeepCNNModel, self).__init__()
        self.camera_follow = camera_follow
        self.tau = tau
        self.replay_buffer = ReplayBuffer(REPLAY_BUF_CAPACITY)
        self.device = 'cpu'
        self.step_count = 0

        # initialize tau-sized frame buffers with zeros
        self.state_buffer = deque(maxlen=tau)
        self.next_state_buffer = deque(maxlen=tau)
        for i in range(self.tau):
            self.state_buffer.append(np.zeros(DOWNSAMPLE_SIZE))
            self.next_state_buffer.append(np.zeros(DOWNSAMPLE_SIZE))

        self.net = CNN(tau, DOWNSAMPLE_SIZE, len(Action))

        self.epsilon = epsilon

    def get_action(self, stacked_state):
        """Given the current game state, determine what action the model will output"""
        # take a random action epsilon fraction of the time
        if random.random() < self.epsilon:
            return Action(np.random.randint(len(Action)))
        # otherwise, take the action which maximizes expected reward
        else:
            q_values = self.net(torch.FloatTensor(stacked_state).to(self.device))
            action_idx = torch.argmax(q_values).item()
            return Action(action_idx)

    def optimize(self):
        """Given reward received, optimize the model"""
        # wait for a full training batch before doing any optimizing
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        self.optimizer.zero_grad()
        batch = self.replay_buffer.sample(BATCH_SIZE)
        loss = self.calculate_loss(batch)

    def remember(self, state, action, next_state, reward, done):
        """Update replay buffer with what model chose to do"""
        # stack last tau states buffers into single array
        stacked_state = np.stack([self.state_buffer])
        stacked_next_state = np.stack([self.next_state_buffer])

        # push to memory
        self.replay_buffer.push((deepcopy(stacked_state), action,
                                 deepcopy(stacked_next_state), reward, done))

    def calculate_loss(self, batch):
        states, actions, next_states, rewards, dones = zip(*batch)

        actions_t = torch.tensor(actions).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        dones_t = torch.tensor(dones).to(self.device)

        states_t = torch.FloatTensor(states).to(self.device)
        qvals = self.net(states_t).gather(1, actions_t).squeeze()
        print(qvals)
        raise ValueError


    def preprocess_state(self, state):
        # convert RGB to grayscale via relative luminance
        gray_state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
        # size down the image to speed up training
        resized_state = transform.resize(gray_state, DOWNSAMPLE_SIZE)
        return resized_state