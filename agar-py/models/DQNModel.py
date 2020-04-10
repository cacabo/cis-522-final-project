from collections import deque
import torch
import numpy as np
import random
import torch
import torch.nn as nn
from models.ModelInterface import ModelInterface
from actions import Action

# Exploration (this could be moved to the agent instead though)
EPSILON = 0.95
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

GAMMA = 0.99

batch_size = 32
BUFFER_LENGTH = 1000


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        qvals = self.fc3(x)
        return qvals

class DQNModel(ModelInterface):
    def __init__(self):
        # init replay buffer
        self.replay_buffer = deque(maxlen=BUFFER_LENGTH)

        # init model
        self.model = DQN(observation_space.shape, len(Action))  # TODO: fix w/ observation space
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss = torch.nn.SmoothL1Loss()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.epsilon = EPSILON
        self.gamma = GAMMA

    def get_action(self, state):
        if self.done:
            return None
        if random.random() > self.epsilon:
            q_values = self.model.predict(state)
            action = np.argmax(q_values)  # TODO: placeholder
        else:
            action = Action(np.random.randint(len(Action))) # random action
        return action

    def remember(self, state, action, next_state, reward, done):
        if self.done:
            return
        else:
            self.replay_buffer.append((state, action, next_state, reward, done))
            self.done = done

    def optimize(self):  # or experience replay
        if self.done:
            return
        # TODO: could toggle batch_size to be diff from minibatch below
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # states, actions, rewards, next_states, dones = list(states), list(actions), list(rewards), list(next_states), list(dones)
        states = torch.Tensor(list(states)).to(self.device)
        actions = torch.Tensor(list(actions)).to(self.device)
        rewards = torch.Tensor(list(rewards)).to(self.device)
        next_states = torch.Tensor(list(next_states)).to(self.device)
        # what about these/considering terminating states
        dones = torch.Tensor(list(dones)).to(self.device)

        # do Q computation TODO: understand the equations
        currQ = self.model(states).gather(1, actions)  # TODO: understand this
        nextQ = self.model(next_states)
        max_nextQ = torch.max(nextQ, 1)[0]
        expectedQ = rewards + self.gamma * max_nextQ

        loss = self.loss(currQ, expectedQ)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epislon
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(MIN_EPSILON, self.epsilon)
