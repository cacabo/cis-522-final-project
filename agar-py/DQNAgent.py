from collections import deque
import torch
import numpy as np
import random

# Exploration (this could be moved to the agent instead though)
epsilon = 0.99 
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

batch_size = 32

class DQNAgent:
    def __init__(self):
        #init model
        #init replay buffer
        self.replay_buffer = deque(maxlen=1000) #TODO: maxlen macro

        self.model = 
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss = 
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        pass
    
    def get_action(self, state):
        if np.random.random() > epsilon:
            action = # agent picks action based on state
        else:
            action = #random action TODO: define action space
        return action

    def remember(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

    def train(self): #or experience replay
        if len(self.replay_buffer) < batch_size: #TODO: could toggle batch_size to be diff from minibatch below
            return
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = batch

        # do Q computation

        loss = self.loss()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epislon
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)