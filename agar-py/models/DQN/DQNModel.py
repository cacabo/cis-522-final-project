from collections import deque
import torch
import numpy as np
import random

# Exploration (this could be moved to the agent instead though)
epsilon = 0.99 
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

batch_size = 32
BUFFER_LENGTH = 1000

class DQNModel:
    def __init__(self):        
        #init replay buffer
        self.replay_buffer = deque(maxlen = BUFFER_LENGTH)

        #init model
        self.model = 
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss = 
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        pass
    
    def get_action(self, state):
        if np.random.random() > epsilon:
            q_values = self.model.predict(state)
            action = np.argmax(q_values) #TODO: placeholder
        else:
            action = #random action TODO: define action space
        return action

    def remember(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

    def train(self): #or experience replay
        if len(self.replay_buffer) < batch_size: #TODO: could toggle batch_size to be diff from minibatch below
            return
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # states, actions, rewards, next_states, dones = list(states), list(actions), list(rewards), list(next_states), list(dones)
        states = torch.Tensor(list(states)).to(self.device)
        actions = torch.Tensor(list(actions)).to(self.device)
        rewards = torch.Tensor(list(rewards)).to(self.device)
        next_states = torch.Tensor(list(next_states)).to(self.device)
        dones = torch.Tensor(list(dones)).to(self.device)

        # do Q computation TODO: understand the equations
        currQ = self.model(states).gather(1, actions) #TODO: understand this
        nextQ = self.model(next_states)
        expectedQ = 
        
        loss = self.loss()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epislon
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)