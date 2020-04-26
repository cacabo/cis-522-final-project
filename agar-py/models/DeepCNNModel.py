from models.ModelInterface import ModelInterface

import torch
import torch.nn as nn
import torch.optim as optim

class DeepCNNModel(ModelInterface):
    def __init__(self):
        super(DeepCNNModel, self).__init__()
        

    def get_action(self, state):
        """Given the current game state, determine what action the model will output"""
        raise NotImplementedError('Model get_action() is not implemented')

    def optimize(self):
        """Given reward received, optimize the model"""
        raise NotImplementedError('Model optimize() is not implemented')

    def remember(self, state, action, next_state, reward, done):
        """Update replay buffer with what model chose to do"""
        raise NotImplementedError('Model remember() is not implemented')