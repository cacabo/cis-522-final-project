import numpy as np
import ModelInterface

class RandomModel(ModelInterface):
    def __init__(self, min_steps, max_steps):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.steps_remaining = 0
        self.angle = 0

    # RandomModel always moves between min_steps and max_steps (inclusive) steps
    # in randomly selected direction
    def get_action(self, state):
        if self.steps_remaining <= 0:
            self.steps_remaining = np.random.randint(min_steps, max_steps)
            self.angle = ModelInterface.angles[np.random.randint(8)]
        
        self.steps_remaining -= 1
        return self.angle
    
    # no optimization occurs for RandomModel
    def optimize(self, reward):
        return