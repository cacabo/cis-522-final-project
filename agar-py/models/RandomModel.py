import numpy as np
from models.ModelInterface import ModelInterface
from actions import Action


class RandomModel(ModelInterface):
    def __init__(self, min_steps, max_steps):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.steps_remaining = 0
        self.curr_action = None

    # RandomModel always moves between min_steps and max_steps (inclusive) steps
    # in randomly selected direction
    def get_action(self, state):
        if self.steps_remaining <= 0:
            self.steps_remaining = np.random.randint(
                self.min_steps, self.max_steps)
            self.curr_action = Action(np.random.randint(8))

        self.steps_remaining -= 1
        return self.curr_action

    # no optimization occurs for RandomModel
    def optimize(self, reward):
        return
