from models.DeepCNNModel import DeepCNNModel
from gamestate import GameState
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

from trainutil import select_model_actions

def train_deepcnn_model(adversary_models):
    env = GameState()
    model = DeepCNNModel(camera_follow=True)        # ensure that this agent is at center of board
    models = [model] + adversary_models

    env.reset(models)
    state = env.get_state()
    pixels = env.get_pixels()
    # plt.imshow(pixels)
    # plt.show()

    training = True
    while training:
        # append current pixel state to buffer, then stack last tau frames to
        # get CNN action based on them
        model.state_buffer.append(model.preprocess_state(pixels))
        s_0 = np.stack([model.state_buffer])
        action = model.get_action(s_0)
        model.step_count += 1
        print(model.step_count)

        # get actions for other models
        actions = select_model_actions(adversary_models, state)

        # determine rewards and dones by letting all models take actions
        rewards, dones = env.update_game_state(models, [action] + actions)

        # get the next pixel state and append to next state buffer
        next_state = env.get_state()
        next_pixels = env.get_pixels()
        model.next_state_buffer.append(model.preprocess_state(next_pixels))

        # push to replay buffer
        model.remember(pixels, action, next_pixels, rewards[0], dones[0])

        # optimize model
        model.optimize()

        if dones[0]:
            break

        state = next_state

train_deepcnn_model([])