from models.DeepCNNModel import DeepCNNModel
from gamestate import GameState
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

from trainutil import select_model_actions

FRAME_SKIP = 4
UPDATE_FREQ = 4
TARGET_NET_SYNC_FREQ = 2000

def train_deepcnn_model(adversary_models):
    env = GameState()
    model = DeepCNNModel(camera_follow=True)        # ensure that this agent is at center of board
    models = [model] + adversary_models

    env.reset(models)
    state = env.get_state()
    pixels = env.get_pixels()
    
    # used for frame skipping, to simply repeat the last action chosen
    action = None

    training = True
    while training:
        # append current pixel state to buffer
        model.state_buffer.append(model.preprocess_state(pixels))

        # only update the current action every FRAME_SKIP frames of the game
        if model.step_count % FRAME_SKIP == 0:
            # stack last tau frames to get CNN action based on them
            s_0 = np.stack([model.state_buffer])
            action = model.get_action(s_0)
            model.net_update_count += 1

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
        if model.step_count % UPDATE_FREQ == 0:
            model.optimize()

        # re-sync the target network to updated network every on function of number
        # of net parameter updates
        if model.net_update_count % TARGET_NET_SYNC_FREQ == 0:
            model.sync_target_net()

        if dones[0]:
            break
        
        # move state and pixels one step forward
        state = next_state
        pixels = next_pixels

train_deepcnn_model([])