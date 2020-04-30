from models.DeepCNNModel import DeepCNNModel
from gamestate import GameState
import numpy as np
from collections import deque
import fsutils

import matplotlib.pyplot as plt

from trainutil import select_model_actions

FRAME_SKIP = 4
UPDATE_FREQ = 4
TARGET_NET_SYNC_FREQ = 500
MAX_EPISODES = 100
MAX_STEPS_PER_EP = 500    # TODO: do we want this? does it make sense?
MEAN_REWARD_WINDOW = 10

def train_deepcnn_model(adversary_models):
    env = GameState()
    model = DeepCNNModel(camera_follow=True)        # ensure that this agent is at center of board
    models = [model] + adversary_models
    training_losses = []
    training_rewards = []
    mean_rewards = []

    for ep in range(MAX_EPISODES):
        env.reset(models)
        state = env.get_state()
        pixels = env.get_pixels()
        
        # used for frame skipping, to simply repeat the last action chosen
        action = None

        update_losses = []
        ep_reward = 0
        print("=== Starting Episode %s===" % ep)
        for step in range(MAX_STEPS_PER_EP):
            if model.step_count % 250 == 0:
                print("Step %s" % model.step_count)
            # append current pixel state to buffer
            model.state_buffer.append(model.preprocess_state(pixels))

            # only update the current action every FRAME_SKIP frames of the game
            if step % FRAME_SKIP == 0:
                # stack last tau frames to get CNN action based on them
                s_0 = np.stack([model.state_buffer])
                action = model.get_action(s_0)
                model.net_update_count += 1

            model.step_count += 1

            # get actions for other models
            actions = select_model_actions(adversary_models, state)

            # determine rewards and dones by letting all models take actions
            rewards, dones = env.update_game_state(models, [action] + actions)
            ep_reward += rewards[0]

            # get the next pixel state and append to next state buffer
            next_state = env.get_state()
            next_pixels = env.get_pixels()
            model.next_state_buffer.append(model.preprocess_state(next_pixels))

            # push to replay buffer
            model.remember(pixels, action, next_pixels, rewards[0], dones[0])

            # optimize model
            if step % UPDATE_FREQ == 0:
                update_loss = model.optimize()
                if update_loss is not None:
                    update_losses.append(update_loss)

            # re-sync the target network to updated network every on function of number
            # of net parameter updates
            if model.net_update_count % TARGET_NET_SYNC_FREQ == 0:
                model.sync_target_net()

            if dones[0]:
                break
            
            # move state and pixels one step forward
            state = next_state
            pixels = next_pixels

        # at end of MAX_STEPS/when game terminates, keep track of
        # episode mean update loss, episode reward, and mean reward over last
        # MEAN_REWARD_WINDOW episodes
        training_losses.append(np.mean(update_losses))
        training_rewards.append(ep_reward)
        mean_rewards.append(np.mean(training_rewards[-MEAN_REWARD_WINDOW:]))

        print('Mean Episode Loss: {:.4f} | Episode Reward: {:.4f} | Mean Reward: {:.4f}'.format(
            np.mean(update_losses), ep_reward, np.mean(training_rewards[-MEAN_REWARD_WINDOW:])
        ))
    
    # save the model!
    fsutils.save_net_to_disk(model.net, 'deepcnn_test_v2')

    # plot training loss and reward
    plt.figure()
    plt.plot([i for i in range(MAX_EPISODES)], training_losses)
    plt.title("Mean Loss per Training Episode")
    plt.xlabel("episode")
    plt.ylabel("loss")

    plt.figure()
    plt.plot([i for i in range(MAX_EPISODES)], training_rewards, 'c-',
             [i for i in range(MAX_EPISODES)], mean_rewards, 'r-')
    plt.title("Reward per Training Episode")
    plt.xlabel("epsiode")
    plt.ylabel("reward")

    plt.show()

train_deepcnn_model([])