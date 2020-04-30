from models.DeepCNNModel import DeepCNNModel
from gamestate import GameState, start_ai_only_game
import numpy as np
from collections import deque
import fsutils
import time
from utils import current_milli_time, get_random_action

import matplotlib.pyplot as plt

from trainutil import select_model_actions

def train_deepcnn_model(cnn_model, model_name, adversary_models, frame_skip=4,
                        update_freq=4, target_net_sync_freq=500, max_eps=200,
                        max_steps_per_ep=500, mean_reward_window=10):
    env = GameState()
    cnn_model.camera_follow = True              # ensure the CNN model is centered in window
    models = [cnn_model] + adversary_models
    training_losses = []
    training_rewards = []
    mean_rewards = []

    start_time = current_milli_time()

    # burn in the replay buffer to fill it with some examples before starting to train
    print('Filling replay buffer to ' + str(cnn_model.replay_buffer.prefill_amt * 100 / cnn_model.replay_buffer.capacity) + '% capacity...')
    env.reset(models)
    pixels = env.get_pixels()
    while cnn_model.replay_buffer.prefill_capacity() < 1.0:
        cnn_model.state_buffer.append(cnn_model.preprocess_state(pixels))

        actions = [get_random_action() for m in models]
        rewards, dones = env.update_game_state(models, actions)

        next_pixels = env.get_pixels()
        cnn_model.next_state_buffer.append(cnn_model.preprocess_state(pixels))
        cnn_model.remember(pixels, actions[0], next_pixels, rewards[0], dones[0])

        if dones[0]:
            env.reset(models)
            pixels = env.get_pixels()
        else:
            pixels = next_pixels

    print('Replay buffer filled with ' + str(len(cnn_model.replay_buffer)) + ' samples! Beginning training...')

    for ep in range(max_eps):
        env.reset(models)
        state = env.get_state()
        pixels = env.get_pixels()
        
        # used for frame skipping, to simply repeat the last action chosen
        action = None

        update_losses = []
        ep_reward = 0
        print("=== Starting Episode %s ===" % ep)
        for step in range(max_steps_per_ep):
            if cnn_model.step_count % 250 == 0:
                print("Step %s" % cnn_model.step_count)
            # append current pixel state to buffer
            cnn_model.state_buffer.append(cnn_model.preprocess_state(pixels))

            # only update the current action every FRAME_SKIP frames of the game
            if step % frame_skip == 0:
                # stack last tau frames to get CNN action based on them
                s_0 = np.stack([cnn_model.state_buffer])
                action = cnn_model.get_stacked_action(s_0)

            cnn_model.step_count += 1

            # get actions for other models
            actions = select_model_actions(adversary_models, state)

            # determine rewards and dones by letting all models take actions
            rewards, dones = env.update_game_state(models, [action] + actions)
            ep_reward += rewards[0]

            # get the next pixel state and append to next state buffer
            next_state = env.get_state()
            next_pixels = env.get_pixels()
            cnn_model.next_state_buffer.append(cnn_model.preprocess_state(next_pixels))

            # push to replay buffer
            cnn_model.remember(pixels, action, next_pixels, rewards[0], dones[0])

            # optimize model
            if step % update_freq == 0:
                update_loss = cnn_model.optimize()
                cnn_model.net_update_count += 1
                if update_loss is not None:
                    update_losses.append(update_loss)

            # re-sync the target network to updated network every on function of number
            # of net parameter updates
            if cnn_model.net_update_count % target_net_sync_freq == 0:
                cnn_model.sync_target_net()

            if dones[0]:
                break
            
            # move state and pixels one step forward
            state = next_state
            pixels = next_pixels

        # at end of episode, decay epsilon
        if cnn_model.epsilon > cnn_model.end_epsilon:
            cnn_model.epsilon *= cnn_model.epsilon_decay_fac
        else:
            cnn_model.epsilon = cnn_model.end_epsilon

        # also keep track of episode mean update loss, episode reward, and
        # mean reward over last MEAN_REWARD_WINDOW episodes
        training_losses.append(np.mean(update_losses))
        training_rewards.append(ep_reward)
        
        mean_reward = np.mean(training_rewards[-mean_reward_window:])
        mean_rewards.append(mean_reward)

        print('Mean Episode Loss: {:.4f} | Episode Reward: {:.4f} | Mean Reward: {:.4f}'.format(np.mean(update_losses), ep_reward, mean_reward))
        print('Model has been training for {:.4f} minutes.'.format((current_milli_time() - start_time) / 60000))
    
    # save the model!
    fsutils.save_net_to_disk(cnn_model.net, model_name)

    # plot training loss and reward
    eps_enumerated = [i for i in range(max_eps)]
    plt.figure()
    plt.plot(eps_enumerated, training_losses)
    plt.title("Mean Loss per Training Episode")
    plt.xlabel("episode")
    plt.ylabel("loss")
    plt.savefig(str(model_name) + '_training_loss_plot.png')

    plt.figure()
    plt.plot(eps_enumerated, training_rewards, 'c-',
             eps_enumerated, mean_rewards, 'r-')
    plt.title("Reward per Training Episode")
    plt.xlabel("epsiode")
    plt.ylabel("reward")
    plt.savefig(str(model_name) + '_reward_plot.png')

    plt.show()