from models.DeepCNNModel import DeepCNNModel
from gamestate import GameState, start_ai_only_game
import numpy as np
from collections import deque
import fsutils as fs
import time
from utils import current_milli_time, get_random_action
import torch

import matplotlib.pyplot as plt

from trainutil import select_model_actions, plot_training_episode_avg_loss, plot_episode_rewards_with_mean, plot_episode_scores_with_mean, plot_episode_steps_survived_with_mean, get_means_over_window

def train_deepcnn_model(cnn_model, model_name, adversary_models, frame_skip=4,
                        update_freq=4, target_net_sync_freq=500, max_eps=200,
                        max_steps_per_ep=500, mean_window=10, prefill_buffer=False):
    env = GameState()

    # ensure the CNN model is centered in window
    if not cnn_model.camera_follow:
        raise ValueError('Camera needs to be following CNN')

    models = [cnn_model] + adversary_models
    training_losses = []
    training_rewards = []
    training_scores = []
    training_steps_survived = []

    start_time = current_milli_time()

    # if specified, burn in the replay buffer to fill it with some examples before starting to train
    if prefill_buffer:
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

        print('Replay buffer filled with ' + str(len(cnn_model.replay_buffer)) + ' samples!')
    else:
        print('Replay buffer prefill disabled.')

    print ('Beginning training...')
    for ep in range(max_eps):
        env.reset(models)
        state = env.get_state()
        pixels = env.get_pixels()
        
        # used for frame skipping, to simply repeat the last action chosen
        action = None

        update_losses = []
        ep_reward = 0
        print('=== Starting Episode %s ===' % ep)
        for step in range(max_steps_per_ep):
            if cnn_model.step_count % 250 == 0:
                print('Step %s' % cnn_model.step_count)
            # append current pixel state to buffer
            cnn_model.state_buffer.append(cnn_model.preprocess_state(pixels))

            # only update the current action every FRAME_SKIP frames of the game
            if step % frame_skip == 0:
                with torch.no_grad():
                    cnn_model.net.eval()
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

        cnn_agent = env.get_agent_of_model(cnn_model)

        # also keep track of episode mean update loss, episode score,
        # episode reward, and mean score/reward over last {mean_window} episodes
        training_losses.append(np.mean(update_losses))
        training_rewards.append(ep_reward)
        training_scores.append(cnn_agent.max_mass)
        training_steps_survived.append(cnn_agent.steps_taken)

        print('Ep Score: {:.4f} | Mean Score: {:.4f} | Steps Survived: {:d} | Mean Steps Survived: {:.2f}'.format(
            cnn_agent.max_mass, np.mean(training_scores[-mean_window:]),
            cnn_agent.steps_taken, np.mean(training_steps_survived[-mean_window:])))
        print('Mean Ep Loss: {:.4f} | Ep Reward: {:.4f} | Mean Reward: {:.4f}'.format(
            np.mean(update_losses), ep_reward, np.mean(training_scores[-mean_window:])))
        print('Model has been training for {:.4f} minutes.'.format((current_milli_time() - start_time) / 60000))
    

    # save the full model!
    fs.save_deep_cnn_to_disk(cnn_model, model_name)

    # plot training loss, training score, reward, and steps survived
    plot_training_episode_avg_loss(training_losses, model_name, plot_mean=True, window_size=mean_window)
    plot_episode_rewards(training_rewards, model_name, plot_mean=True, window_size=mean_window)
    plot_episode_scores(training_scores, model_name, plot_mean=True, window_size=mean_window)
    plot_episode_steps_survived(training_steps_survived, model_name, plot_mean=True, window_size=mean_window)
    plt.show()