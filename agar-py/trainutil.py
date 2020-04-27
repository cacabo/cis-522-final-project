from operator import add, and_
import numpy as np
from models.DeepCNNModel import DeepCNNModel
import utils
import config as conf
from functools import reduce


def select_model_actions(models, state):
    model_actions = []
    for model in models:
        model_actions.append(model.get_action(state))
    return model_actions


def optimize_models(models):
    for model in models:
        model.optimize()


def update_models_memory(models, state, actions, next_state, rewards, dones):
    for (model, action, reward, done) in zip(models, actions, rewards, dones):
        model.remember(state, action, next_state, reward, done)


def train_models(env, models, episodes=10, steps=2500, print_every=200):
    print("\nTRAIN MODE")
    for episode in range(episodes):
        # done = False  # whether game is done or not (terminal state)
        # reset the environment to fresh starting state with game agents initialized for models
        episode_rewards = [0 for _ in models]
        for model in models:
            model.done = False
            model.eval = False

        env.reset(models)
        state = env.get_state()  # get the first state

        for step in range(steps):  # cap the num of game ticks
            actions = select_model_actions(models, state)

            # environment determines new state, reward, whether terminal, based on actions taken by all models
            rewards, dones = env.update_game_state(models, actions)
            next_state = env.get_state()

            episode_rewards = list(
                map(add, episode_rewards, rewards))  # update rewards
            update_models_memory(models, state, actions, next_state,
                                 rewards, dones)  # update replay memory

            # optimize models
            optimize_models(models)

            # check for termination of our player #TODO
            if dones[0]:
                break
            # terminate if all other players are dead
            if (len(dones) > 1) & reduce(and_, dones[1:]):
                break

            state = next_state  # update the state

            if step % print_every == 0:
                print("----STEP %s rewards----" % step)
                for idx, model in enumerate(models):
                    print("Model %s: %s" % (model.id, episode_rewards[idx]))
        print("------EPISODE %s rewards------" % episode)
        for idx, model in enumerate(models):
            print("Model %s: %s" % (model.id, episode_rewards[idx]))


def test_models(env, models, steps=2500, print_every=200):
    print("\nTEST MODE")
    episode_rewards = [0 for _ in models]
    for model in models:
        model.done = False
        model.eval = True

    env.reset(models)
    state = env.get_state()  # get the first state

    for step in range(steps):  # cap the num of game ticks
        actions = select_model_actions(models, state)

        # environment determines new state, reward, whether terminal, based on actions taken by all models
        rewards, dones = env.update_game_state(models, actions)

        # TODO: update dones for other models, persist (otherwise negative rewards)

        next_state = env.get_state()

        episode_rewards = list(
            map(add, episode_rewards, rewards))  # update rewards

        # check for termination of our player #TODO
        if dones[0]:
            break

        state = next_state  # update the state

        if step % print_every == 0:
            print("----STEP %s rewards----" % step)
            for idx, model in enumerate(models):
                print("Model %s: %s" % (model.id, episode_rewards[idx]))
    print("------TEST rewards------")
    for idx, model in enumerate(models):
        print("Model %s: %s" % (model.id, episode_rewards[idx]))