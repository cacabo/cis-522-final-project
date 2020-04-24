from gamestate import GameState, start_ai_only_game
from models.RandomModel import RandomModel
from models.HeuristicModel import HeuristicModel
from models.DeepRLModel import DeepRLModel
from operator import add
import numpy as np
import utils
import config as conf
import fsutils as fs


def select_model_actions(models, state):
    # TODO: for each model in models, give it the current state to compute the action it will take
    model_actions = []
    for model in models:
        model_actions.append(model.get_action(state))
    return model_actions


def optimize_models(models):
    # TODO: for each model in models, optimize it based on reward received
    # for (model, reward) in zip(models, rewards):
    #     model.optimize(reward)
    for model in models:
        model.optimize()


def update_models_memory(models, state, actions, next_state, rewards, dones):
    for (model, action, reward, done) in zip(models, actions, rewards, dones):
        model.remember(state, action, next_state, reward, done)


EPISODES = 10  # the number of games we're playing
MAX_STEPS = 2500

# Define environment
env = GameState()

deep_rl_model = DeepRLModel()
# heuristic_model = HeuristicModel()
rand_model_1 = RandomModel(min_steps=5, max_steps=10)
rand_model_2 = RandomModel(min_steps=5, max_steps=10)

models = [deep_rl_model, rand_model_1, rand_model_2]

def train_models(env, models):
    print("TRAIN mode")
    for episode in range(EPISODES):
        # done = False  # whether game is done or not (terminal state)
        # reset the environment to fresh starting state with game agents initialized for models
        episode_rewards = [0 for _ in models]
        for model in models:
            model.done = False
            model.eval = False

        env.reset(models)
        state = env.get_state()  # get the first state

        for step in range(MAX_STEPS):  # cap the num of game ticks
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

            state = next_state  # update the state
        print("------EPISODE %s rewards------" % episode)
        for idx, model in enumerate(models):
            print("Model %s: %s" % (model.id, episode_rewards[idx]))

def test_models(env, models):
    print("TEST mode")
    episode_rewards = [0 for _ in models]
    for model in models:
        model.done = False
        model.eval = True

    env.reset(models)
    state = env.get_state()  # get the first state

    for step in range(MAX_STEPS):  # cap the num of game ticks
        actions = select_model_actions(models, state)

        # environment determines new state, reward, whether terminal, based on actions taken by all models
        rewards, dones = env.update_game_state(models, actions)

        #TODO: update dones for other models, persist (otherwise negative rewards)

        next_state = env.get_state()

        episode_rewards = list(
            map(add, episode_rewards, rewards))  # update rewards

        # check for termination of our player #TODO
        if dones[0]:
            break

        state = next_state  # update the state
    print("------TEST rewards------")
    for idx, model in enumerate(models):
        print("Model %s: %s" % (model.id, episode_rewards[idx]))

train_models(env, models)
test_models(env, models)
fs.save_net_to_disk(deep_rl_model.model, "test_drl_3_100kbuff")

# deep_rl_model.eval = True
# main_model = ('DeepRL', deep_rl_model)
# other_models = [('Random1', rand_model_1), ('Random2', rand_model_2)]
# start_ai_only_game(main_model, other_models)


# ---------------------------------------------------------------------------- #

# for episode in range(EPISODES):
#     # done = False # whether game is done or not (terminal state)
#     # reset the environment to fresh starting state TODO: implement reset in game.py
#     state = env.reset(models)
#     # get the players from the env
#     players = dict()
#     # TODO: think about how we are assigning agent, and keeping track of main agent
#     for player in env.get_player_names():
#         players[player] = {"episode_reward": 0, "agent": agent}
#     # episode_reward = 0 # some notion of the reward in this episode

#     for step in range(MAX_STEPS):  # cap the # of game ticks
#         player_action = dict()
#         for player, (episode_reward, agent) in players.items():
#             action = agent.get_action(state)
#             player_action[player] = action

#         # environment determines new state, reward, whether terminal, based on action taken #TODO: implement in game.py
#         next_state, rewards, dones = env.update_game_state(player_action)
#         # episode_reward += reward
#         for player, (episode_reward, agent) in players.items():
#             reward = rewards[player]
#             done = dones[player]
#             episode_reward += reward

#             # update replay memory and train network
#             agent.remember(state, action, next_state, reward, done)
#             agent.train()

#             if done:
#                 del players[player]

#         # check for termination (no more players)
#         if not players | (len(players) == 1):
#             break
