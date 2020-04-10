from gamestate import GameState
from models.RandomModel import RandomModel
# from models.DQNModel import DQNModel


def select_model_actions(models, state):
    # TODO: for each model in models, give it the current state to compute the action it will take
    model_actions = []
    for model in models:
        model_actions.append(model.get_action(state))
    return model_actions


def optimize_models(models, rewards):
    # TODO: for each model in models, optimize it based on reward received
    for (model, reward) in zip(models, rewards):
        model.optimize(reward)


EPISODES = 10  # the number of games we're playing
MAX_STEPS = 1000

# Define environment
env = GameState()
# Define the DQNAgent
# agent = VanillaAgent(env.ACTION_SPACE, None) #TODO: update observation space

rand_model_1 = RandomModel(min_steps=5, max_steps=10)
rand_model_2 = RandomModel(min_steps=5, max_steps=10)
models = [rand_model_1, rand_model_2]

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

for episode in range(EPISODES):
    done = False  # whether game is done or not (terminal state)
    # reset the environment to fresh starting state with game agents initialized for models
    env.reset(models)
    # episode_reward = 0 # some notion of the reward in this episode
    # steps? (essentially game ticks? if want to cap # of ticks so game doesn't go on indefinitely)

    # get the first state
    state = env.get_state()
    while not done:  # game loop, can also incorporate steps here
        actions = select_model_actions(models, state)

        # environment determines new state, reward, whether terminal, based on actions taken by all models
        rewards, done = env.update_game_state(models, actions)

        # look at the new state
        if not done:
            next_state = env.get_state()
        else:
            next_state = None
        # update replay memory
        # memory.remember(state, action, next_state, reward)

        state = next_state  # update the state

        # optimize models
        optimize_models(models, rewards)

        # check for termination?
        if done:
            break
