from gamestate import GameState, start_ai_only_game
from models.RandomModel import RandomModel
from models.HeuristicModel import HeuristicModel
from operator import add
from models.DQNModel import DQNModel
import numpy as np
import utils
import config as conf

# -------------------------------
# Other Helpers
# -------------------------------


def get_avg_angles():
    """
    For example, this would go from conf.ANGLES of [0, 90, 180, 270] to
    [45, 135, 225, 315]
    """
    angles = conf.ANGLES + [360]
    avg_angles = []
    for idx in range(0, len(angles) - 1):
        angle = angles[idx]
        next_angle = angles[idx + 1]
        avg_angle = (angle + next_angle) / 2
        avg_angles.append(avg_angle)
    return avg_angles


avg_angles = get_avg_angles()


def get_direction_score(agent, objs, obj_angles, obj_dists, min_angle, max_angle):
    """
    Returns score for all objs that are between min_angle and max_angle relative
    to the agent. Gives a higher score to objects which are closer. Returns 0 if
    there are no objects between the provided angles.

    Parameters

        agent      : Agent
        objs       : list of objects with get_pos() methods
        obj_angles : list of angles between agent and each object
        obj_dists  : list of distance between agent and each object
        min_angle  : nubmer
        max_angle  : number greater than min_angle

    Returns

        number
    """
    if min_angle is None or max_angle is None or min_angle < 0 or max_angle < 0:
        raise Exception('max_angle and min_angle must be positive numbers')
    elif min_angle >= max_angle:
        raise Exception('max_angle must be larger than min_angle')

    filtered_objs = [
        objs[idx] for (idx, angle) in enumerate(obj_angles) if (
            angle >= min_angle and angle < max_angle
        )
    ]
    obj_dists = [utils.get_object_dist(
        agent, obj) for obj in filtered_objs]
    obj_dists_np = np.array(obj_dists)
    obj_dists_inv_np = 1 / np.sqrt(obj_dists_np)
    return np.sum(obj_dists_inv_np)


def get_direction_scores(agent, objs):
    """
    Parameters

        agent : Agent
        objs  : list of objects with get_pos() methods

    Returns

        list of numbers of length the number of directions agent can move in
    """
    obj_angles = [utils.get_angle_between_objects(agent, obj) for obj in objs]
    obj_dists = [utils.get_object_dist(agent, obj) for obj in objs]

    zero_to_first_angle = get_direction_score(
        agent, objs, obj_angles, obj_dists, avg_angles[-1], 360)
    last_angle_to_360 = get_direction_score(
        agent, obj_angles, obj_dists, objs, 0, avg_angles[0])
    first_direction_state = zero_to_first_angle + last_angle_to_360

    direction_states = [first_direction_state]

    for i in range(1, len(avg_angles) - 1):
        min_angle = avg_angles[i]
        max_angle = avg_angles[i + 1]
        state = get_direction_score(
            agent, objs, obj_angles, obj_dists, min_angle, max_angle)
        direction_states.append(state)

    return direction_states


def encode_agent_state(model, state):
    (agents, foods, viruses, masses, time) = state
    agent = agents[model.id]
    agent_mass = agent.get_mass()

    # TODO what about ones that are larger but can't eat you?
    # TODO what if this agent is split up a bunch?? Many edge cases with bias to consider
    # TODO factor in size of a given agent in computing score
    # TODO if objects are close "angle" can be a lot wider depending on radius

    all_agent_cells = []
    for other_agent in agents.values():
        if other_agent == agent:
            continue
        all_agent_cells.extend(other_agent.cells)

    all_larger_agent_cells = []
    all_smaller_agent_cells = []
    for cell in all_agent_cells:
        if cell.mass >= agent_mass:
            all_larger_agent_cells.append(cell)
        else:
            all_smaller_agent_cells.append(cell)

    larger_agent_state = get_direction_scores(agent, all_larger_agent_cells)
    smaller_agent_state = get_direction_scores(agent, all_smaller_agent_cells)

    other_agent_state = np.concatenate(
        (larger_agent_state, smaller_agent_state))
    food_state = get_direction_scores(agent, foods)
    virus_state = get_direction_scores(agent, viruses)
    mass_state = get_direction_scores(agent, masses)
    time_state = [time]
    this_agent_state = [
        agent_mass,
        len(agent.cells),
        agent.get_avg_x_pos(),
        agent.get_avg_y_pos(),
        agent.get_stdev_mass(),
    ]

    return np.concatenate((
        food_state,
        this_agent_state,
        other_agent_state,
        virus_state,
        mass_state,
        time_state,
    ))


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


def update_models_memory(models, state, actions, next_state, rewards, dones):
    for (model, action, reward, done) in zip(models, actions, rewards, dones):
        model.remember(state, action, next_state, reward, done)


EPISODES = 1  # the number of games we're playing
MAX_STEPS = 1000

# Define environment
env = GameState()
# Define the DQNAgent
# agent = VanillaAgent(env.ACTION_SPACE, None) #TODO: update observation space

heuristic_model = HeuristicModel()
rand_model_1 = RandomModel(min_steps=5, max_steps=10)
rand_model_2 = RandomModel(min_steps=5, max_steps=10)

models = [heuristic_model, rand_model_1, rand_model_2]

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
    # done = False  # whether game is done or not (terminal state)
    # reset the environment to fresh starting state with game agents initialized for models
    env.reset(models)
    episode_rewards = [0 for _ in models]
    # episode_reward = 0 # some notion of the reward in this episode

    state = env.get_state()  # get the first state

    for step in range(MAX_STEPS):  # cap the num of game ticks
        actions = select_model_actions(models, state)

        # environment determines new state, reward, whether terminal, based on actions taken by all models
        rewards, dones = env.update_game_state(models, actions)
        next_state = env.get_state()

        # TODO here is how to get state:
        # print('state', next_state)
        # encode_agent_state(models[0], next_state)

        episode_rewards = list(
            map(add, episode_rewards, rewards))  # update rewards
        update_models_memory(models, state, actions, next_state,
                             rewards, dones)  # update replay memory

        # optimize models
        optimize_models(models, rewards)

        # check for termination of our player #TODO
        # if dones[0]:
        #     break

        state = next_state  # update the state
    print("------EPISODE %s rewards------" % episode)
    for idx, model in enumerate(models):
        print("Model %s: %s" % (model.id, episode_rewards[idx]))

main_model = ('Heuristic', heuristic_model)
other_models = [('Random1', rand_model_1), ('Random2', rand_model_2)]
# start_ai_only_game(main_model, other_models)
