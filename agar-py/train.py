from gamestate import GameState, start_ai_only_game
from models.RandomModel import RandomModel
from models.HeuristicModel import HeuristicModel
from models.DeepRLModel import DeepRLModel
from trainutil import train_models, test_models, get_epsilon_decay_factor
import random
import fsutils as fs
import sys

"""
Constants
"""

PRINT_EVERY = 1000

"""
Hyperparameters
"""

START_EPSILON = 1.0  # NOTE this is the starting value, which decays over time
MIN_EPSILON = 0.001
DECAY_EPISODE_WINDOW = 150

GAMMA = 0.99
BATCH_SIZE = 128

REPLAY_BUFFER_LEARN_THRESH = 0.1
REPLAY_BUFFER_CAPACITY = 10000

EPISODES = 200
STEPS_PER_EPISODE = 500

# TODO learning rate


def train(episodes=EPISODES, steps=STEPS_PER_EPISODE):
    """
    Training loop for training the DeepRLModel
    """
    print("Running Train | Episodes: {} | Steps: {}".format(episodes, steps))

    # Define environment
    env = GameState()

    epsilon_decay = get_epsilon_decay_factor(
        START_EPSILON, MIN_EPSILON, DECAY_EPISODE_WINDOW)
    deep_rl_model = DeepRLModel(
        epsilon=START_EPSILON,
        min_epsilon=MIN_EPSILON,
        epsilon_decay=epsilon_decay,
        buffer_capacity=REPLAY_BUFFER_CAPACITY,
        replay_buffer_learn_thresh=REPLAY_BUFFER_LEARN_THRESH,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
    )

    models = [deep_rl_model]

    # heuristic_model = HeuristicModel()
    # rand_model_1 = RandomModel(min_steps=5, max_steps=10)
    # rand_model_2 = RandomModel(min_steps=5, max_steps=10)

    # models = [deep_rl_model, rand_model_1, rand_model_2]

    train_models(env, models, episodes=episodes,
                 steps=steps, print_every=PRINT_EVERY)
    test_models(env, models, steps=steps)
    fs.save_net_to_disk(deep_rl_model.model,
                        "deep-rl-temp-{}".format(random.uniform(0, 2 ** 16)))

    # deep_rl_model.eval = True
    # main_model = ('DeepRL', deep_rl_model)
    # other_models = [('Random1', rand_model_1), ('Random2', rand_model_2)]
    # start_ai_only_game(main_model, other_models)


if __name__ == "__main__":
    num_args = len(sys.argv)

    if num_args == 3:
        episodes = int(sys.argv[1])
        steps = int(sys.argv[2])

        if not (episodes > 0 and steps > 0):
            raise ValueError('Usage: train.py {episodes} {steps}')

        train(episodes, steps)
    elif num_args == 1:
        train()
    else:
        raise ValueError('Usage: train.py {episodes} {steps}')
else:
    train()
