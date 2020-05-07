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

PRINT_EVERY = 2000
NUM_CHECKPOINTS = 5

"""
Hyperparameters
"""

START_EPSILON = 1.0  # NOTE this is the starting value, which decays over time
MIN_EPSILON = 0.02
DECAY_EPISODE_WINDOW = 200

GAMMA = 0.8
BATCH_SIZE = 32

REPLAY_BUFFER_LEARN_THRESH = 0.1
REPLAY_BUFFER_CAPACITY = 100000

EPISODES = 200
STEPS_PER_EPISODE = 1000
LEARNING_RATE = 0.001


def train(episodes=EPISODES, steps=STEPS_PER_EPISODE):
    """
    Training loop for training the DeepRLModel
    """
    print("Running Train | Episodes: {} | Steps: {}".format(episodes, steps))

    # Define environment
    env = GameState(with_masses=False, with_viruses=False,
                    with_random_mass_init=True)

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
        lr=LEARNING_RATE,
    )

    # deep_rl_model.model = fs.load_net_from_device(
    #     deep_rl_model.model, "train_drl_54143_200.pt")

    heuristic_model = HeuristicModel()
    rand_model_1 = RandomModel(min_steps=5, max_steps=10)
    rand_model_2 = RandomModel(min_steps=5, max_steps=10)

    models = [deep_rl_model, heuristic_model, rand_model_1, rand_model_2]
    # models = [deep_rl_model]

    train_models(
        env,
        models,
        episodes=episodes,
        steps=steps,
        print_every=PRINT_EVERY,
        model_name="train_drl_with_others_{}".format(
            random.randint(0, 2 ** 16)),
        num_checkpoints=NUM_CHECKPOINTS)
    test_models(env, models, steps=steps)

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
