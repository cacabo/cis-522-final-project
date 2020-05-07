import numpy as np
import matplotlib.pyplot as plt
import pygame

from models.RandomModel import RandomModel
from models.HeuristicModel import HeuristicModel
from models.DeepCNNModel import DeepCNNModel
from models.DeepRLModel import DeepRLModel
from gamestate import start_game, start_ai_only_game
import config as conf
import fsutils as fs

def evaluate_human_eating_food(num_trials):
    running_scores = []
    for i in range(num_trials):
        scores = start_game([], eval_mode=True)
        running_scores.append(scores)
    return np.mean(running_scores, axis=0)

def evaluate_model_eating_food(models, num_trials):
    avg_running_scores = []
    human_scores = evaluate_human_eating_food(num_trials)

    for model in models:
        running_scores = []
        for i in range(num_trials):
            scores = start_ai_only_game(model, [], eval_mode=True)
            running_scores.append(scores)
        avg_running_scores.append(np.mean(running_scores, axis=0))
    
    # add human evaluation at the end
    models.append(('Human', None))
    avg_running_scores.append(human_scores)

    pygame.quit()
    plt.figure()
    plt.title('Food Eating Performance')
    handles = []
    for i in range(len(avg_running_scores)):
        handle, = plt.plot([i for i in range(conf.TIME_LIMIT)], avg_running_scores[i], label=models[i][0])
        handles.append(handle)
        print(str(models[i][0]) + ' Final Mean Score: ' + str(avg_running_scores[i][-1]))
    plt.legend(handles=handles)
    plt.show()

TAU = 4
GAMMA = 0.95
EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY_WINDOW = 50
REPLAY_BUF_CAPACITY = 10000
REPLAY_BUF_PREFILL_AMT = 5000
LR = 0.001
DOWNSAMPLE_SIZE = (112, 112)
BATCH_SIZE = 32

cnn_model = DeepCNNModel(tau=TAU, gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END,
                    eps_decay_window=EPS_DECAY_WINDOW, replay_buf_capacity=REPLAY_BUF_CAPACITY,
                    replay_buf_prefill_amt=REPLAY_BUF_PREFILL_AMT, lr=LR,
                    downsample_size=DOWNSAMPLE_SIZE, batch_size=BATCH_SIZE)
cnn_model.net = fs.load_net_from_disk(cnn_model.net, 'important/dqn_cnn_500ep_v2')
cnn_model.eval = True

rl_model = DeepRLModel()
rl_model.model = fs.load_net_from_disk(rl_model.model, 'drl_fc32')
rl_model.eval = True

evaluate_model_eating_food([('DQN-CNN', cnn_model), ('DQN-FF', rl_model), ('Heuristic', HeuristicModel()), ('Random', RandomModel(min_steps=5, max_steps=10))], num_trials=10)