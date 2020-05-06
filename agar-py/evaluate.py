import numpy as np
import matplotlib.pyplot as plt
import pygame

from models.RandomModel import RandomModel
from models.HeuristicModel import HeuristicModel
from models.DeepCNNModel import DeepCNNModel
from gamestate import start_ai_only_game
import config as conf
import fsutils as fs

def evaluate_model_eating_food(models, num_trials=10):
    avg_running_scores = []
    for model in models:
        running_scores = []
        for i in range(num_trials):
            scores = start_ai_only_game(model, [], eval_mode=True)
            running_scores.append(scores)
        avg_running_scores.append(np.mean(running_scores, axis=0))
    
    pygame.quit()
    plt.figure()
    plt.title('Food Eating Performance')
    handles = []
    for i in range(len(models)):
        handle, = plt.plot([i for i in range(conf.TIME_LIMIT)], avg_running_scores[i], label=models[i][0])
        handles.append(handle)
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

agarai_model = DeepCNNModel(tau=TAU, gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END,
                    eps_decay_window=EPS_DECAY_WINDOW, replay_buf_capacity=REPLAY_BUF_CAPACITY,
                    replay_buf_prefill_amt=REPLAY_BUF_PREFILL_AMT, lr=LR,
                    downsample_size=DOWNSAMPLE_SIZE, batch_size=BATCH_SIZE)
agarai_model.net = fs.load_net_from_disk(agarai_model.net, 'important/dqn_cnn_500ep_v2')
agarai_model.eval = True
main_model = ('AgarAI', agarai_model)

evaluate_model_eating_food([main_model, ('Heury', HeuristicModel()), ('Rando', RandomModel(min_steps=5, max_steps=10))])