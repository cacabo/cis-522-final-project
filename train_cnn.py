from models.DeepCNNModel import DeepCNNModel
from trainutil import train_deepcnn_model, get_epsilon_decay_factor

# model hyperparameters
MODEL_NAME = 'my_cnn_model'
TAU = 4
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_WINDOW = 50
REPLAY_BUF_CAPACITY = 10000
REPLAY_BUF_PREFILL_AMT = 1000
LR = 0.001
DOWNSAMPLE_SIZE = (112, 112)
BATCH_SIZE = 32

# training hyperparameters
ADVERSARY_MODELS = []
FRAME_SKIP = 4
UPDATE_FREQ = 4
TARGET_NET_SYNC_FREQ = 1000
MAX_EPS = 250
MAX_STEPS_PER_EP = 1000
WINDOW_SIZE = 10
ENABLE_PREFILL_BUFFER = True

cnn_model = DeepCNNModel(tau=TAU, gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END,
                         eps_decay_factor=get_epsilon_decay_factor(EPS_START, EPS_END, EPS_DECAY_WINDOW),
                         replay_buf_capacity=REPLAY_BUF_CAPACITY, replay_buf_prefill_amt=REPLAY_BUF_PREFILL_AMT,
                         lr=LR, downsample_size=DOWNSAMPLE_SIZE, batch_size=BATCH_SIZE)

train_deepcnn_model(cnn_model, MODEL_NAME, ADVERSARY_MODELS, frame_skip=FRAME_SKIP,
                    update_freq=UPDATE_FREQ, target_net_sync_freq=TARGET_NET_SYNC_FREQ,
                    max_eps=MAX_EPS, max_steps_per_ep=MAX_STEPS_PER_EP,
                    mean_window=WINDOW_SIZE, prefill_buffer=ENABLE_PREFILL_BUFFER)