import torch
import os
import pickle
from copy import deepcopy

ROOT = './store/'
DF_PATH = ROOT + 'dfs/'
NET_PATH = ROOT + 'nets/'
REPLAY_BUF_PATH = ROOT + 'replay_bufs/'


def save_deep_cnn_to_disk(cnn_model, filename):
    save_net_to_disk(cnn_model.net, filename)
    save_replay_buf_to_disk(cnn_model.replay_buffer, filename.rsplit('.', 1)[0] + '_replay_buf.dill')


def load_deep_cnn_from_device(cnn_model, filename, device):
    saved_net = load_net_from_device(cnn_model.net, filename, device)
    saved_buf = load_replay_buf_from_disk(filename.rsplit('.', 1)[0] + '_replay_buf.dill')

    cnn_model.net = saved_net
    cnn_model.target_net = deepcopy(cnn_model.net)
    cnn_model.replay_buffer = saved_buf
    return cnn_model


def save_net_to_disk(net, filename):
    """
    Save a net's parameters to a file(persistent store) at the provided filename
    """
    if not os.path.exists(NET_PATH):
        os.mkdir(NET_PATH)

    # Set up checkpoint
    checkpoint = {'net': net.state_dict()}
    torch.save(checkpoint, NET_PATH + filename + '.pt')


def load_net_from_disk(net, filename):
    """
    NOTE prints IncompatibleKeys(missing_keys=[], unexpected_keys=[]) upon a
    successful load_state_dict ¯\_(ツ)_/¯
    """
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'gpu'
    return load_net_from_device(net, filename, device)


def load_net_from_device(net, filename, device):
    checkpoint = torch.load(NET_PATH + filename + '.pt', map_location=device)
    net.load_state_dict(checkpoint['net'])
    return net


def save_replay_buf_to_disk(buf, filename):
    if not os.path.exists(REPLAY_BUF_PATH):
        os.mkdir(REPLAY_BUF_PATH)
    
    with open(REPLAY_BUF_PATH + filename + '.dill', 'wb') as f:
        pickle.dump(buf, f)


def load_replay_buf_from_disk(filename):
    with open(REPLAY_BUF_PATH + filename + '.dill', 'rb') as f:
        return pickle.load(f)


def save_df_to_disk(df, fname):
    if not os.path.exists(DF_PATH):
        os.mkdir(DF_PATH)

    df.to_csv(DF_PATH + fname + ".csv")
