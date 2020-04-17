import torch
import os

ROOT = './store/'
DF_PATH = ROOT + 'dfs/'
NET_PATH = ROOT + 'nets/'


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
    checkpoint = torch.load(NET_PATH + filename + '.pt')
    net.load_state_dict(checkpoint['net'])
    return net


def save_df_to_disk(df, fname):
    if not os.path.exists(DF_PATH):
        os.mkdir(DF_PATH)

    df.to_csv(DF_PATH + fname + ".csv")
