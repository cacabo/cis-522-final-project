from collections import deque
import torch
import numpy as np
import random
import torch
import torch.nn as nn
import math
from models.ModelInterface import ModelInterface
from model_utils.ReplayBuffer import ReplayBuffer
from actions import Action
import config as conf
import utils

STATE_ENCODING_LENGTH = 45 - 8 - 16 - 8 - 4

# Anything further than max_dist will (likely, unless very large) be outside
# of the agent's field of view
# TODO we could account for radius in making this decision
# TODO this could also be based on actual screen dimensions, not just the
# longest radius
max_dist = math.sqrt(conf.BOARD_WIDTH ** 2 + conf.BOARD_HEIGHT ** 2)

# -------------------------------
# Other helpers
# -------------------------------


def get_avg_angles(angles):
    """
    For example, this would go from conf.ANGLES of [0, 90, 180, 270] to
    [45, 135, 225, 315]

    NOTE this is effectively a closure that should only be run once
    """
    angles = angles + [360]
    avg_angles = []
    for idx in range(0, len(angles) - 1):
        angle = angles[idx]
        next_angle = angles[idx + 1]
        avg_angle = (angle + next_angle) / 2
        avg_angles.append(avg_angle)
    return avg_angles


avg_angles = get_avg_angles(conf.ANGLES)


def get_direction_score(agent, obj_angles, obj_dists, min_angle, max_angle):
    """
    Returns score for all objs that are between min_angle and max_angle relative
    to the agent. Gives a higher score to objects which are closer. Returns 0 if
    there are no objects between the provided angles.

    Parameters

        agent      : Agent
        obj_angles : tensor of angles between agent and each object
        obj_dists  : tensor of distance between agent and each object
        min_angle  : number
        max_angle  : number greater than min_angle

    Returns

        score      : number
    """
    if min_angle is None or max_angle is None or min_angle < 0 or max_angle < 0:
        raise Exception('max_angle and min_angle must be positive numbers')
    elif min_angle >= max_angle:
        raise Exception('max_angle must be larger than min_angle')

    filter_mask_tensor = (obj_angles < max_angle) & (obj_angles >= min_angle)
    filtered_obj_dists = obj_dists[filter_mask_tensor]

    if filtered_obj_dists.shape[0] == 0:
        return 0

    # TODO A/B test this encoding
    obj_dists_inv = 1 / torch.sqrt(filtered_obj_dists)
    return torch.sum(obj_dists_inv).item()


def get_obj_poses_tensor(objs):
    obj_poses = []
    for obj in objs:
        (x, y) = obj.get_pos()
        obj_poses.append([x, y])
    obj_poses_tensor = torch.Tensor(obj_poses)
    return obj_poses_tensor


def get_diff_tensor(agent, objs):
    obj_poses_tensor = get_obj_poses_tensor(objs)
    (agent_x, agent_y) = agent.get_pos()
    agent_pos_tensor = torch.Tensor([agent_x, agent_y])
    diff_tensor = obj_poses_tensor - agent_pos_tensor
    return diff_tensor


def get_dists_tensor(diff_tensor):
    diff_sq_tensor = diff_tensor ** 2
    sum_sq_tensor = torch.sum(diff_sq_tensor, 1)  # sum all x's and y's
    dists_tensor = torch.sqrt(sum_sq_tensor)
    return dists_tensor


def get_filtered_angles_tensor(filtered_diff_tensor):
    diff_invert_y_tensor = filtered_diff_tensor * torch.Tensor([1, -1])
    dx = diff_invert_y_tensor[:, 0]
    dy = diff_invert_y_tensor[:, 1]
    radians_tensor = torch.atan2(dy, dx)
    filtered_angles_tensor = radians_tensor * 180 / math.pi

    # Convert negative angles to positive ones
    filtered_angles_tensor = filtered_angles_tensor + \
        ((filtered_angles_tensor < 0) * 360.0)
    filtered_angles_tensor = filtered_angles_tensor.to(torch.float)

    return filtered_angles_tensor


def get_direction_scores(agent, objs):
    """
    For each direction (from right around the circle to down-right), compute a
    score quantifying how many and how close the proided objects are in each
    direction.

    Parameters

        agent : Agent
        objs  : list of objects with get_pos() methods

    Returns

        list of numbers of length the number of directions agent can move in
    """
    if len(objs) == 0:
        return np.zeros(len(conf.ANGLES))

    # Build an array to put into a tensor
    diff_tensor = get_diff_tensor(agent, objs)
    dists_tensor = get_dists_tensor(diff_tensor)

    filter_mask_tensor = (dists_tensor <= max_dist) & (dists_tensor > 0)
    filter_mask_tensor = filter_mask_tensor.to(
        torch.bool)  # Ensure type is correct
    fitlered_dists_tensor = dists_tensor[filter_mask_tensor]
    filtered_diff_tensor = diff_tensor[filter_mask_tensor]

    # Invert y dimension since y increases as we go down
    filtered_angles_tensor = get_filtered_angles_tensor(filtered_diff_tensor)

    """
    Calculate score for the conic section immediately in the positive x
    direction of the agent (this is from -22.5 degrees to 22.5 degrees if
    there are 8 allowed directions)

    This calculation is unique from the others because it requires summing the
    state across two edges based on how angles are stored
    """
    zero_to_first_angle = get_direction_score(
        agent,
        filtered_angles_tensor,
        fitlered_dists_tensor,
        avg_angles[-1],
        360)
    last_angle_to_360 = get_direction_score(
        agent,
        filtered_angles_tensor,
        fitlered_dists_tensor,
        0,
        avg_angles[0])
    first_direction_state = zero_to_first_angle + last_angle_to_360

    # Compute score for each conic section
    direction_states = [first_direction_state]

    for i in range(0, len(avg_angles) - 1):
        min_angle = avg_angles[i]
        max_angle = avg_angles[i + 1]
        state = get_direction_score(
            agent,
            filtered_angles_tensor,
            fitlered_dists_tensor,
            min_angle,
            max_angle)
        direction_states.append(state)

    # Return list of scores (one for each direction)
    return direction_states


def encode_agent_state(model, state):
    (agents, foods, viruses, masses, time) = state

    # If the agent is dead
    if model.id not in agents:
        return np.zeros((STATE_ENCODING_LENGTH,))

    agent = agents[model.id]
    agent_mass = agent.get_mass()

    # TODO improvements to how we encode state
    # TODO what about ones that are larger but can't eat you?
    # TODO what if this agent is split up a bunch?? Many edge cases with bias to consider
    # TODO factor in size of a given agent in computing score
    # TODO if objects are close "angle" can be a lot wider depending on radius
    # TODO include mass in other agent cell score calculations? especially if eating them...

    # Compute a list of all cells in the game not belonging to this model's agent
    all_agent_cells = []
    for other_agent in agents.values():
        if other_agent == agent:
            continue
        all_agent_cells.extend(other_agent.cells)

    # Partition all other cells into sets of those larger and smaller than
    # the current agent in aggregate
    all_larger_agent_cells = []
    all_smaller_agent_cells = []
    for cell in all_agent_cells:
        if cell.mass >= agent_mass:
            all_larger_agent_cells.append(cell)
        else:
            all_smaller_agent_cells.append(cell)

    # Compute scores for cells for each direction
    larger_agent_state = get_direction_scores(agent, all_larger_agent_cells)
    smaller_agent_state = get_direction_scores(agent, all_smaller_agent_cells)

    # other_agent_state = np.concatenate(
    #     (larger_agent_state, smaller_agent_state))
    food_state = get_direction_scores(agent, foods)
    virus_state = get_direction_scores(agent, viruses)
    # mass_state = get_direction_scores(agent, masses)

    # Encode important attributes about this agent
    this_agent_state = [
        agent_mass,
        # len(agent.cells),
        # agent.get_avg_x_pos(),
        # agent.get_avg_y_pos(),
        # agent.get_stdev_mass(),
    ]

    encoded_state = np.concatenate((
        this_agent_state,
        food_state,
        # other_agent_state,
        # virus_state,
        # mass_state,
    ))

    return encoded_state


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()

    def forward(self, state):
        x = state
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        qvals = self.fc4(x)
        return qvals


class DeepRLModel(ModelInterface):
    def __init__(
            self,
            epsilon=1,
            min_epsilon=0.01,
            epsilon_decay=0.999,
            buffer_capacity=1000,
            gamma=0.9,
            batch_size=64,
            replay_buffer_learn_thresh=0.5,
            lr = 1e-3,
            model=None):
        super().__init__()

        # init replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.replay_buffer_learn_thresh = replay_buffer_learn_thresh

        # Before we start learning, we populate the replay buffer with states
        # derived from taking random actions
        self.learning_start = False

        # init model
        if model:
            self.model = model
        else:
            self.model = DQN(STATE_ENCODING_LENGTH, len(Action))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()

        # run on a GPU if we have access to one in this env
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.model.to(self.device)

        #target net
        self.target_net = DQN(STATE_ENCODING_LENGTH, len(Action)).to(self.device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.done = False

        self.min_steps = 5
        self.max_steps = 10
        self.steps_remaining = 0
        self.curr_action = None

    def is_replay_buffer_ready(self):
        return len(self.replay_buffer) >= self.replay_buffer_learn_thresh * self.replay_buffer.capacity

    def get_action(self, state):
        if self.eval:
            with torch.no_grad():
                state = encode_agent_state(self, state)
                state = torch.Tensor(state)
                q_values = self.model(state)
                action = torch.argmax(q_values).item()
                # print(Action(action))
                return Action(action)
        if self.done:
            return None

        if not self.is_replay_buffer_ready():
            return Action(np.random.randint(len(Action)))
            # if self.steps_remaining <= 0:
            #     self.steps_remaining = np.random.randint(
            #         self.min_steps, self.max_steps)
            #     self.curr_action = Action(np.random.randint(len(Action)))

            #     self.steps_remaining -= 1
            # return self.curr_action

        if random.random() > self.epsilon:
            # take the action which maximizes expected reward
            with torch.no_grad():
                state = encode_agent_state(self, state)
                state = torch.Tensor(state).to(self.device)
                q_values = self.model(state)
                action = torch.argmax(q_values).item()
                action = Action(action)
        else:
            # take a random action
            action = Action(np.random.randint(len(Action)))  # random action
        return action

    def remember(self, state, action, next_state, reward, done):
        """Update the replay buffer with this example if we are not done yet"""
        if self.done:
            return

        self.replay_buffer.push(
            (encode_agent_state(self, state), action.value, encode_agent_state(self, next_state), reward, done))
        self.done = done

    def optimize(self):  # or experience replay
        if self.done:
            return

        # TODO: could toggle batch_size to be diff from minibatch below
        if len(self.replay_buffer) < self.batch_size:
            return

        if not self.is_replay_buffer_ready():
            return

        if not self.learning_start:
            # Stop taking only random actions
            self.learning_start = True
            print("----LEARNING BEGINS----")

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.Tensor(states).to(self.device)
        actions = torch.LongTensor(list(actions)).to(self.device)

        rewards = torch.Tensor(list(rewards)).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)

        # what about these/considering terminating states
        dones = torch.Tensor(list(dones)).to(self.device)

        # do Q computation
        currQ = self.model(states).gather(
            1, actions.unsqueeze(1))
        nextQ = self.target_net(next_states)
        max_nextQ = torch.max(nextQ, 1)[0].detach()
        expectedQ = rewards + (1 - dones) * self.gamma * max_nextQ

        loss = self.loss(currQ, expectedQ.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        # decay epsilon
        if self.epsilon != self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

        return self.epsilon

    def sync_target_net(self):
        self.target_net.load_state_dict(self.model.state_dict())
