from collections import deque
import torch
import numpy as np
import random
import torch
import torch.nn as nn
from models.ModelInterface import ModelInterface
from actions import Action
import config as conf
import utils

# Exploration (this could be moved to the agent instead though)
EPSILON = 0.95  # NOTE this is the starting value, which decays over time
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

GAMMA = 0.99

BATCH_SIZE = 32
REPLAY_BUFFER_LENGTH = 1000

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
        agent, objs, obj_angles, obj_dists, 0, avg_angles[0])
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

    # TODO improvements to how we encode state
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

# Model agent


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        qvals = self.fc3(x)
        return qvals


class DeepRLModel(ModelInterface):
    def __init__(self):
        super().__init__()

        # init replay buffer
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_LENGTH)

        # init model
        self.model = DQN(41, len(Action))  # TODO: fix w/ observation space
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss = torch.nn.SmoothL1Loss()

        # run on a GPU if we have access to one in this env
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.done = False

    def get_action(self, state):
        if self.done:
            return None
        if random.random() > self.epsilon:
            # take the action which maximizes expected reward
            state = encode_agent_state(self, state)

            # print(state)
            state = torch.Tensor(state)
            q_values = self.model(state)
            action = torch.argmax(q_values).item()  # TODO: placeholder
            action = Action(action)
        else:
            # take a random action
            action = Action(np.random.randint(len(Action)))  # random action
        return action

    def remember(self, state, action, next_state, reward, done):
        """Update the replay buffer with this example if we are not done yet"""
        if self.done:
            return

        self.replay_buffer.append(
            (state, action.value, next_state, reward, done))
        self.done = done

    def optimize(self):  # or experience replay
        if self.done:
            return

        # TODO: could toggle batch_size to be diff from minibatch below
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, next_states, rewards, dones = zip(*batch)

        states = [encode_agent_state(self, state) for state in states]
        next_states = [encode_agent_state(self, state)
                       for state in next_states]
        states = torch.Tensor(states).to(self.device)
        actions = torch.LongTensor(list(actions)).to(self.device)
        # print(rewards)
        # print(type(rewards))
        rewards = torch.Tensor(list(rewards)).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)

        # what about these/considering terminating states TODO: mask, properly update the eqns
        dones = torch.Tensor(list(dones)).to(self.device)

        # do Q computation
        # TODO: understand the equations
        currQ = self.model(states).gather(
            1, actions.unsqueeze(1))  # TODO: understand this
        nextQ = self.model(next_states)
        max_nextQ = torch.max(nextQ, 1)[0]
        expectedQ = rewards + (1 - dones) * self.gamma * max_nextQ

        loss = self.loss(currQ, expectedQ.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epislon
        if self.epsilon != MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)
