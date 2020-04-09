import numpy as np


EPISODES = 10 # the number of games we're playing

# Exploration
epsilon = 0.99 
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

# TODO: define environment 
# TODO: define the DQNAgent

for episode in range(EPISODES):
    done = False # whether game is done or not (terminal state)
    state =  # reset the environment to fresh starting state
    episode_reward = 0 # some notion of the reward in this episode
    # steps? (essentially game ticks? if want to cap # of ticks so game doesn't go on indefinitely)

    while not done: #game loop, can also incorporate steps here

        if np.random.random() > epsilon:
            action = # agent picks action based on state
        else:
            action = #random action TODO: define action space
        next_state, reward, done = #environment determines new state, reward, whether terminal, based on action taken

        # update replay memory and train network

        episode_reward += reward
        state = next_state #update the state

        # Decay epislon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

