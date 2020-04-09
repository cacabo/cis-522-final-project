import numpy as np
from game import Game


EPISODES = 10 # the number of games we're playing

# Exploration (this could be moved to the agent instead though)
epsilon = 0.99 
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

# TODO: define environment 
env = Game()
# TODO: define the DQNAgent

for episode in range(EPISODES):
    done = False # whether game is done or not (terminal state)
    state =  env.reset() # reset the environment to fresh starting state TODO: implement reset in game.py
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

