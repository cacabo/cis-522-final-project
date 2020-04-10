from game import Game
from DQNAgent import VanillaAgent


EPISODES = 10 # the number of games we're playing

# Define environment 
env = Game()
# Define the DQNAgent
agent = VanillaAgent(env.ACTION_SPACE, None) #TODO: update observation space

for episode in range(EPISODES):
    done = False # whether game is done or not (terminal state)
    state =  env.reset() # reset the environment to fresh starting state TODO: implement reset in game.py
    episode_reward = 0 # some notion of the reward in this episode
    # steps? (essentially game ticks? if want to cap # of ticks so game doesn't go on indefinitely)

    while not done: #game loop, can also incorporate steps here

        action = agent.get_action(state)
        next_state, reward, done = env.update_game_state(action) #environment determines new state, reward, whether terminal, based on action taken #TODO: implement in game.py
        episode_reward += reward

        # update replay memory and train network
        agent.remember(state, action, next_state, reward, done)
        agent.train()

        #check for termination?

        state = next_state #update the state

        

