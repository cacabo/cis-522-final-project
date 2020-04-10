from game import Game
from DQNAgent import VanillaAgent


EPISODES = 10 # the number of games we're playing
MAX_STEPS = 1000

# Define environment 
env = Game()
# Define the DQNAgent
agent = VanillaAgent(env.ACTION_SPACE, None) #TODO: update observation space

for episode in range(EPISODES):
    # done = False # whether game is done or not (terminal state)
    state =  env.reset() # reset the environment to fresh starting state TODO: implement reset in game.py
    # get the players from the env
    players = dict()
    for player in env.get_player_names(): #TODO: think about how we are assigning agent, and keeping track of main agent
        players[player] = {"episode_reward":0, "agent": agent}
    # episode_reward = 0 # some notion of the reward in this episode

    for step in range(MAX_STEPS): # cap the # of game ticks
        player_action = dict()
        for player, (episode_reward, agent) in players.items():
            action = agent.get_action(state)
            player_action[player] = action

        next_state, rewards, dones = env.update_game_state(player_action) #environment determines new state, reward, whether terminal, based on action taken #TODO: implement in game.py
        # episode_reward += reward
        for player, (episode_reward, agent) in players.items():
            reward = rewards[player]
            done = dones[player]
            episode_reward += reward

            # update replay memory and train network
            agent.remember(state, action, next_state, reward, done)
            agent.train()

            if done:
                del players[player]

        #check for termination (no more players)
        if not players | (len(players) == 1):
            break

        state = next_state #update the state

        

        

