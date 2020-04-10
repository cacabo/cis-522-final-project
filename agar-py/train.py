from game import Game
from DQNModel import DQNModel
from RandomModel import RandomModel

def select_model_actions(models, state):
    # TODO: for each model in models, give it the current state to compute the action it will take
    model_actions = []
    for model in models:
        model_actions.append(model.get_action(state))
    return model_actions

def optimize_models(models, rewards):
    # TODO: for each model in models, optimize it based on reward received
    for (model, reward) in zip(models, rewards):
        model.optimize(reward)

EPISODES = 10 # the number of games we're playing

# Define environment 
env = Game()

# Define the models
# dqn_model = DQNModel()
rand_model_1 = RandomModel()
rand_model_2 = RandomModel()
models = [rand_model_1, rand_model_2]


for episode in range(EPISODES):
    done = False # whether game is done or not (terminal state)
    env.reset() # reset the environment to fresh starting state
    episode_reward = 0 # some notion of the reward in this episode
    # steps? (essentially game ticks? if want to cap # of ticks so game doesn't go on indefinitely)

    # get the first state
    state = env.get_state()
    while not done: #game loop, can also incorporate steps here
        actions = select_model_actions(models, state)
        # environment determines new state, reward, whether terminal, based on actions taken by all models
        # TODO: implement in game.py
        rewards, done = env.update_game_state_w(actions)

        # look at the new state
        if not done:
            next_state = env.get_state()
        else:
            next_state = None

        # update replay memory
        memory.remember(state, action, next_state, reward)

        state = next_state #update the state

        # optimize models
        optimize_models(models, rewards)

        #check for termination?
        if done:
            break
        

