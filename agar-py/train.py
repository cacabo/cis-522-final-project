from gamestate import GameState, start_ai_only_game
from models.RandomModel import RandomModel
from models.HeuristicModel import HeuristicModel
from models.DeepRLModel import DeepRLModel
from trainutil import train_models, test_models
# import utils
# import config as conf
import fsutils as fs

# Define environment
env = GameState()

deep_rl_model = DeepRLModel()
# heuristic_model = HeuristicModel()
rand_model_1 = RandomModel(min_steps=5, max_steps=10)
rand_model_2 = RandomModel(min_steps=5, max_steps=10)

models = [deep_rl_model, rand_model_1, rand_model_2]

train_models(env, models, episodes=1, steps=500)
test_models(env, models, steps=500)
# fs.save_net_to_disk(deep_rl_model.model, "test_drl_4")

# deep_rl_model.eval = True
# main_model = ('DeepRL', deep_rl_model)
# other_models = [('Random1', rand_model_1), ('Random2', rand_model_2)]
# start_ai_only_game(main_model, other_models)