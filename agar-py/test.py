from models.DeepRLModel import DeepRLModel
from models.RandomModel import RandomModel
from gamestate import GameState, start_ai_only_game
import fsutils as fs
# from train import test_models

deep_rl_model = DeepRLModel()
rand_model_1 = RandomModel(min_steps=5, max_steps=10)
rand_model_2 = RandomModel(min_steps=5, max_steps=10)

deep_rl_model.model = fs.load_net_from_disk(deep_rl_model.model, "test_drl_10")

# models = [deep_rl_model, rand_model_1, rand_model_2]
# env = GameState()
# test_models(env, models)

deep_rl_model.eval = True
main_model = ('DeepRL', deep_rl_model)
other_models = [('Random1', rand_model_1), ('Random2', rand_model_2)]
start_ai_only_game(main_model, other_models)