from models.DeepRLModel import DeepRLModel
from models.RandomModel import RandomModel
from models.DeepCNNModel import DeepCNNModel
from gamestate import GameState, start_ai_only_game
import fsutils as fs
# from train import test_models

deep_cnn_model = DeepCNNModel(camera_follow=True)
rand_model_1 = RandomModel(min_steps=5, max_steps=10)
rand_model_2 = RandomModel(min_steps=5, max_steps=10)

deep_cnn_model.net = fs.load_net_from_disk(deep_cnn_model.net, "dqn_cnn_v1")

# models = [deep_rl_model, rand_model_1, rand_model_2]
# env = GameState()
# test_models(env, models)

#deep_rl_model.eval = True
main_model = ('DeepCNN', deep_cnn_model)
other_models = [('Random1', rand_model_1), ('Random2', rand_model_2)]
start_ai_only_game(main_model, [])