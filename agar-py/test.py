from models.DeepRLModel import DeepRLModel
from models.RandomModel import RandomModel
from models.DeepCNNModel import DeepCNNModel
from gamestate import GameState, start_ai_only_game
import fsutils as fs
import sys
# from train import test_models

def test(model_type, model_name):
    # deep_cnn_model = DeepCNNModel(camera_follow=True)
    if model_type == 'drl':
        agarai_model = DeepRLModel()
        fs.load_net_from_disk(agarai_model.model, model_name)
    elif model_type == 'cnn':
        agarai_model = DeepCNNModel(camera_follow=True)

    # models = [deep_rl_model, rand_model_1, rand_model_2]
    # env = GameState()
    # test_models(env, models)

    agarai_model.eval = True
    main_model = ('AgarAI', agarai_model)

    rand_model_1 = RandomModel(min_steps=5, max_steps=10)
    rand_model_2 = RandomModel(min_steps=5, max_steps=10)
    other_models = [('Random1', rand_model_1), ('Random2', rand_model_2)]
    start_ai_only_game(main_model, [])

if __name__ == "__main__":
    num_args = len(sys.argv)

    if num_args == 3:
        model_type = sys.argv[1]
        model_name = sys.argv[2]

        if not ((model_type == 'drl') | (model_type == 'cnn')):
            raise ValueError('Usage: test.py {drl/cnn} {model_name}')

        test(model_type, model_name)
    else:
        raise ValueError('Usage: test.py {drl/cnn} {model_name}')
# else:
#     train()