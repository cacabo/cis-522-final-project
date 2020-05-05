import pygame
import config as conf
from gamestate import start_game
from models.HeuristicModel import HeuristicModel
from models.RandomModel import RandomModel

ai_models = [('Random', RandomModel(5, 10)), ('Heuristic', HeuristicModel())]
start_game(ai_models)
