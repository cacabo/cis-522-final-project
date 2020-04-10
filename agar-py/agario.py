import pygame
import config as conf
from gamestate import GameState

game = GameState()

# main game loop
game.init_manual_agent('AgarAI')
game.init_ai_agents(conf.NUM_AI)
game.main_loop()