import pygame

# Display settings
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 830
AGENT_NAME_FONT = "comicsans"
AGENT_NAME_FONT_SIZE = 20
NUM_DISPLAYED_ON_LEADERBOARD = 5

# Game settings
BOARD_WIDTH = 1600
BOARD_HEIGHT = 830
FOOD_MASS = 1
GAME_MASS = 20000
MAX_FOOD = 200

MAX_VIRUSES = 5
VIRUS_MASS = 100
VIRUS_CONSUME_MASS_FACTOR = 1.33    # how much larger one has to be to eat a virus
VIRUS_COLOR = ((125, 250, 91))
VIRUS_OUTLINE_COLOR = ((112, 226, 81))


AGENT_STARTING_MASS = 20
AGENT_STARTING_SPEED = 10
NUM_AI = 2
CONSUME_MASS_FACTOR = 1.1   # how much larger one has to be to eat a colliding agent
SPLIT_LIMIT = 16            # max number of individual cells a single agent can have