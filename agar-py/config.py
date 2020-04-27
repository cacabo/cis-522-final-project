import pygame

# ------------------------------------------------------------------------------
# Display settings
# ------------------------------------------------------------------------------

FULL_SCREEN = True
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
AGENT_NAME_FONT = 'comicsans'
AGENT_NAME_FONT_SIZE = 20
NUM_DISPLAYED_ON_LEADERBOARD = 5
CLOCK_TICK = 120

# ------------------------------------------------------------------------------
# Game settings
# ------------------------------------------------------------------------------

BOARD_WIDTH = 1600
BOARD_HEIGHT = 830
FOOD_MASS = 1
GAME_MASS = 20000
MAX_FOOD = 200

# ------------------------------------------------------------------------------
# Virus state
# ------------------------------------------------------------------------------

MAX_VIRUSES = 5
VIRUS_MASS = 100
VIRUS_CONSUME_MASS_FACTOR = 1.33  # how much larger one has to be to eat a virus

# ------------------------------------------------------------------------------
# Colors
# ------------------------------------------------------------------------------

WHITE_COLOR = ((255, 255, 255))
VIRUS_COLOR = ((125, 250, 91))
VIRUS_OUTLINE_COLOR = ((112, 226, 81))

RED_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (0, 0, 255)

# ------------------------------------------------------------------------------
# Agent state
# ------------------------------------------------------------------------------

MASS_DECAY_FACTOR = 0.9999
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
AGENT_STARTING_MASS = 200
MIN_MASS_TO_SHOOT = 50
MASS_MASS = 20  # The mass which cells can shoot out
MIN_CELL_MASS = 50
CELL_CONSUME_MASS_FACTOR = 1.1
AGENT_STARTING_SPEED = 20
AGENT_CELL_LIMIT = 16  # max number of individual cells a single agent can have
# can merge cells this many ticks after splitting
AGENT_TICKS_TO_MERGE_CELLS = 1000

# must wait a certain number of ticks after splitting before splitting again
AGENT_TICKS_TO_SPLIT_AGAIN = 5

# ------------------------------------------------------------------------------
# Agent orientations
# ------------------------------------------------------------------------------

UP = 'UP'
UP_RIGHT = 'UP_RIGHT'
RIGHT = 'RIGHT'
DOWN_RIGHT = 'DOWN_RIGHT'
DOWN = 'DOWN'
DOWN_LEFT = 'DOWN_LEFT'
LEFT = 'LEFT'
UP_LEFT = 'UP_LEFT'

# ------------------------------------------------------------------------------
# AI state
# ------------------------------------------------------------------------------

NUM_AI = 2

# ------------------------------------------------------------------------------
# Other rewards
# ------------------------------------------------------------------------------

SURVIVAL_REWARD = 0
DEATH_REWARD = -100
