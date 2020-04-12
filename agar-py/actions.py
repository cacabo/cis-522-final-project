from enum import Enum

# TODO add split and shoot


class Action(Enum):
    MOVE_RIGHT = 0
    MOVE_UP_RIGHT = 1
    MOVE_UP = 2
    MOVE_UP_LEFT = 3
    MOVE_LEFT = 4
    MOVE_DOWN_LEFT = 5
    MOVE_DOWN = 6
    MOVE_DOWN_RIGHT = 7
