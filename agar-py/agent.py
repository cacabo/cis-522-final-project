import pygame
import numpy as np
import config as conf
from shoot_mass import ShootMass


class Agent():
    def __init__(self, x, y, r, mass, color, name, manual_control):
        self.x_pos = x
        self.y_pos = y
        self.radius = r
        self.mass = mass
        self.color = color
        self.name = name
        self.orientation = conf.UP  # For deciding direction to shoot in

        self.is_alive = True
        self.manual_control = manual_control
        self.ai_dir = None
        self.ai_steps = 0
        self.directions = [self.move_left, self.move_right, self.move_up, self.move_down,
                           self.move_upleft, self.move_upright, self.move_downleft, self.move_downright]

    def move_left(self, vel):
        self.x_pos = max(self.x_pos - vel, self.radius)

    def move_right(self, vel):
        self.x_pos = min(self.x_pos + vel, conf.BOARD_WIDTH - self.radius)

    def move_up(self, vel):
        self.y_pos = max(self.y_pos - vel, self.radius)

    def move_down(self, vel):
        self.y_pos = min(self.y_pos + vel, conf.BOARD_HEIGHT - self.radius)

    def move_upleft(self, vel):
        self.move_up(vel)
        self.move_left(vel)

    def move_upright(self, vel):
        self.move_up(vel)
        self.move_right(vel)

    def move_downleft(self, vel):
        self.move_down(vel)
        self.move_left(vel)

    def move_downright(self, vel):
        self.move_down(vel)
        self.move_right(vel)

    def handle_move_keys(self, keys, camera):
        # TODO: better velocity control
        vel = int(max(conf.AGENT_STARTING_SPEED - (self.mass * 0.05), 1))

        is_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        is_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        is_up = keys[pygame.K_UP] or keys[pygame.K_w]
        is_down = keys[pygame.K_DOWN] or keys[pygame.K_s]

        # remove contradictory keys
        if is_left and is_right:
            is_left = False
            is_right = False

        if is_down and is_up:
            is_down = False
            is_up = False

        # set orientation and move
        if is_up:
            if is_left:
                self.move_upleft(vel)
                self.orientation = conf.UP_LEFT
            elif is_right:
                self.move_upright(vel)
                self.orientation = conf.UP_RIGHT
            else:
                self.move_up(vel)
                self.orientation = conf.UP
        elif is_down:
            if is_left:
                self.move_downleft(vel)
                self.orientation = conf.DOWN_LEFT
            elif is_right:
                self.move_downright(vel)
                self.orientation = conf.DOWN_RIGHT
            else:
                self.move_down(vel)
                self.orientation = conf.DOWN
        elif is_left:
            self.move_left(vel)
            self.orientation = conf.LEFT
        elif is_right:
            self.move_right(vel)
            self.orientation = conf.RIGHT

        # NOTE if none of the above cases are matched, the orientation does not change

        # move camera
        if is_left:
            camera.move_left(vel)

        if is_right:
            camera.move_right(vel)

        if is_up:
            camera.move_up(vel)

        if is_down:
            camera.move_down(vel)

    def handle_shoot(self):
        if self.mass < conf.MIN_MASS_TO_SHOOT:
            return

        self.mass = self.mass - conf.SHOOT_MASS

        # TODO shoot
        return

    def handle_split(self):
        if self.mass < conf.MIN_MASS_TO_SPLIT:
            return

        # TODO create other part to this agent (and if there are two parts split to 4)
        self.mass = self.mass / 2
        return

    def handle_other_keys(self, keys, camera):
        is_split = keys[pygame.K_SPACE]
        is_shoot = keys[pygame.K_q]

        if is_split:
            self.handle_split()
        elif is_shoot:
            self.handle_shoot()

    def ai_move(self):
        # TODO: better velocity control
        vel = int(max(conf.AGENT_STARTING_SPEED - (self.mass * 0.05), 1))

        # force AI to move between 5 and 10 (inclusive) steps in random direction
        if self.ai_steps <= 0:
            self.ai_steps = np.random.randint(5, 11)
            self.ai_dir = self.directions[np.random.randint(8)]

        # move in randomly chosen direction
        self.ai_dir(vel)
        self.ai_steps -= 1
