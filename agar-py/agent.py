import pygame
import numpy as np
import config as conf


class AgentCell():
    def __init__(self, x, y, r, mass):
        self.x_pos = x
        self.y_pos = y
        self.radius = r
        self.mass = mass

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


class Agent():
    orientations = [
        conf.UP, conf.UP_RIGHT, conf.RIGHT, conf.DOWN_RIGHT, conf.DOWN,
        conf.DOWN_LEFT, conf.LEFT, conf.UP_LEFT]

    def __init__(self, x, y, r, mass, color, name, manual_control):
        cell = AgentCell(x, y, r, mass)
        self.cells = [cell]

        self.color = color
        self.name = name
        self.orientation = None  # For deciding direction to shoot in

        self.is_alive = True
        self.manual_control = manual_control
        self.ai_dir = None
        self.ai_steps = 0

    def get_avg_x_pos(self):
        return sum([cell.x_pos for cell in self.cells]) / len(self.cells)

    def get_avg_y_pos(self):
        return sum([cell.y_pos for cell in self.cells]) / len(self.cells)

    def get_avg_radius(self):
        return sum([cell.radius for cell in self.cells]) / len(self.cells)

    def get_mass(self):
        return sum([cell.mass for cell in self.cells])

    def move(self, vel):
        if self.orientation is None:
            return

        for cell in self.cells:
            {
                conf.UP: cell.move_up,
                conf.UP_RIGHT: cell.move_upright,
                conf.RIGHT: cell.move_right,
                conf.DOWN_RIGHT: cell.move_downright,
                conf.DOWN: cell.move_down,
                conf.DOWN_LEFT: cell.move_downleft,
                conf.LEFT: cell.move_left,
                conf.UP_LEFT: cell.move_upleft,
            }[self.orientation](vel)

    def handle_move_keys(self, keys, camera):
        # TODO: better velocity control
        # TODO want vel to be unique per cell...
        vel = int(max(conf.AGENT_STARTING_SPEED - (self.get_mass() * 0.05), 1))

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

        # set orientation
        if is_up:
            if is_left:
                self.orientation = conf.UP_LEFT
            elif is_right:
                self.orientation = conf.UP_RIGHT
            else:
                self.orientation = conf.UP
        elif is_down:
            if is_left:
                self.orientation = conf.DOWN_LEFT
            elif is_right:
                self.orientation = conf.DOWN_RIGHT
            else:
                self.orientation = conf.DOWN
        elif is_left:
            self.orientation = conf.LEFT
        elif is_right:
            self.orientation = conf.RIGHT
        else:
            self.orientation = None

        self.move(vel)

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
        vel = int(max(conf.AGENT_STARTING_SPEED - (self.get_mass() * 0.05), 1))

        # force AI to move between 5 and 10 (inclusive) steps in random direction
        if self.ai_steps <= 0:
            self.ai_steps = np.random.randint(5, 11)
            self.orientation = Agent.orientations[np.random.randint(8)]

        # move in randomly chosen direction
        self.move(vel)
        self.ai_steps -= 1
