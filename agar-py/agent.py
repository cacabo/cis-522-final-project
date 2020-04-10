import pygame
import numpy as np
import config as conf
import utils
import math
import random
from mass import Mass

NORMAL_MODE = 'normal'
SHOOTING_MODE = 'shooting'


class AgentCell():
    def __init__(self, agent, x, y, radius=None, mass=None, mode=NORMAL_MODE):
        """
        An AgentCell is a single cell of an Agent

        Parameters:

            agent  - pointer to agent
            x      - x position
            y      - y position
            radius - optional radius of the cell
            mass   - mass of the cell
            mode   - either NORMAL_MODE or SPLITTING_MODE
        """
        self.agent = agent
        self.x_pos = x
        self.y_pos = y

        self.mass = mass
        self.mode = mode

        if radius is not None:
            self.radius = radius
        else:
            self.radius = utils.massToRadius(mass)

    def get_velocity(self):
        # return int(max(conf.AGENT_STARTING_SPEED - (self.mass * 0.05), 1))
        if self.mass > 0:
            return max(utils.massToVelocity(self.mass), 1)
        else:
            # TODO what is this case for?
            return 1

    def set_mass(self, mass):
        """
        Setter method for the mass

        Also updates AgentCell radius
        """
        if mass is None or mass <= 0:
            raise Exception('Mass must be positive')

        self.mass = mass
        self.radius = utils.massToRadius(mass)

    def split(self):
        """
        Split this cell and return the newly created cell
        """
        self.set_mass(self.mass / 2)
        new_cell = AgentCell(self.agent, self.x_pos, self.y_pos,
                             self.radius, self.mass)
        return new_cell

    def eat_virus(self, virus):
        if virus is None:
            raise Exception('Cannot eat virus which is None')

        self.mass += virus.mass

        # TODO
        max_cells_based_on_count = conf.AGENT_CELL_LIMIT - \
            len(self.agent.cells) + 1
        max_cells_based_on_size = int(self.mass / (conf.MIN_MASS_TO_SPLIT / 2))
        num_cells_to_split_into = min(
            max_cells_based_on_count, max_cells_based_on_size)

        new_cells = []

        new_mass = self.mass / num_cells_to_split_into
        self.mass = new_mass

        for i in range(1, num_cells_to_split_into):
            new_cell = AgentCell(self.agent, self.x_pos,
                                 self.y_pos, mass=new_mass)
            new_cells.append(new_cell)

        self.agent.cells = self.agent.cells + new_cells
        self.agent.last_split = self.agent.game.get_time()

    def shoot(self, angle):
        self.mode = SHOOTING_MODE
        self.shooting_angle = angle
        self.shooting_velocity = self.radius * 2
        self.shooting_acceleration = self.radius / 2

    def move_shoot(self):
        """
        Move in response to being shot
        """
        utils.moveObject(self, self.shooting_angle, self.shooting_velocity)
        self.shooting_velocity = self.shooting_velocity - self.shooting_acceleration

        if self.shooting_velocity <= 0:
            # We are done being controlled by acceleration and can be controlled
            # by agent decisions

            # Change the mode
            self.mode = NORMAL_MODE

            # Clean out shooting state
            self.shooting_acceleration = None
            self.shooting_velocity = None
            self.shooting_acceleration = None

    def move(self, angle, vel):
        """
        Move in the direction specified by `angle` from the x axis in pos dir

        If `mode` is `shooting`, move behavior gets overriden

        @param angle
        @param vel
        """
        if self.mode == SHOOTING_MODE:
            self.move_shoot()
        else:
            vel = vel if vel is not None else self.get_velocity()
            utils.moveObject(self, angle, vel)

    def shift(self, dx=None, dy=None):
        """
        Adjust position by dx and dy

        NOTE does not check for collisions, borders, etc.

        @param dx
        @param dy
        """
        if dx is not None:
            self.x_pos += dx
        if dy is not None:
            self.y_pos += dy

    def get_pos(self):
        return (self.x_pos, self.y_pos)


class Agent():
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    def __init__(self, game, x, y, radius, mass=None, color=None, name=None, manual_control=False):
        """
        An `Agent` is a player in the `Game`. An `Agent` can have many
        `AgentCells` (just one to start out with).

        @param game          - game that this `Agent` belongs to
        @param x
        @param y
        @param radius
        @param mass
        @param color
        @param name           - unique ID for the agent, displayed on the game
        @param manual_control - if should be controlled by user's keyboard
        """
        self.game = game

        self.color = color
        self.name = name
        self.angle = None  # For deciding direction to move in

        self.is_alive = True
        self.manual_control = manual_control
        self.ai_dir = None
        self.ai_steps = 0
        self.last_split = self.game.get_time()

        cell = AgentCell(self, x, y, radius=radius, mass=mass)
        self.cells = [cell]

    def get_avg_x_pos(self):
        """
        @returns average x pos of all `AgentCells` belonging to this `Agent`
        """
        return sum([cell.x_pos for cell in self.cells]) / len(self.cells)

    def get_avg_y_pos(self):
        """
        @returns average y pos of all `AgentCells` belonging to this `Agent`
        """
        return sum([cell.y_pos for cell in self.cells]) / len(self.cells)

    def get_avg_radius(self):
        """
        @returns average radius of all `AgentCells` belonging to this `Agent`
        """
        return sum([cell.radius for cell in self.cells]) / len(self.cells)

    def get_mass(self):
        """
        @returns summed mass of all `AgentCells` belonging to this `Agent`
        """
        return sum([cell.mass for cell in self.cells])

    def move(self, vel=None):
        if self.angle is None:
            return

        avg_x = self.get_avg_x_pos()
        avg_y = self.get_avg_y_pos()

        for (idx, cell) in enumerate(self.cells):
            # Handle converging towards the middle
            penalty = -2  # Move this many pixels towards the center
            angle_to_avg = utils.getAngleBetweenPoints(
                (avg_x, avg_y), cell.get_pos())

            if angle_to_avg is not None:
                cell.move(angle_to_avg, penalty)

            # Handle overlapping cells
            for otherIdx in range(idx + 1, len(self.cells)):
                otherCell = self.cells[otherIdx]
                overlap = utils.getObjectOverlap(cell, otherCell)
                if overlap < 0:
                    continue
                dist_to_move = overlap / 2
                angle = utils.getAngleBetweenObjects(cell, otherCell)
                if angle is None:
                    angle = random.randrange(360)
                cell.move(angle, -1 * dist_to_move)
                otherCell.move(angle, dist_to_move)

            # Handle normal movement
            cell.move(self.angle, vel)

    def handle_move_keys(self, keys, camera):
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

        # set angle
        if is_up:
            if is_left:
                self.angle = 135
            elif is_right:
                self.angle = 45
            else:
                self.angle = 90
        elif is_down:
            if is_left:
                self.angle = 225
            elif is_right:
                self.angle = 315
            else:
                self.angle = 270
        elif is_left:
            self.angle = 180
        elif is_right:
            self.angle = 0
        # else:
        #     self.angle = None

        self.move()

        camera.pan(self.get_avg_x_pos(), self.get_avg_y_pos())

    def handle_shoot(self):
        # You can only shoot if you are a single cell
        if len(self.cells) > 1:
            return

        if self.get_mass() < conf.MIN_MASS_TO_SHOOT:
            return

        # Must be moving in a direction in order to shoot
        if self.angle is None:
            return

        cell = self.cells[0]
        cell.mass = cell.mass - conf.MASS_MASS

        (mass_x, mass_y) = cell.get_pos()
        mass = Mass(mass_x, mass_y, self.color, self.angle, cell.radius)
        self.game.add_mass(mass)

    def handle_merge(self):
        # TODO merge with actual sibling cell if possible, not just arbitrary index?
        # TODO normally this should only happen if the user moves cells near each other
        if len(self.cells) < 2:
            return

        curr_time = self.game.get_time()
        if curr_time < self.last_split + conf.AGENT_TICKS_TO_MERGE_CELLS:
            return

        # Merge pairs of cells in the body
        self.last_split = curr_time
        merged_cells = []
        mid_idx = int(len(self.cells) / 2)
        for idx in range(0, mid_idx):
            cell = self.cells[idx]
            other_cell = self.cells[idx + mid_idx]
            avg_x_pos = (cell.x_pos + other_cell.x_pos) / 2
            avg_y_pos = (cell.y_pos + other_cell.y_pos) / 2
            merged_mass = cell.mass + other_cell.mass
            merged_cell = AgentCell(
                self, avg_x_pos, avg_y_pos, radius=None, mass=merged_mass)
            merged_cells.append(merged_cell)

        if len(self.cells) % 2 == 1:
            # Append last cell if there are an odd number
            merged_cells.append(self.cells[-1])

        self.cells = merged_cells

    def handle_split(self):
        print('[AGENT] handle split')
        if self.angle is None:
            return
        if len(self.cells) * 2 >= conf.AGENT_CELL_LIMIT:
            # Limit the nubmer of cells that an agent can be in
            return

        curr_time = self.game.get_time()
        if curr_time < self.last_split + conf.AGENT_TICKS_TO_SPLIT_AGAIN:
            return

        for cell in self.cells:
            # Each cell needs to be at least a certain size in order to split
            if cell.mass < conf.MIN_MASS_TO_SPLIT:
                return

        new_cells = []
        for cell in self.cells:
            new_cell = cell.split()
            new_cell.shoot(self.angle)
            new_cells.append(new_cell)

        self.cells = self.cells + new_cells
        self.last_split = self.game.get_time()

    def handle_other_keys(self, keys, camera):
        is_split = keys[pygame.K_SPACE]
        is_shoot = keys[pygame.K_q]

        if is_split:
            self.handle_split()
        elif is_shoot:
            self.handle_shoot()

    def ai_move(self):
        # TODO: better velocity control
        # vel = max(conf.AGENT_STARTING_SPEED - (self.get_mass() * 0.05), 1)
        vel = 1
        if self.get_mass() > 0:
            vel = max(utils.massToVelocity(self.get_mass()), vel)

        # force AI to move between 5 and 10 (inclusive) steps in random direction
        if self.ai_steps <= 0:
            self.ai_steps = np.random.randint(5, 11)
            self.angle = Agent.angles[np.random.randint(8)]

        # move in randomly chosen direction
        self.move(vel)
        self.ai_steps -= 1

    def act(self, action):
        # TODO: parse actions
        return
