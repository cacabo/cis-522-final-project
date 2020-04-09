import pygame
import numpy as np
import config as conf
import utils
import math

NORMAL_MODE = 'normal'
SHOOTING_MODE = 'shooting'


class AgentCell():
    def __init__(self, x, y, radius=None, mass=None, mode=NORMAL_MODE):
        """
        An AgentCell is a single cell of an Agent

        @param x
        @param y
        @param radius
        @param mass
        @param mode
        """
        self.x_pos = int(x)  # NOTE pygame expects ints
        self.y_pos = int(y)

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
        if mass is None or mass <= 0:
            raise Exception('Mass must be positive')

        self.mass = int(mass)
        self.radius = utils.massToRadius(mass)

    def move_left(self, vel=None):
        vel = vel if vel is not None else self.get_velocity()
        self.x_pos = max(self.x_pos - vel, self.radius)

    def move_right(self, vel=None):
        vel = vel if vel is not None else self.get_velocity()
        self.x_pos = min(self.x_pos + vel, conf.BOARD_WIDTH - self.radius)

    def move_up(self, vel=None):
        vel = vel if vel is not None else self.get_velocity()
        self.y_pos = max(self.y_pos - vel, self.radius)

    def move_down(self, vel=None):
        vel = vel if vel is not None else self.get_velocity()
        self.y_pos = min(self.y_pos + vel, conf.BOARD_HEIGHT - self.radius)

    def move_upleft(self, vel=None):
        vel = vel if vel is not None else self.get_velocity()
        sqrt_vel = math.sqrt(vel)
        self.move_up(sqrt_vel)
        self.move_left(sqrt_vel)

    def move_upright(self, vel=None):
        vel = vel if vel is not None else self.get_velocity()
        self.move_up(vel)
        self.move_right(vel)

    def move_downleft(self, vel=None):
        vel = vel if vel is not None else self.get_velocity()
        sqrt_vel = math.sqrt(vel)
        self.move_down(sqrt_vel)
        self.move_left(sqrt_vel)

    def move_downright(self, vel=None):
        vel = vel if vel is not None else self.get_velocity()
        sqrt_vel = math.sqrt(vel)
        self.move_down(sqrt_vel)
        self.move_right(sqrt_vel)

    def shoot(self, orientation):
        self.mode = SHOOTING_MODE
        self.shooting_orientation = orientation
        self.shooting_velocity = self.radius * 2
        self.shooting_acceleration = self.radius / 2

    def move_shoot(self):
        """
        Move in response to being shot
        """
        self.move_normal(self.shooting_orientation, self.shooting_velocity)
        self.shooting_velocity = self.shooting_velocity - self.shooting_acceleration

        if self.shooting_velocity <= 0:
            # Change the mode
            self.mode = NORMAL_MODE

            # Clean out shooting state
            self.shooting_acceleration = None
            self.shooting_velocity = None
            self.shooting_acceleration = None

    def move_normal(self, orientation, vel):
        {
            conf.UP: self.move_up,
            conf.UP_RIGHT: self.move_upright,
            conf.RIGHT: self.move_right,
            conf.DOWN_RIGHT: self.move_downright,
            conf.DOWN: self.move_down,
            conf.DOWN_LEFT: self.move_downleft,
            conf.LEFT: self.move_left,
            conf.UP_LEFT: self.move_upleft,
        }[orientation](vel)

    def move(self, orientation, vel):
        """
        Move in the direction specified by `orientation`

        If `mode` is `shooting`, move behavior gets overriden

        @param orientation
        @param vel
        """
        if self.mode == SHOOTING_MODE:
            self.move_shoot()
        else:
            self.move_normal(orientation, vel)

    def get_pos(self):
        return (self.x_pos, self.y_pos)


class Agent():
    orientations = [
        conf.UP, conf.UP_RIGHT, conf.RIGHT, conf.DOWN_RIGHT, conf.DOWN,
        conf.DOWN_LEFT, conf.LEFT, conf.UP_LEFT]

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

        cell = AgentCell(x, y, radius=radius, mass=mass)
        self.cells = [cell]

        self.color = color
        self.name = name
        self.orientation = None  # For deciding direction to shoot in

        self.is_alive = True
        self.manual_control = manual_control
        self.ai_dir = None
        self.ai_steps = 0
        self.last_split = self.game.get_time()

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
        if self.orientation is None:
            return

        avg_x = self.get_avg_x_pos()
        avg_y = self.get_avg_y_pos()
        dest_radius = 800
        sqrt_dest_radius = math.sqrt(dest_radius)

        if self.orientation == conf.UP:
            dest_x = avg_x
            dest_y = avg_y + dest_radius
        elif self.orientation == conf.UP_RIGHT:
            dest_x = avg_x + sqrt_dest_radius
            dest_y = avg_y + sqrt_dest_radius
        elif self.orientation == conf.RIGHT:
            dest_x = avg_x + dest_radius
            dest_y = avg_y
        elif self.orientation == conf.DOWN_RIGHT:
            dest_x = avg_x + sqrt_dest_radius
            dest_y = avg_y - sqrt_dest_radius
        elif self.orientation == conf.DOWN:
            dest_x = avg_x
            dest_y = avg_y - dest_radius
        elif self.orientation == conf.DOWN_LEFT:
            dest_x = avg_x - sqrt_dest_radius
            dest_y = avg_y - sqrt_dest_radius
        elif self.orientation == conf.LEFT:
            dest_x = avg_x - dest_radius
            dest_y = avg_y
        elif self.orientation == conf.UP_LEFT:
            dest_x = avg_x - sqrt_dest_radius
            dest_y = avg_y + sqrt_dest_radius

        for cell in self.cells:
            cell.move(self.orientation, vel)
            # TODO have some notion of penalty?
            # TODO also handle cells overlapping with each other

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
        # else:
        #     self.orientation = None

        self.move()

        camera.pan(self.get_avg_x_pos(), self.get_avg_y_pos())

    def handle_shoot(self):
        raise Exception('Not implemented')
        # TODO: this crashes, needs to be handled for each specific split cell
        # if self.mass < conf.MIN_MASS_TO_SHOOT:
        #     return

        # self.mass = self.mass - conf.SHOOT_MASS

        # TODO shoot
        # return

    def handle_merge(self):
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
                avg_x_pos, avg_y_pos, radius=None, mass=merged_mass)
            merged_cells.append(merged_cell)

        if len(self.cells) % 2 == 1:
            # Append last cell if there are an odd number
            merged_cells.append(self.cells[-1])

        self.cells = merged_cells

    def handle_split(self):
        print('[AGENT] handle split')
        if self.orientation is None:
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

        for cell in self.cells:
            cell.set_mass(cell.mass / 2)

        new_cells = []
        for cell in self.cells:
            new_cell = AgentCell(cell.x_pos, cell.y_pos,
                                 cell.radius, cell.mass)
            new_cell.shoot(self.orientation)
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
        #vel = int(max(conf.AGENT_STARTING_SPEED - (self.get_mass() * 0.05), 1))
        vel = 1
        if self.get_mass() > 0:
            vel = max(utils.massToVelocity(self.get_mass()), vel)

        # force AI to move between 5 and 10 (inclusive) steps in random direction
        if self.ai_steps <= 0:
            self.ai_steps = np.random.randint(5, 11)
            self.orientation = Agent.orientations[np.random.randint(8)]

        # move in randomly chosen direction
        self.move(vel)
        self.ai_steps -= 1

    def act(self, action):
        #TODO: parse actions
        return