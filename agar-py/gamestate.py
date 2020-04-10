import pygame
import config as conf
import utils
from food import Food
from virus import Virus
from agent import Agent
from camera import Camera

# ------------------------------------------------------------------------------
# Constants and config
# ------------------------------------------------------------------------------

pygame.init()
text_font = pygame.font.SysFont(
    conf.AGENT_NAME_FONT, conf.AGENT_NAME_FONT_SIZE)

# ------------------------------------------------------------------------------
# Game class
# ------------------------------------------------------------------------------


# ACTION_SPACE = {
#     0: conf.UP,
#     1: conf.UP_RIGHT,
#     2: conf.RIGHT,
#     3: conf.DOWN_RIGHT,
#     4: conf.DOWN,
#     5: conf.DOWN_LEFT,
#     6: conf.LEFT,
#     7: conf.UP_RIGHT,
#     9: "SHOOT",
#     10: "SPLIT",
# }

class GameState():
    def __init__(self):
        self.camera = None
        self.agents = {}
        self.foods = []
        self.viruses = []
        self.masses = []
        self.time = 0

    def get_player_names(self):
        return list(self.agents.keys())

    def get_time(self):
        return self.time

    def add_mass(self, mass):
        self.masses.append(mass)

    def add_food(self, n):
        """
        Insert food at random places on the board

        Parameters

            n : number of food to spawn
        """
        if n is None or n <= 0:
            raise Exception('n must be positive')

        radius = utils.massToRadius(conf.FOOD_MASS)
        for _ in range(n):
            # TODO: could include uniform distribution here
            pos = utils.randomPosition(radius)
            self.foods.append(Food(pos[0], pos[1], radius, (255, 0, 0)))

    def add_virus(self, n):
        """
        Insert viruses at random places on the board

        Parameters

            n : number of viruses to spawn
        """
        if n is None or n <= 0:
            raise Exception('n must be positive')

        radius = utils.massToRadius(conf.VIRUS_MASS)
        for _ in range(n):
            # TODO: could include uniform distribution here
            pos = utils.randomPosition(radius)
            self.viruses.append(Virus(pos[0], pos[1], radius, conf.VIRUS_MASS))

    def balance_mass(self):
        """ensure that the total mass of the game is balanced between food and players"""
        total_food_mass = len(self.foods) * conf.FOOD_MASS
        total_agent_mass = sum([agent.get_mass()
                                for agent in self.agents.values()])
        total_mass = total_food_mass + total_agent_mass
        mass_diff = conf.GAME_MASS - total_mass
        max_num_food_to_add = conf.MAX_FOOD - len(self.foods)
        max_mass_food_to_add = mass_diff / conf.FOOD_MASS

        num_food_to_add = min(max_num_food_to_add, max_mass_food_to_add)
        if num_food_to_add > 0:
            self.add_food(num_food_to_add)

        # TODO removing food if necessary

        num_virus_to_add = conf.MAX_VIRUSES - len(self.viruses)

        if num_virus_to_add > 0:
            self.add_virus(num_virus_to_add)

    def check_overlap(self, a, b):
        """
        Check if two generic objects with `get_pos` function and `radius`
        properties overlap with each other

        @returns boolean
        """
        return utils.isPointInCircle(b.get_pos(), a.get_pos(), a.radius)

    def check_cell_collision(self, agent_cell, other_cell):
        if agent_cell.mass < other_cell.mass * conf.CELL_CONSUME_MASS_FACTOR:
            return False
        return self.check_overlap(agent_cell, other_cell)

    def check_virus_collision(self, agent_cell, virus):
        if agent_cell.mass < conf.VIRUS_CONSUME_MASS_FACTOR * virus.mass:
            return False
        return self.check_overlap(agent_cell, virus)

    def check_food_collision(self, agent_cell, food):
        return self.check_overlap(agent_cell, food)

    def handle_eat_agent(self, agent, other):
        """
        Agent eats other if:

        1. it has mass greater by at least CONSUME_MASS_FACTOR, and

        2. the agent's circle overlaps with the center of other

        @return boolean
        """
        if agent.name == other.name:
            return

        not_consumed = []
        # consumed = []

        for agent_cell in agent.cells:
            for other_cell in other.cells:
                if self.check_cell_collision(agent_cell, other_cell):
                    print(self.time + ' [CELL] %s ate one of %s\'s cells' %
                          (agent.name, other.name))
                    # consumed.append((agent_cell, other_cell))
                    agent_cell.set_mass(agent_cell.mass + other_cell.mass)
                else:
                    not_consumed.append(other_cell)

        other.cells = not_consumed

        if len(not_consumed) == 0:
            print('[GAME] ' + str(other.name) +
                  ' died! Was eaten by ' + str(agent.name))
            other.is_alive = False

    def handle_food(self, agent, food):
        for cell in agent.cells:
            if not self.check_food_collision(cell, food):
                continue
            print('[FOOD] %s ate food item %s' %
                  (agent.name, food.id))
            cell.mass += food.mass
            return food

    def handle_mass(self, agent, mass):
        for cell in agent.cells:
            if not self.check_cell_collision(cell, mass):
                continue
            print('[MASS] %s ate mass %s' % (agent.name, mass.id))
            cell.mass += mass.mass
            return mass

    def handle_virus(self, agent, virus):
        """
        @return None if virus not effected
        @return virus if virus should be deleted
        """
        for cell in agent.cells:
            if not self.check_virus_collision(cell, virus):
                continue
            print('[VIRUS] %s ate virus %s' % (agent.name, virus.id))
            cell.eat_virus(virus)

            # Return early without considering other cells
            # That is, the virus can only be eaten once
            return virus

        return None

    def init_model_agent(self, model):
        """
        Initialize a game agent for the given learning model
        @param model - the learning model to create an agent for
        """
        if model is None:
            raise ValueError('asked to initialize agent for None model')

        radius = utils.massToRadius(conf.AGENT_STARTING_MASS)
        pos = utils.randomPosition(radius)
        # TODO: make model name better, maybe give ID to Agent() instead
        model_agent = Agent(
            self,
            pos[0],
            pos[1],
            radius,
            mass=conf.AGENT_STARTING_MASS,
            color=conf.BLUE_COLOR,
            name='Agent' + str(model.id),
            manual_control=False
        )
        self.agents[model.id] = model_agent

    def _filter_objects(self, agent, arr, handler):
        """
        Parameters:

            agent   - current agent
            arr     - list of objects the agent might interact with
            handler - takes in `agent` and items from `arr` on by one, returns
                      the object if it should be removed else returns None
        """
        obj_or_none = [handler(
            agent, obj) for obj in arr]
        not_removed_objs = [arr[idx] for (
            idx, obj_or_none) in enumerate(obj_or_none) if obj_or_none is None]
        return not_removed_objs

    def tick_agent(self, agent):
        # find all food items which are not currently being eaten by this agent, and
        # update global foods list
        self.foods = self._filter_objects(
            agent, self.foods, self.handle_food)

        # Iterate over all masses, remove those which were eaten
        self.masses = self._filter_objects(
            agent, self.masses, self.handle_mass)

        # Iterate over all viruses, remove viruses which were eaten
        self.viruses = self._filter_objects(
            agent, self.viruses, self.handle_virus)

        # get a list of all agents which have collided with the current one, and see
        # if it eats any of them
        for other in self.agents.values():
            self.handle_eat_agent(agent, other)

    def tick_game_state(self):
        # make sure food/virus/player mass is balanced on the board
        self.balance_mass()

        # move all mass
        for mass in self.masses:
            if mass.is_moving():
                mass.move()

        # check results of all agent actions
        # TODO: get reward experienced by each agent depending on what happens to them
        for agent in self.agents.values():
            self.tick_agent(agent)

        # after ticking all the agents, remove the dead ones
        dead_agents = [agent for agent in self.agents.values()
                       if not agent.is_alive]
        for dead_agent in dead_agents:
            del self.agents[dead_agent.name]

        self.time += 1

        # TODO: return reward experienced by each agent depending on what happens to them
        return None

    # ------------------------------------------------------------------------------
    # Methods for interfacing with learning models
    # ------------------------------------------------------------------------------
    # reset the game to its initial state, and initialize a game agent for each model
    def reset(self, models):
        self.__init__()
        for model in models:
            self.init_model_agent(model)

    # get the current game state
    def get_state(self):
        return self.agents, self.foods, self.viruses, self.time

    # update the game state based on actions taken by models
    def update_game_state(self, models, actions):
        # first, update the current game state by performing each model's selected
        # action with its agent
        for (model, action) in zip(models, actions):
            agent = self.agents[model.id]
            agent.do_action(action)

        rewards = self.tick_game_state()

        # TODO: go through rewards to see if anyone died and if so, add "done"
        return [None for model in models], False

    # ------------------------------------------------------------------------------
    # Methods for playing the game in interactive mode
    # ------------------------------------------------------------------------------
    # TODO: rename. only used for interactive mode
    def update_agent_state(self, agent):
        if agent.manual_control:
            # get key presses
            keys = pygame.key.get_pressed()
            agent.handle_move_keys(keys, self.camera)
            agent.handle_other_keys(keys, self.camera)
            agent.handle_merge()
        else:
            agent.ai_move()

    def init_manual_agent(self, name):
        radius = utils.massToRadius(conf.AGENT_STARTING_MASS)
        pos = utils.randomPosition(radius)
        player = Agent(
            self,
            pos[0],
            pos[1],
            radius,
            mass=conf.AGENT_STARTING_MASS,
            color=conf.GREEN_COLOR,
            name=name,
            manual_control=True,
        )
        self.agents[player.name] = player
        self.camera = Camera((conf.SCREEN_WIDTH / 2 - player.get_avg_x_pos()),
                             (conf.SCREEN_HEIGHT / 2 - player.get_avg_y_pos()),
                             player.get_avg_radius())

    def init_ai_agents(self, num_agents):
        """
        Create agents which have self-contained strategies

        @param num_agents - how many agents to create
        """
        if num_agents is None or num_agents <= 0:
            raise Exception('num_agents must be positive')

        for i in range(num_agents):
            radius = utils.massToRadius(conf.AGENT_STARTING_MASS)
            pos = utils.randomPosition(radius)
            ai_agent = Agent(
                self,
                pos[0],
                pos[1],
                radius,
                mass=conf.AGENT_STARTING_MASS,
                color=conf.BLUE_COLOR,
                name='Agent' + str(i),
                manual_control=False,
            )
            self.agents[ai_agent.name] = ai_agent

    def is_exit_command(self, event):
        """
        check if the user is pressing an exit key
        """
        return event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)

    def draw_circle(self, board, obj, color=None, stroke=None):
        x, y = obj.get_pos()
        pos = (int(round(x)), int(round(y)))
        radius = int(round(obj.radius))
        if stroke is not None:
            pygame.draw.circle(board, color, pos, radius, stroke)
        else:
            pygame.draw.circle(board, color, pos, radius)

    def draw_window(self, board, window):
        # fill screen white, to clear old frames
        window.fill(conf.WHITE_COLOR)
        board.fill(conf.WHITE_COLOR)

        for mass in self.masses:
            self.draw_circle(board, mass, color=mass.color)

        # TODO don't redraw everything?
        for food in self.foods:
            self.draw_circle(board, food, color=food.color)

        for agent in sorted(self.agents.values(), key=lambda a: a.get_mass()):
            for cell in agent.cells:
                self.draw_circle(board, cell, color=agent.color)
                agent_name_text = text_font.render(agent.name, 1, (0, 0, 0))
                board.blit(agent_name_text, (cell.x_pos - (agent_name_text.get_width() / 2),
                                             cell.y_pos - (agent_name_text.get_height() / 2)))

        for virus in self.viruses:
            self.draw_circle(board, virus, color=conf.VIRUS_COLOR)
            self.draw_circle(
                board, virus, color=conf.VIRUS_OUTLINE_COLOR, stroke=4)

        window.blit(board, self.camera.get_pos())

        # draw leaderboard
        sorted_agents = list(
            reversed(sorted(self.agents.values(), key=lambda x: x.get_mass())))
        leaderboard_title = text_font.render("Leaderboard", 1, (0, 0, 0))
        start_y = 25
        x = conf.SCREEN_WIDTH - leaderboard_title.get_width() - 20
        window.blit(leaderboard_title, (x, 5))
        top_n = min(len(self.agents), conf.NUM_DISPLAYED_ON_LEADERBOARD)
        for idx, agent in enumerate(sorted_agents[:top_n]):
            score = int(round(agent.get_mass()))
            text = text_font.render(
                str(idx + 1) + ". " + str(agent.name) + ' (' + str(score) + ')', 1, (0, 0, 0))
            window.blit(text, (x, start_y + idx * 20))

    def main_loop(self):
        window = pygame.display.set_mode(
            (conf.SCREEN_WIDTH, conf.SCREEN_HEIGHT))
        pygame.display.set_caption('CIS 522: Final Project')
        board = pygame.Surface((conf.BOARD_WIDTH, conf.BOARD_HEIGHT))
        clock = pygame.time.Clock()
        running = True
        while running:
            clock.tick(conf.CLOCK_TICK)
            for agent in self.agents.values():
                self.update_agent_state(agent)

            self.tick_game_state()

            # take in user input and draw/update the game board
            for event in pygame.event.get():
                # stop the game if user exits
                if self.is_exit_command(event):
                    running = False

            # redraw window then update the frame
            self.draw_window(board, window)
            pygame.display.update()

        pygame.quit()
        quit()
