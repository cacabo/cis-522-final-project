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
# GameState class
# ------------------------------------------------------------------------------
class GameState():
    def __init__(self):
        self.camera = None
        self.agents = {}
        self.foods = []
        self.viruses = []
        self.time = 0

    def get_time(self):
        return self.time

    def add_food(self, n):
        """
        insert food at random places on the board

        @param n - number of food to spawn
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
        insert viruses at random places on the board

        @param n - number of viruses to spawn
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

    def check_food_collision(self, agent, food):
        """
        if the center of a food is inside the agent, the agent eats it

        @returns None if no cell can eat it
        @returns idx of cell eating the food
        """
        for (idx, cell) in enumerate(agent.cells):
            if self.check_overlap(cell, food):
                print('[FOOD] %s ate food item %s' % (agent.name, food.id))
                return idx

        return None

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
        consumed = []

        for agent_cell in agent.cells:
            for other_cell in other.cells:
                if self.check_cell_collision(agent_cell, other_cell):
                    print('[CELL] %s ate one of %s\'s cells' %
                          (agent.name, other.name))
                    consumed.append((agent_cell, other_cell))
                else:
                    not_consumed.append(other_cell)

        if len(consumed) == 0:
            return

        for (agent_cell, consumed_cell) in consumed:
            agent_cell.mass += consumed_cell.mass
            agent_cell.radius = utils.massToRadius(agent_cell.mass)

        other.cells = not_consumed

        if len(not_consumed) == 0:
            print('[GAME] ' + str(other.name) +
                  ' died! Was eaten by ' + str(agent.name))
            other.is_alive = False

    def handle_virus(self, agent, virus):
        """
        @return None if virus not effected
        @return virus if virus should be deleted
        """
        for cell in agent.cells:
            if not self.check_virus_collision(cell, virus):
                continue
            print('[VIRUS] %s ate virus %s' % (agent.name, virus.id))
            cell.mass += virus.mass
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

    def tick_agent(self, agent):
        # find all food items which are not currently being eaten by this agent, and
        # update global foods list
        food_collisions = [self.check_food_collision(
            agent, food) for food in self.foods]
        foods_remaining = []
        for (idx, cellidx) in enumerate(food_collisions):
            if cellidx is None:
                foods_remaining.append(self.foods[idx])
                continue
            cell = agent.cells[cellidx]
            cell.mass += conf.FOOD_MASS

        self.foods = foods_remaining

        # Iterate over all viruses, remove viruses which were eaten
        removed_viruses_or_none = [self.handle_virus(
            agent, virus) for virus in self.viruses]
        not_removed_viruses = [self.viruses[idx] for (
            idx, virus_or_none) in enumerate(removed_viruses_or_none) if virus_or_none is None]
        self.viruses = not_removed_viruses

        # get a list of all agents which have collided with the current one, and see
        # if it eats any of them
        for other in self.agents.values():
            self.handle_eat_agent(agent, other)

    def tick_game_state(self):
        # make sure food/virus/player mass is balanced on the board
        self.balance_mass()

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

    # check if the user is pressing an exit key
    def is_exit_command(self, event):
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
            text = text_font.render(str(
                idx + 1) + ". " + str(agent.name) + ' (' + str(agent.get_mass()) + ')', 1, (0, 0, 0))
            window.blit(text, (x, start_y + idx * 20))
    
    def main_loop(self):
        window = pygame.display.set_mode((conf.SCREEN_WIDTH, conf.SCREEN_HEIGHT))
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