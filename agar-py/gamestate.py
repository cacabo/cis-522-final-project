import pygame
import config as conf
import utils
from food import Food
from virus import Virus
from agent import Agent
from camera import Camera
from models.DQNModel import DQNModel

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
    ID_counter = 0

    def __init__(self):
        self.camera = None
        self.agents = {}
        self.foods = []
        self.viruses = []
        self.masses = []
        self.time = 0
        self.board = None
        self.window = None

        # set up the game state with balanced mass
        self.balance_mass()

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

        radius = utils.mass_to_radius(conf.FOOD_MASS)
        for _ in range(n):
            pos = utils.gen_non_overlap_position(self.agents.values(), radius)
            self.foods.append(Food(pos[0], pos[1], radius, (255, 0, 0)))

    def add_virus(self, n):
        """
        Insert viruses at random places on the board

        Parameters

            n : number of viruses to spawn
        """
        if n is None or n <= 0:
            raise Exception('n must be positive')

        radius = utils.mass_to_radius(conf.VIRUS_MASS)
        for _ in range(n):
            pos = utils.gen_non_overlap_position(self.agents.values(), radius)
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
        return utils.is_point_in_circle(b.get_pos(), a.get_pos(), a.radius)

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
        if agent == other or agent.name == other.name:
            return 0

        # consumed = []
        mass_consumed = 0

        for agent_cell in agent.cells:
            for other_cell in other.cells:
                if not other_cell.is_alive:
                    continue
                elif self.check_cell_collision(agent_cell, other_cell):
                    print('[%s] [CELL] %s ate one of %s\'s cells' %
                          (self.get_time(), agent.name, other.name))
                    agent_cell.set_mass(agent_cell.mass + other_cell.mass)
                    mass_consumed += other_cell.mass
                    other_cell.is_alive = False

        other.cells = [cell for cell in other.cells if cell.is_alive]

        if len(other.cells) == 0:
            print('[%s] [GAME] %s died! Was eaten by %s' %
                  (self.get_time(), other.name, agent.name))
            other.is_alive = False
        return mass_consumed

    def handle_food(self, agent, food):
        for cell in agent.cells:
            if not self.check_food_collision(cell, food):
                continue
            print('[%s] [FOOD] %s ate food item %s' %
                  (self.get_time(), agent.name, food.id))
            cell.eat_food(food)
            return food

    def handle_mass(self, agent, mass):
        for cell in agent.cells:
            if not self.check_cell_collision(cell, mass):
                continue
            print('[%s] [MASS] %s ate mass %s' %
                  (self.get_time(), agent.name, mass.id))
            cell.eat_mass(mass)
            return mass

    def handle_virus(self, agent, virus):
        """
        @return None if virus not effected
        @return virus if virus should be deleted
        """
        new_cells = []
        ate_virus = False
        for cell in agent.cells:
            if not virus.is_alive or not self.check_virus_collision(cell, virus):
                continue
            print('[%s] [VIRUS] %s ate virus %s' %
                  (self.get_time(), agent.name, virus.id))
            new_cells = cell.eat_virus(virus)
            ate_virus = True
            break

        # Return early without considering other cells
        # That is, the virus can only be eaten once
        # return virus
        if ate_virus:
            agent.cells = agent.cells + new_cells
            return virus

        return None

    def init_model_agent(self, model):
        """
        Initialize a game agent for the given learning model
        @param model - the learning model to create an agent for
        """
        if model is None:
            raise ValueError('asked to initialize agent for None model')

        radius = utils.mass_to_radius(conf.AGENT_STARTING_MASS)
        pos = utils.gen_random_position(radius)
        # TODO: make model name better, maybe give ID to Agent() instead
        model_agent = Agent(
            self,
            model,
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
                      April 10th: returns what remains, and an mask of None (for remaining objs)/objs
        """
        obj_or_none = [handler(
            agent, obj) for obj in arr]
        not_removed_objs = [arr[idx] for (
            idx, obj_or_none) in enumerate(obj_or_none) if obj_or_none is None]
        return (not_removed_objs, obj_or_none)

    def tick_agent(self, agent):
        """
        Have the provided agent eat all food, mass, virus, and other agents
        which it is capable of eating. Update global game state accordingly.

        Parameters

            agent : Agent

        Returns

            void
        """

        # find all food items which are not currently being eaten by this agent, and
        # update global foods list
        remaining_food, food_eaten_or_none = self._filter_objects(
            agent, self.foods, self.handle_food)
        self.foods = remaining_food
        num_food_eaten = len(
            list(filter(lambda x: x != None, food_eaten_or_none)))

        # Iterate over all masses, remove those which were eaten
        remaining_mass, mass_eaten_or_none = self._filter_objects(
            agent, self.masses, self.handle_mass)
        self.masses = remaining_mass
        num_mass_eaten = len(
            list(filter(lambda x: x != None, mass_eaten_or_none)))

        # Iterate over all viruses, remove viruses which were eaten
        remaining_virus, virus_eaten_or_none = self._filter_objects(
            agent, self.viruses, self.handle_virus)
        self.viruses = remaining_virus
        num_virus_eaten = len(
            list(filter(lambda x: x != None, virus_eaten_or_none)))

        # get a list of all agents which have collided with the current one, and see
        # if it eats any of them
        agent_mass_eaten = 0
        for other in self.agents.values():
            agent_mass_eaten += self.handle_eat_agent(agent, other)
        return agent_mass_eaten + conf.FOOD_MASS * num_food_eaten + conf.VIRUS_MASS * num_virus_eaten + conf.MASS_MASS * num_mass_eaten

    def tick_game_state(self, models):
        # make sure food/virus/player mass is balanced on the board
        self.balance_mass()

        # move all mass
        for mass in self.masses:
            if mass.is_moving():
                mass.move()

        # check results of all agent actions
        # TODO: get reward experienced by each agent depending on what happens to them
        if models == None:
            for agent in self.agents.values():
                self.tick_agent(agent)
        else:
            rewards = []
            for model in models:
                if model.id in self.agents:
                    agent = self.agents[model.id]
                    rewards.append(self.tick_agent(agent))
                else:
                    rewards.append(0)

        # after ticking all the agents, remove the dead ones
        dead_agents = [key for key, agent in self.agents.items()
                       if not agent.is_alive]
        for dead_agent in dead_agents:
            # TODO: handle this better. interactive player is keyed on name, rest are keyed on model ID
            # if dead_agent.manual_control:
            #     del self.agents[dead_agent.name]
            # else:
            #     del self.agents[dead_agent.model.id]
            del self.agents[dead_agent]

        if models:
            dones = []
            for (idx, model) in enumerate(models):
                if model.id in self.agents:
                    dones.append(False)
                    rewards[idx] += conf.SURVIVAL_REWARD
                else:
                    dones.append(True)
                    rewards[idx] = conf.DEATH_REWARD

        self.time += 1

        # TODO: return reward experienced by each agent depending on what happens to them
        if models:
            return (rewards, dones)

    # ------------------------------------------------------------------------------
    # Methods for interfacing with learning models
    # ------------------------------------------------------------------------------

    # reset the game to its initial state, and initialize a game agent for each model
    def reset(self, models):
        self.__init__()
        for model in models:
            self.init_model_agent(model)

    def get_state(self):
        """get the current game state"""
        return self.agents, self.foods, self.viruses, self.time

    def get_board_state(self):
        """get game board image"""
        state = pygame.surfarray.array3d(self.window)
        return state

    def update_game_state(self, models, actions):
        """update the game state based on actions taken by models"""

        # first, update the current game state by performing each model's selected
        # action with its agent
        for (model, action) in zip(models, actions):
            if model.id in self.agents:
                agent = self.agents[model.id]
                agent.do_action(action)

        # TODO: abstract away computation of rewards?
        rewards, dones = self.tick_game_state(models)

        return rewards, dones

        # TODO: go through rewards to see if anyone died and if so, add "done"
        # return [0 for model in models], [False for model in models]

    # ------------------------------------------------------------------------------
    # Methods for playing the game in interactive mode
    # ------------------------------------------------------------------------------

    def update_interactive_state(self, agent):
        """NOTE this is only used for interactive mode"""
        if agent.manual_control:
            # get key presses
            keys = pygame.key.get_pressed()
            agent.handle_move_keys(keys, self.camera)
            agent.handle_other_keys(keys, self.camera)
        else:
            if (isinstance(agent.model, DQNModel)):
                agent.act(self.get_board_state())
            else:
                agent.act(self.get_state())

        agent.handle_merge()

    def init_manual_agent(self, name):
        radius = utils.mass_to_radius(conf.AGENT_STARTING_MASS)
        pos = utils.gen_random_position(radius)
        player = Agent(
            self,
            None,
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

    def init_ai_agents(self, num_agents, model, name=None):
        """
        Create agents which have self-contained strategies

        @param num_agents - how many agents to create
        """
        if num_agents is None or num_agents <= 0:
            raise ValueError('num_agents must be positive')
        if model is None:
            raise ValueError('invalid model given')

        if name == None:
            name = 'Agent' + str(GameState.ID_counter)
            GameState.ID_counter += 1

        for i in range(num_agents):
            radius = utils.mass_to_radius(conf.AGENT_STARTING_MASS)
            pos = utils.gen_random_position(radius)
            ai_agent = Agent(
                self,
                model,
                pos[0],
                pos[1],
                radius,
                mass=conf.AGENT_STARTING_MASS,
                color=conf.BLUE_COLOR,
                name=name,
                manual_control=False,
            )
            print(model.id)
            self.agents[model.id] = ai_agent

    def is_exit_command(self, event):
        """Check if the user is pressing an exit key"""
        return event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)

    def draw_circle(self, board, obj, color=None, stroke=None):
        """Draw a circle on the pygame GUI"""
        if color is None:
            raise Exception('Color cannot be none')

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
        if conf.FULL_SCREEN:
            self.window = pygame.display.set_mode(
                (conf.SCREEN_WIDTH, conf.SCREEN_HEIGHT), pygame.FULLSCREEN)
        else:
            self.window = pygame.display.set_mode(
                (conf.SCREEN_WIDTH, conf.SCREEN_HEIGHT))

        pygame.display.set_caption('CIS 522: Final Project')
        self.board = pygame.Surface((conf.BOARD_WIDTH, conf.BOARD_HEIGHT))
        clock = pygame.time.Clock()
        running = True
        while running:
            clock.tick(conf.CLOCK_TICK)

            for agent in self.agents.values():
                self.update_interactive_state(agent)

            self.tick_game_state(None)

            # take in user input and draw/update the game board
            for event in pygame.event.get():
                # stop the game if user exits
                if self.is_exit_command(event):
                    running = False

            # redraw window then update the frame
            self.draw_window(self.board, self.window)
            pygame.display.update()

        pygame.quit()
        quit()
