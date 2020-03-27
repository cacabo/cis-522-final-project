import pygame
import config as conf
import utils
from food import Food
from virus import Virus
from agent import Agent
from camera import Camera

pygame.init()

camera = None
agents = {}
foods = []
viruses = []

text_font = pygame.font.SysFont(
    conf.AGENT_NAME_FONT, conf.AGENT_NAME_FONT_SIZE)


def add_food(n):
    """
    insert food at random places on the board

    @param n - number of food to spawn
    """
    global foods
    radius = utils.massToRadius(conf.FOOD_MASS)
    for _ in range(n):
        # TODO: could include uniform distribution here
        pos = utils.randomPosition(radius)
        foods.append(Food(pos[0], pos[1], radius, (255, 0, 0)))


def add_virus(n):
    """insert viruses at random places on the board"""
    global viruses
    radius = utils.massToRadius(conf.VIRUS_MASS)
    for _ in range(n):
        # TODO: could include uniform distribution here
        pos = utils.randomPosition(radius)
        viruses.append(Virus(pos[0], pos[1], radius, conf.VIRUS_COLOR))


def balance_mass():
    """ensure that the total mass of the game is balanced between food and players"""
    global foods
    total_mass = len(foods) * conf.FOOD_MASS + \
        sum([agent.mass for agent in agents.values()])
    mass_diff = conf.GAME_MASS - total_mass
    max_num_food_to_add = conf.MAX_FOOD - len(foods)
    max_mass_food_to_add = mass_diff / conf.FOOD_MASS

    num_food_to_add = min(max_num_food_to_add, max_mass_food_to_add)
    if num_food_to_add > 0:
        add_food(num_food_to_add)
    # TODO: removing food if necessary

    num_virus_to_add = conf.MAX_VIRUSES - len(viruses)
    if num_virus_to_add > 0:
        add_virus(num_virus_to_add)


def check_food_collision(agent, food):
    """if the center of a food is inside the agent, the agent eats it"""
    return utils.isPointInCircle((food.x_pos, food.y_pos), (agent.x_pos, agent.y_pos), agent.radius)


def check_agent_collision(agent, other):
    """
    agent eats other if it (1) has mass greater by at least CONSUME_MASS_FACTOR
    and (2) agent's circle overlaps with the center of other
    """
    if agent.name != other.name:
        return agent.mass >= other.mass * conf.CONSUME_MASS_FACTOR and utils.isPointInCircle((other.x_pos, other.y_pos), (agent.x_pos, agent.y_pos), agent.radius)


def init_manual_agent(name):
    global agents, camera
    radius = utils.massToRadius(conf.AGENT_STARTING_MASS)
    pos = utils.randomPosition(radius)
    player = Agent(pos[0], pos[1], radius,
                   conf.AGENT_STARTING_MASS, (0, 255, 0), name, True)
    agents[player.name] = player
    camera = Camera((conf.SCREEN_WIDTH / 2 - player.x_pos),
                    (conf.SCREEN_HEIGHT / 2 - player.y_pos), player.radius)


def init_ai_agents(num_agents):
    global agents
    for i in range(num_agents):
        radius = utils.massToRadius(conf.AGENT_STARTING_MASS)
        pos = utils.randomPosition(radius)
        ai_agent = Agent(pos[0], pos[1], radius, conf.AGENT_STARTING_MASS,
                         (0, 0, 255), 'Agent' + str(i), False)
        agents[ai_agent.name] = ai_agent


def move_agent(agent):
    if agent.manual_control:
        # get key presses
        keys = pygame.key.get_pressed()
        agent.manual_move(keys, camera)
    else:
        agent.ai_move()


def tick_agent(agent):
    global foods, agents

    move_agent(agent)

    # find all food items which are not currently being eaten by this agent, and update global foods list
    foods_remaining = [
        food for food in foods if not check_food_collision(agent, food)]
    num_food_eaten = len(foods) - len(foods_remaining)
    foods = foods_remaining
    food_mass_gained = num_food_eaten * conf.FOOD_MASS

    # get a list of all agents which have collided with the current one, and see if it eats any of them
    eaten_agents = [other for other in agents.values(
    ) if check_agent_collision(agent, other)]
    eaten_agent_mass_gained = sum(
        [eaten_agent.mass for eaten_agent in eaten_agents])
    # remove each eaten agent from the game
    for eaten_agent in eaten_agents:
        eaten_agent.is_alive = False
        print('[GAME] ' + str(eaten_agent.name) +
              ' died! Was eaten by ' + str(agent.name))

    # update the agent's mass and radius
    total_mass_gained = food_mass_gained + eaten_agent_mass_gained
    agent.mass += total_mass_gained
    agent.radius = utils.massToRadius(agent.mass)


def draw_window(agents, foods, board):
    # fill screen white, to clear old frames
    WIN.fill((255, 255, 255))
    board.fill((255, 255, 255))

    for food in foods:
        pygame.draw.circle(board, food.color,
                           (food.x_pos, food.y_pos), food.radius)

    for agent in sorted(agents.values(), key=lambda x: x.mass):
        pygame.draw.circle(board, agent.color,
                           (agent.x_pos, agent.y_pos), agent.radius)
        agent_name_text = text_font.render(agent.name, 1, (0, 0, 0))
        board.blit(agent_name_text, (agent.x_pos - (agent_name_text.get_width() / 2),
                                     agent.y_pos - (agent_name_text.get_height() / 2)))

    for virus in viruses:
        pygame.draw.circle(board, virus.color,
                           (virus.x_pos, virus.y_pos), virus.radius)
        pygame.draw.circle(board, conf.VIRUS_OUTLINE_COLOR,
                           (virus.x_pos, virus.y_pos), virus.radius, 4)

    # draw leaderboard
    sorted_agents = list(
        reversed(sorted(agents.values(), key=lambda x: x.mass)))
    leaderboard_title = text_font.render("Leaderboard", 1, (0, 0, 0))
    start_y = 25
    x = conf.SCREEN_WIDTH - leaderboard_title.get_width() - 20
    WIN.blit(leaderboard_title, (x, 5))
    top_n = min(len(agents), conf.NUM_DISPLAYED_ON_LEADERBOARD)
    for idx, agent in enumerate(sorted_agents[:top_n]):
        text = text_font.render(str(
            idx + 1) + ". " + str(agent.name) + ' (' + str(agent.mass) + ')', 1, (0, 0, 0))
        WIN.blit(text, (x, start_y + idx * 20))

    WIN.blit(board, (camera.x_pos, camera.y_pos))


def is_exit_command(event):
    return event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)


def main_loop():
    global agents, foods
    board = pygame.Surface((conf.BOARD_WIDTH, conf.BOARD_HEIGHT))

    running = True
    while running:
        clock.tick(960)  # 30 fps max

        # make sure food/virus/player mass is balanced on the board
        balance_mass()

        # perform updates for all agents
        for agent in agents.values():
            tick_agent(agent)

        # after ticking all the agents, remove the dead ones
        dead_agents = [agent for agent in agents.values()
                       if not agent.is_alive]
        for dead_agent in dead_agents:
            del agents[dead_agent.name]

        for event in pygame.event.get():
            # stop the game if user exits
            if is_exit_command(event):
                running = False

        # redraw window then update the frame
        draw_window(agents, foods, board)
        pygame.display.update()
    pygame.quit()
    quit()


# setup pygame window
WIN = pygame.display.set_mode((conf.SCREEN_WIDTH, conf.SCREEN_HEIGHT))
pygame.display.set_caption('CIS 522: Final Project')
clock = pygame.time.Clock()

# main game loop
init_manual_agent('AgarAI')
init_ai_agents(conf.NUM_AI)
main_loop()
