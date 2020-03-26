import pygame
import config as conf
import utils
from food import Food
from agent import Agent

pygame.init()

agents = []
foods = []

text_font = pygame.font.SysFont(conf.AGENT_NAME_FONT, conf.AGENT_NAME_FONT_SIZE)

# insert food at random places on the board
def add_food(n):
    global foods
    radius = utils.massToRadius(conf.FOOD_MASS)
    for _ in range(n):
        # TODO: could include uniform distribution here
        pos = utils.randomPosition(radius)
        foods.append(Food(pos[0], pos[1], radius, (255, 0, 0)))

# ensure that the total mass of the game is balanced between food and players
def balance_mass():
    global foods
    total_mass = len(foods) * conf.FOOD_MASS + sum([agent.mass for agent in agents])
    mass_diff = conf.GAME_MASS - total_mass
    max_num_food_to_add = conf.MAX_FOOD - len(foods)
    max_mass_food_to_add = mass_diff / conf.FOOD_MASS

    num_food_to_add = min(max_num_food_to_add, max_mass_food_to_add)
    if num_food_to_add > 0:
        add_food(num_food_to_add)
    # TODO: removing food if necessary

# if the center of a food is inside the agent, the agent eats it
def check_food_collision(agent, food):
    return utils.isPointInCircle((food.x_pos, food.y_pos), (agent.x_pos, agent.y_pos), agent.radius)

def init_manual_agent(name):
    radius = utils.massToRadius(conf.AGENT_STARTING_MASS)
    pos = utils.randomPosition(radius)
    player = Agent(pos[0], pos[1], radius, conf.AGENT_STARTING_MASS, (0, 255, 0), name, True)
    agents.append(player)

def init_ai_agents(num_agents):
    for i in range(num_agents):
        radius = utils.massToRadius(conf.AGENT_STARTING_MASS)
        pos = utils.randomPosition(radius)
        ai_agent = Agent(pos[0], pos[1], radius, conf.AGENT_STARTING_MASS, (0, 0, 255), 'Agent' + str(i), False)
        agents.append(ai_agent)

def move_agent(agent):
    if agent.manual_control:
        # get key presses
        keys = pygame.key.get_pressed()
        agent.manual_move(keys)
    else:
        agent.ai_move()

def tick_agent(agent):
    global foods

    move_agent(agent)

    # find all food items which are not currently being eaten by this agent, and update global foods list
    foods_remaining = [food for food in foods if not check_food_collision(agent, food)]
    num_food_eaten = len(foods) - len(foods_remaining)
    foods = foods_remaining

    mass_gained = num_food_eaten * conf.FOOD_MASS

    # update the agent's mass and radius
    agent.mass += mass_gained
    agent.radius = utils.massToRadius(agent.mass)


def draw_window(agents, foods):
    # fill screen white, to clear old frames
    WIN.fill((255, 255, 255))

    for food in foods:
        pygame.draw.circle(WIN, food.color, (food.x_pos, food.y_pos), food.radius)
    
    for agent in agents:
        pygame.draw.circle(WIN, agent.color, (agent.x_pos, agent.y_pos), agent.radius)
        agent_name_text = text_font.render(agent.name, 1, (0,0,0))
        WIN.blit(agent_name_text, (agent.x_pos - (agent_name_text.get_width() / 2), agent.y_pos - (agent_name_text.get_height() / 2)))

def is_exit_command(event):
    return event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)

def main_loop():
    global agents, foods

    running = True
    while running:
        clock.tick(240) # 30 fps max

        # make sure food/virus/player mass is balanced on the board
        balance_mass()

        for agent in agents:
            tick_agent(agent)

        for event in pygame.event.get():
            # stop the game if user exits
            if is_exit_command(event):
                running = False

        # redraw window then update the frame
        draw_window(agents, foods)
        pygame.display.update()
    pygame.quit()
    quit()

# setup pygame window
WIN = pygame.display.set_mode((conf.SCREEN_WIDTH, conf.SCREEN_HEIGHT))
pygame.display.set_caption("AgarAI")
clock = pygame.time.Clock()

# main game loop
init_manual_agent('CIS 522 UNIT')
init_ai_agents(conf.NUM_AI)
main_loop()