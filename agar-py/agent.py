import pygame
import numpy as np
import config as conf

class Agent():
    def __init__(self, x, y, r, mass, color, name, manual_control):
        self.x_pos = x
        self.y_pos = y
        self.radius = r
        self.mass = mass
        self.color = color
        self.name = name
        
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

    def manual_move(self, keys):
        # TODO: better velocity control
        vel = int(max(conf.AGENT_STARTING_SPEED - (self.mass * 0.05), 1))

        # movement based on key presses
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.move_left(vel)

        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.move_right(vel)

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.move_up(vel)

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.move_down(vel)
    
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