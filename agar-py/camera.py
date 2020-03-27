import config as conf


class Camera():
    def __init__(self, x, y, player_radius):
        self.x_pos = x
        self.y_pos = y
        self.player_radius = player_radius

    def move_left(self, vel):
        self.x_pos = min(self.x_pos + vel, conf.SCREEN_WIDTH /
                         2 - self.player_radius)

    def move_right(self, vel):
        self.x_pos = max(self.x_pos - vel,
                         self.player_radius - conf.SCREEN_WIDTH / 2)

    def move_up(self, vel):
        self.y_pos = min(self.y_pos + vel,
                         conf.SCREEN_HEIGHT / 2 - self.player_radius)

    def move_down(self, vel):
        self.y_pos = max(self.y_pos - vel,
                         self.player_radius - conf.SCREEN_HEIGHT / 2)
