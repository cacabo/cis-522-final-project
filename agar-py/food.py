class Food():
    def __init__(self, x, y, r, color):
        self.x_pos = x
        self.y_pos = y
        self.radius = r
        self.color = color

    def get_pos(self):
        return (self.x_pos, self.y_pos)