class Agent():
    def __init__(self, x, y, r, mass, color, name, manual_control):
        self.x_pos = x
        self.y_pos = y
        self.radius = r
        self.mass = mass
        self.color = color
        self.name = name
        self.velocity = (0, 0)
        self.manual_control = manual_control