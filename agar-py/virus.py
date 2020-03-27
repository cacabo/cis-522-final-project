class Virus():
    def __init__(self, x, y, r, mass):
        self.x_pos = x
        self.y_pos = y
        self.radius = r
        self.mass = mass
    
    def get_pos(self):
        return (self.x_pos, self.y_pos)