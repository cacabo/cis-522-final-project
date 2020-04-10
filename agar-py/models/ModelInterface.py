class ModelInterface:
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    def get_action(self, state):
        """Given the current game state, determine what action the model will output"""
        raise NotImplementedError('Model get_action() is not implemented')

    def optimize(self, reward):
        """Given reward received, optimize the model"""
        raise NotImplementedError('Model optimize() is not implemented')