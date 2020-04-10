import numpy as np
from models.ModelInterface import ModelInterface
import utils

class HeuristicModel(ModelInterface):
    def __init__(self):
        super().__init__()
    def get_action(self, state):
        (agents, foods, viruses, time) = state
        my_pos = agents[self.id].get_avg_pos()

        nearest_food_action = self.get_nearest_food_action(my_pos, foods)

        return nearest_food_action

    # no optimization occurs for HeuristicModel
    def optimize(self, reward):
        return
    
    # no remembering occurs for HeuristicModel
    def remember(self, state, action, next_state, reward, done):
        return


    def get_nearest_food_action(self, my_pos, foods):
        # find the nearest food object
        nearest_food = None
        nearest_food_dist = np.inf
        for food in foods:
            food_pos = food.get_pos()
            curr_dist = utils.getEuclideanDistance(my_pos, food_pos)
            if curr_dist < nearest_food_dist:
                nearest_food = food
                nearest_food_dist = curr_dist

        # if there is no nearest food, choose a random action
        # TODO: do something better?
        if nearest_food == None:
            return utils.getRandomAction()
        # otherwise, get the direction that goes most directly to the nearest food object
        else:
            angle_to_food = utils.getAngleBetweenPoints(my_pos, nearest_food.get_pos())
            return utils.getActionClosestToAngle(angle_to_food)