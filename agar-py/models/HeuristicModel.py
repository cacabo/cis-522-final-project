import numpy as np
from models.ModelInterface import ModelInterface
import utils


class HeuristicModel(ModelInterface):
    def __init__(self):
        super().__init__()

    def get_action(self, state):
        (agents, foods, viruses, masses, time) = state
        my_agent = agents[self.id]

        # TODO: technically only considers the agent as one single cell. Could feasibly reach "unreachable" foods if its multiple cells
        nearest_food_action = self.get_nearest_food_action(my_agent, foods)

        return nearest_food_action

    # no optimization occurs for HeuristicModel
    def optimize(self):
        return

    # no remembering occurs for HeuristicModel
    def remember(self, state, action, next_state, reward, done):
        return

    def get_nearest_food_action(self, my_agent, foods):
        my_pos = my_agent.get_avg_pos()
        my_rad = my_agent.get_avg_radius()

        # find the nearest food object reachable by the shortest path direction
        nearest_food = None
        nearest_food_dist = np.inf
        for food in foods:
            food_pos = food.get_pos()
            if self.is_food_reachable(my_pos, my_rad, food_pos):
                curr_dist = utils.get_euclidean_dist(my_pos, food_pos)
                if curr_dist < nearest_food_dist:
                    nearest_food = food
                    nearest_food_dist = curr_dist

        # if there is no nearest food, choose a random action
        if nearest_food == None:
            return utils.get_random_action()
        # otherwise, get the direction that goes most directly to the nearest food object
        else:
            angle_to_food = utils.get_angle_between_points(
                my_pos, nearest_food.get_pos())
            return utils.get_action_closest_to_angle(angle_to_food)

    # helper function to determine if an agent can physically reach the given food
    def is_food_reachable(self, my_pos, my_rad, food_pos):
        angle_to_food = utils.get_angle_between_points(my_pos, food_pos)
        action_to_angle = utils.get_action_closest_to_angle(angle_to_food)
        return utils.is_action_feasible(action_to_angle, my_pos, my_rad)
