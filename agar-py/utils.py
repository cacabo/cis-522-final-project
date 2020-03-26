import numpy as np
import config as conf

# determine mass from radius of circle
def massToRadius(mass):
    return int(4 + np.sqrt(mass) * 4)

# generate a random position within the field of play
def randomPosition(radius):
    return randomInRange(radius, conf.BOARD_WIDTH - radius), randomInRange(radius, conf.BOARD_HEIGHT - radius)

# generate random int in the range lo to hi
def randomInRange(lo, hi):
    return int(np.floor(np.random.random() * (hi - lo)) + lo)

# calculate euclidean distance
def getEuclideanDistance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# TODO: find better than brute force way to solve
def isPointInCircle(point, circle_center, circle_radius):
    return circle_radius >= getEuclideanDistance(point, circle_center)