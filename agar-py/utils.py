import numpy as np
import config as conf


def massToRadius(mass):
    """determine radius from mass of blob"""
    return int(4 + np.sqrt(mass) * 4)


def massToVelocity(mass):
    """determine velocity from mass of blob"""
    return int(2.2 * np.power(mass / 1000, -0.439))


def randomPosition(radius):
    """generate a random position within the field of play"""
    return randomInRange(radius, conf.BOARD_WIDTH - radius), randomInRange(radius, conf.BOARD_HEIGHT - radius)


def randomInRange(lo, hi):
    """generate random int in the range lo to hi"""
    return int(np.floor(np.random.random() * (hi - lo)) + lo)


def getEuclideanDistance(p1, p2):
    """calculate euclidean distance"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def isPointInCircle(point, circle_center, circle_radius):
    """TODO: find better than brute force way to solve"""
    return circle_radius >= getEuclideanDistance(point, circle_center)


def areCirclesColliding(center1, radius1, center2, radius2):
    """TODO: find better way to solve"""
    return np.linalg.norm(np.array(center1) - np.array(center2)) < radius1 + radius2
