import numpy as np
import math
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


def getAngleBetweenPoints(p1, p2):
    """
    @param p1 - tuple (x, y)
    @param p2 - tuple (x, y)
    @returns angle in degrees of line drawn between points and positive x dir
    """
    (x1, y1) = p1
    (x2, y2) = p2
    dx = x2 - x1
    dy = y1 - y2  # Since 0 is in the top left corner

    if dx == 0 and dy != 0:
        angle = 90 if dy > 0 else 270
        return angle
    elif dy == 0 and dx != 0:
        angle = 0 if dx > 0 else 180
        return angle
    elif dx != 0:
        radians = math.atan(abs(dy / dx))
        if dx < 0 and dy > 0:
            radians += math.pi / 2
        elif dy < 0 and dx < 0:
            radians += math.pi
        elif dy < 0:
            radians += 3 * math.pi / 2

        # radians = radians if dx > 0 else radians + math.pi / 2
        angle = radians / math.pi * 180
        return angle
    else:
        return None
