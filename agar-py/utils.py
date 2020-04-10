import numpy as np
import math
import config as conf


def massToRadius(mass):
    """
    Determine radius from mass of blob

    Parameters:

        mass - number

    Returns number
    """
    return int(4 + np.sqrt(mass) * 4)


def massToVelocity(mass):
    """
    Determine velocity from mass of blob

    Parameters:

        mass - number

    Returns number
    """
    return int(2.2 * np.power(mass / 1000, -0.439))


def randomPosition(radius):
    """
    Generate a random position within the field of play. NOTE the `radius` is
    used to position within bounds of the board.

    Parameters

        radius : number

    Returns

        tuple (x y)
    """
    return (
        randomInRange(radius, conf.BOARD_WIDTH - radius),
        randomInRange(radius, conf.BOARD_HEIGHT - radius),
    )


def randomInRange(lo, hi):
    """
    Generate random int in the range lo to hi

    Parameters

        lo : number
        hi : number

    Returns

        number
    """
    return int(np.floor(np.random.random() * (hi - lo)) + lo)


def getEuclideanDistance(p1, p2):
    """
    Calculate euclidean distance between two points

    Parameters

        p1 : tuple (x, y)
        p2 : tuple (x, y)

    Returns

        number
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def isPointInCircle(point, circle, radius):
    """
    Return if the provided point is within the circle

    Parameters

        point  : tuple (x, y)
        circle : tuple (x, y)
        radius : number

    Returns

        boolean
    """
    return radius >= getEuclideanDistance(point, circle)


def getCircleOverlap(c1, r1, c2, r2):
    """
    Measure overlap between two circles

    NOTE this is more positive the more cirlces overlap, this is negative if the
    circles do not overlap

    Parameters

        c1 - tuple (x, y) position of circle
        r1 - radius
        c2 - tuple (x, y) position of circle
        r2 - radius

    Returns

        number
    """
    dist_btwn_centers = np.linalg.norm(np.array(c1) - np.array(c2))
    sum_radii = r1 + r2
    return sum_radii - dist_btwn_centers


def getObjectOverlap(a, b):
    return getCircleOverlap(a.get_pos(), a.radius, b.get_pos(), b.radius)


def areCirclesColliding(c1, r1, c2, r2):
    """
    Parameters

        c1 : tuple (x, y) position of circle
        r1 : radius
        c2 : tuple (x, y) position of circle
        r2 : radius

    Returns boolean if the circles overlap
    """
    overlap = getCircleOverlap(c1, r1, c2, r2)
    return overlap > 0


def getAngleBetweenPoints(p1, p2):
    """
    Parameters

        p1 : tuple (x, y)
        p2 : tuple (x, y)

    Returns angle in degrees of line drawn between points and positive x dir
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


def getAngleBetweenObjects(a, b):
    return getAngleBetweenPoints(a.get_pos(), b.get_pos())


def moveObjectLeft(obj, vel):
    obj.x_pos = max(obj.x_pos - vel, obj.radius)


def moveObjectRight(obj, vel):
    obj.x_pos = min(obj.x_pos + vel, conf.BOARD_WIDTH - obj.radius)


def moveObjectUp(obj, vel):
    obj.y_pos = max(obj.y_pos - vel, obj.radius)


def moveObjectDown(obj, vel):
    obj.y_pos = min(obj.y_pos + vel, conf.BOARD_HEIGHT - obj.radius)


def moveObject(obj, angle, vel):
    if angle is None:
        return

    radians = angle / 180 * math.pi
    dx = math.cos(radians) * vel
    dy = math.sin(radians) * vel
    if dx > 0:
        moveObjectRight(obj, dx)
    elif dx < 0:
        moveObjectLeft(obj, dx * -1)

    if dy > 0:
        moveObjectUp(obj, dy)
    elif dy < 0:
        moveObjectDown(obj, dy * -1)
