import math
from cdg import cdg1

import numpy as np


def velocity_eqxyz(
    v, wv, t, t_gap, mass, diameter, distance, bc=-1, rho=0.07967, g=32.17404855643
):
    v2 = np.square(v)
    scalar_v = math.sqrt(np.sum(v2))
    c = cdg1(scalar_v)

    # if no value for bc, bc will be predicted using mass and diameter of the bullet:
    if bc == -1:
        _bc = mass / (diameter ** 2)
    else:
        _bc = bc * 144  # change of units

    cd_h = (rho * math.pi * c) / (8 * _bc)

    dv = np.subtract(-cd_h * scalar_v * v, np.array([0, g, wv]))

    new_v = np.add(v, dv * t_gap)
    new_t = t + t_gap

    distance = np.add(distance, np.divide(t_gap * np.add(new_v, v), 2))

    return new_v, new_t, distance


def velocityatdistance(
    ini_v, distance, mass, diameter, bc, rho=0.07967, g=32.17404855643, t_gap=0.0001
):
    """
    Input: initial velocity and a distance from the starting point.
        ini_v(numpy array, 1*3): initial velocity on x, y, z direction respectively. Measure in feet/second.
        distance(float): the final distance that you will have the velocity at. Measure in feet.
        mass(float): Mass of the bullet. Measure in grain.
        t_gap(float): The time precision of the function. Measure in second.
    return: Scalar velocity at different distances
    """

    mass = mass / 7000
    v2 = np.square(ini_v)
    scalar_v = math.sqrt(np.sum(v2))

    v_list = [scalar_v]
    distance_now = 0
    d = [np.array([0, 0, 0])]
    v = ini_v
    t = 0

    while d[-1][0] <= distance:

        new_v, t, distance_now = velocity_eqxyz(
            v,
            wv=14.6666667,
            t=t,
            t_gap=t_gap,
            mass=mass,
            bc=bc,
            diameter=diameter,
            distance=distance_now,
            rho=rho,
            g=g,
        )
        v = new_v
        d.append(np.divide(distance_now, 3))
        v_list.append(math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
    return v_list, d
