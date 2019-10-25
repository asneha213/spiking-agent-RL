import numpy as np
import math
import argparse
import itertools
import random
from matplotlib import pyplot as plt
import pdb


class MountainCar:
    def __init__(self):
        current_state = None;

    def transition_function(self, state, action):
        x = state[0]
        v = state[1]
        v_n = v + 0.001*action - 0.0025*np.cos(3*x)
        x_n = x + v_n
        if x_n < -1.2:
            x_n = -1.2
            v_n = 0
        elif x_n > 0.5:
            x_n = 0.5
            v_n = 0

        if v_n > 0.07:
            v_n = 0.07
        elif v_n < -0.07:
            v_n = -0.07

        return [x_n, v_n]

    def param_state(self, state, weights):
        param_state = 0
        for i in range(len(state)):
            param_state += state[i]*weights[i]
        return param_state


