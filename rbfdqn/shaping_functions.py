"""
These will be functions maybe for each class that specify a tighter upper bound for q_max,
as opposed to q_max everywhere. These will just be state-based... Easy example would using
distance-from-goal if you know the maximum step size. The benefit over a dense reward is
that this will be "provably" convergent even if you'd think there were local optima.

Note that these should probably depend on gamma for the most part. Don't know how to
add that easily.
"""

import numpy as np
import torch

def maze_shaping(states, gamma=0.99):
    """
    Distance-based shaping for the PointMaze domain
    """
    assert len(states.shape) == 2, states.shape
    assert states.shape[1] == 2, states.shape


    distance_from_goal = torch.absolute(-states + 9.0)
    # distance_from_goal = np.maximum(distance_from_goal, 0)
    num_steps_from_goal = distance_from_goal.sum(dim=1) / 0.95
    num_discounts = torch.pow(gamma, num_steps_from_goal)

    assert len(num_discounts) == len(states)

    return num_discounts

def mcar_shaping_1(states, gamma=0.99):
    """
    Shaping towards the goal
    """
    x_coord_states = states[:,0]

    min_steps_away = (0.45 - x_coord_states) / 0.07
    min_steps_away = torch.clamp(min_steps_away, min=0.)

    Q_max = 100. * (gamma ** min_steps_away)
    return Q_max

def mcar_shaping_bad(states, gamma=0.99):
    """
    The opposite of our good shaping function. Shapes towards the right side,
    while maintaining the Qmax > Q* property
    """
    x_coord_states = states[:,0]

    min_steps_away = (0.45 - x_coord_states) / 0.07
    min_steps_away = torch.clamp(min_steps_away, min=0.)
    reversed_min_steps_away = -1 * min_steps_away

    Q_max = 100. * (gamma ** reversed_min_steps_away)
    return Q_max

def plane_shaping(states, gamma=0.99):
    assert len(states.shape) == 2, states.shape
    assert states.shape[1] == 2, states.shape
    x_coords = states[:,0]
    shifted_x_coords = x_coords + 50

    # print('plane')

    assert len(shifted_x_coords) == len(states)
    return shifted_x_coords

def plane_quadrants_shaping(states, gamma=0.99):
    values = [(s[0] >= 0) == (s[1] >= 0) for s in states]
    values = np.array([1. if v else 0. for v in values])
    assert len(values) == len(states)
    return values

def plane_reverse_quadrants_shaping(states, gamma=0.99):
    orig_values = plane_quadrants_shaping(states, gamma=gamma)
    new_values = 1 - orig_values
    return new_values

class ShapingFunctions:

    """Okay, I want this to take in a tensor. Much faster that way. """

    shaping_functions = {
        'SibRivPointMaze-v1': {
            'default': maze_shaping
        },
        'PlaneWorld-v1': {
            'default': plane_shaping,
            'force_right': plane_shaping,
            'force_xor': plane_quadrants_shaping,
            'force_reverse_xor': plane_reverse_quadrants_shaping,
        },
        'MountainCarContinuous-v0': {
            'default': mcar_shaping_1,
            'max_steps_away': mcar_shaping_1,
            'reversed_max_steps_away': mcar_shaping_bad,
        }
    }

    def __init__(self, env_name, gamma, func_name=None):

        assert env_name in self.shaping_functions.keys(), env_name

        self.env_name = env_name
        self.gamma = gamma
        self.func_name = func_name

        self._assign_shaping_function()
    
    def _assign_shaping_function(self):
        sfd = self.shaping_functions[self.env_name]

        func_name = 'default' if self.func_name is None else self.func_name
        sf = sfd[func_name]
        self._shaping_function = sf

    def get_values(self, states):
        reshaped = False
        if len(states.shape) == 1:
            print('reshping!')
            reshaped = True
            states = states.view(1,-1)

        shaping_values = self._shaping_function(states, self.gamma)

        if reshaped == True:
            shaping_values = shaping_values.view(-1)
        
        return shaping_values



def _test_shaping():
    state = np.array([[20.,20.],[0.,0.]])
    upper_bounds = maze_shaping(state)
    print(upper_bounds)

    states = np.array([[1.,1.],[-1.,-1.],[-1.,1.],[1.,-1.]])
    values = plane_quadrants_shaping(states)
    print(values)


    pass

if __name__ == "__main__":
    _test_shaping()