"""
It's annoying that we have to do it here but for something like Ant, we're not going to be able to
specify it easily inside of the rbf_hyper_parameters file. Because, for something like Ant, we have
2 COM dimensions, and Bipedal we have 1. 

So, we're going to do something similar to shaping_functions.

The way it'll work is, to make it modular, we'll take in a single string that we then separate out to
get the scaling. I like that. So, it'll be something like, for Ant:
    {
        uniform: func_get_ones()
        special_loc_scaling: func_special_loc(com, rest_state, actions)
    }

There's an argument for making these things know about the environment already. Only because we need
the state and action dimensions. So maybe you pass the environment into the constructor?

It's sort of annoying -- do we do the automatic scaling or not? I'd say leave the option, for something like Ant,
it's unavoidable to use it, even though it does make the problem non-stationary.

And it figures out the rest from there.

So, in the end this will just return an array. 
"""

import numpy as np


def uniform_scaling(*args, **kwargs):
    return 1.


def action_scaling(env, action_scaler):
    """
    This is actually going to just be "action scaling". Because,
    it's all about the ratio, and the ratio doesn't change!
    """
    try:
        state_dim = len(env.observation_space.low)
    except AttributeError:
        print("Using dm_control so need to get state_dim differently")
        state_dim = len(env.observation_space['observations'].low)

    action_dim = len(env.action_space.low)

    # state_scaling = float(state_scaling)
    action_scaler = float(action_scaler)

    state_scaler_array = np.ones((state_dim,), dtype=np.float32)
    action_scaler_array = np.ones((action_dim,), dtype=np.float32) * action_scaler

    return np.concatenate([state_scaler_array, action_scaler_array], axis=0)

def per_dim_scaling(env, *args):
    try:
        state_dim = len(env.observation_space.low)
    except AttributeError:
        print("Using dm_control so need to get state_dim differently")
        state_dim = len(env.observation_space['observations'].low)
    action_dim = len(env.action_space.low)
    assert len(args) == state_dim + action_dim
    return np.array(args, dtype=np.float32)

def ant_maze_scaling(env, com_scaling, other_feature_scaling, action_scaling):
    """
    Not sure how this is correct, but: I'm assuming that the COM is the first 2 states. Then,
    the rest of the state is the pos/vel of everything.
    """
    state_dim = len(env.observation_space.low)
    action_dim = len(env.action_space.low)

    num_com_features = 2
    num_other_features = state_dim - num_com_features

    com_scaler = np.ones((num_com_features,), dtype=np.float32) * float(com_scaling)
    other_feature_scaler = np.ones((num_other_features,), dtype=np.float32) * float(other_feature_scaling)
    action_scaler = np.ones((action_dim,), dtype=np.float32) * float(action_scaling)

    return np.concatenate([com_scaler, other_feature_scaler, action_scaler], axis=0)

    # assert 

    print("Just a note that you should PROBABLY be normalizing one way or another for this one.")



"""
This has an interesting interface -- scaling_string is a string where the arguments are double-underscore-separated.
That lets us pass stuff in through a CLI interface a bit easier.
"""
_SCALING_FUNCTIONS = {
    'action_scaling': action_scaling,
    'per_dim_scaling': per_dim_scaling,
    'ant_maze_scaling': ant_maze_scaling,
}

def get_scaling_array(env, scaling_function_string):
    scaling_string_parsed = scaling_function_string.split("__")
    scaling_method, scaling_args = scaling_string_parsed[0], scaling_string_parsed[1:]
    scaling_array = _SCALING_FUNCTIONS[scaling_method](env, *scaling_args)
    return scaling_array



# class ScalingFunctions:
#     """
#     This has an interesting interface -- scaling_string is a string where the arguments are double-underscore-separated.
#     That lets us pass stuff in through a CLI interface a bit easier.
#     """
#     SCALING_FUNCTIONS = {
#         'state_action_scaling': state_action_scaling,
#         'per_dim_scaling': per_dim_scaling 
#     }

#     def __init__(self, env, scaling_string):
#         scaling_string_parsed = scaling_string.split("__")
#         scaling_method, scaling_args = scaling_string_parsed[0], scaling_string_parsed[1:]
#         scaling_array = self.SCALING_FUNCTIONS[scaling_method](env, *scaling_args)
#         # return scaling_array
