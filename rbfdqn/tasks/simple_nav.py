"""
I want to make the simplest navigation domain possible -- but I also want it to be a hard exploration domain.

Maybe easiest is if I just do the pointmaize u-maze. I can do a distance-based reward, which is probably simpler.
The big plus is that the fastest path isn't the one that follows the gradient.

Well, in tabular domains with Q-learning it's pretty clear that this is similar to value-function initialization, right?

Downside is, it's a harder problem, scaling is more confusing, mujoco, etc.

It would really be better if I didn't wait until the end of the episode in order to do these updates.

So, I think the simplest POSSIBLE domain for this would be just walking along a line, and encouraging exploration
in one direction more than the other. That might be too easy. So, next would be exploring a determinstic
infinite grid-world, where you don't know what the actions do. If you initialize to the proper Q-function, what
happens? Interestingly, I don't think it's perfect. If there are no restarts, and you're on the left frontier,
and you want to be going right, then: you may be 100 steps from the good fronteir, and 1 step from the bad
frontier. If your Q-fucntion is tight, there will be no preference between them. Which is sort of expected,
but annoying.

With restarts, however, it will tell you that it's not worth going left any more. Sort of interesting.

Granted, I'm sort of assuming infinite state space in the first one, so it's not like anyone else does any better.
It's a pretty understandable mistake. And, I think it's sort of a corner-case of only having one dimension to move
available to you.

I'm sort of concerned that "knownness" isn't really rigorous. Because, it's based upon
stochastitiy, and we ain't got that. But even if we did, it's not perfectly linked to
the lipschitz thing.

Speaking of lipschitz, is it any different if you have a non-uniform qmax bound? Well, if you know that


"""

import gym
from gym import spaces
import numpy as np

class PlaneWorldEnv(gym.Env):
    """
    This guy has no walls, no goal, nothing. It's just to see how this guy explores.
    Then, we'll add in potential functions and see how he explores then. Should
    be pretty interesting! I also need to add in a "linear" knownness, to make
    this more exhaustive.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        super(PlaneWorldEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1., high=1., shape=(2, ))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, ))

    def step(self, action):
        assert self.action_space.contains(action), "can't take action outside of bounds!"
        new_state = self._current_state + action
        self._current_state = new_state
        return np.copy(self._current_state), 0., False, {}

    def reset(self):
        self._current_state = np.array([0., 0.])
        # Reset the state of the environment to an initial state
        return np.copy(self._current_state)
        pass

    def render(self, mode='human', close=False):
        print("Can't render, but current state is: {}")
        # Render the environment to the screen


class SimpleNavEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, goal_loc=(10., 10), success_radius=1.):
        super(SimpleNavEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1., high=1., shape=(2, ))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, ))

        self.goal_state = np.array([10., 10.])
        self.success_radius = success_radius

    def step(self, action):
        assert self.action_space.contains(action), "can't take action outside of bounds!"
        new_state = self._current_state + action
        made_it = self.is_success(new_state)
        if made_it:
            done, reward = True, 1.
        else:
            done, reward = False, 0.
        self._current_state = new_state
        return np.copy(self._current_state), reward, done, {}

    def is_success(self, state):
        assert state.shape == self.goal_state.shape
        distance = np.linalg.norm(self.goal_state - state)
        # distance = np.sqrt(((self.goal_state - state)**2).sum())
        return distance <= self.success_radius

    def reset(self):
        self._current_state = np.array([0., 0.])
        # Reset the state of the environment to an initial state
        return np.copy(self._current_state)
        pass

    def render(self, mode='human', close=False):
        print("Can't render, but current state is: {}")
        pass
        # Render the environment to the screen


if __name__ == "__main__":
    env = SimpleNavEnv()

    import ipdb
    ipdb.set_trace()
    print('bango')