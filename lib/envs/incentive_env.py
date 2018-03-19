import gym
from gym import spaces
from gym import utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)

DIM = 1


class IncentiveEnv(gym.Env, utils.EzPickle):
    """Inventory control with lost sales environment

    TO BE EDITED

    This environment corresponds to the version of the inventory control
    with lost sales problem described in Example 1.1 in Algorithms for
    Reinforcement Learning by Csaba Szepesvari (2010).
    https://sites.ualberta.ca/~szepesva/RLBook.html
    """

    def __init__(self, nState=100, nAction=10, k=5, c=2, h=2, p=3, lam=8):
        self.action_space = spaces.Discrete(nAction)
        
        _max = np.array([100])
        _min = np.array([-100])
        self.observation_space = spaces.Box(_min, _max)
        
        self.state = None
        self.k = k
        self.c = c
        self.h = h
        self.p = p
        self.lam = lam

        # Set seed
        self._seed()

        # Start the first round
        self._reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def user(self):
        q = np.random.normal(5, 10)
        c = np.random.normal(50, 100)
        return q, c

    def transition(self, x, q, a, c):
        if q >= a and x >= c:
            next = (x + 1)
        else:
            next = (x - 1)
        return np.array([next])

    def reward(self, x, q, a, c):
        if q >= a and x >= c:
            lam = self.lam
            r = self.lam * q - x
        else:
            r = 0
        return r

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        x, = self.state
        x = int(x)
        q, c = self.user()
        #print (q, c)
        obs2 = self.transition(x, q, action, c)
        self.state = obs2
        reward = self.reward(x, q, action, c)
        done = 0
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=0, high=100, size=(DIM,))
        return self.state