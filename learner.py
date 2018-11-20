# Author: Aaron Parisi
# 11/20/18
from agent import Agent
import abc


class Learner:
    __metaclass__ = abc.ABCMeta
    '''Baseclass for RL learners; these assume the agents are well-suited
    for their respective tasks and handle the implementation of the 
    partiuclar learning algorithm.
    
    Args: *replay: true if random trajectory sampling is enabled
    (to reduce variance / biases from starting conditions)'''
    def __init__(self, agent : Agent, env, optimizer, 
            replay = True, max_traj_len = 30, 
            gamma = 0.98, *args, **kwargs):
        self.agent = agent
        self.env = env
        self.opt = optimizer
        self.replay = True

        self.max_trajectory_len = max_traj_len
        self.gamma = gamma

    @abc.abstractmethod 
    def step(self):
        pass

    @abc.abstractmethod
    def get_discounted_reward(self, start, end = None):
        pass

    @abc.abstractmethod
    def get_policy_entropy(self, start, end = None):
        pass

    def report(self):
        pass 

#TODO TODO: create Agent + Learner for state-dynamic NN coupling


    
class TFLearner:
    pass


class PyTorchLearner:
    pass










