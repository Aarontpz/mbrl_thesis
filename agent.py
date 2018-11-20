# Author: Aaron Parisi
# 11/20/18

import abc
import numpy as np

import tensorflow as tf
import torch

class Agent:
    __metaclass__ = abc.ABCMeta
    '''Convention: Call step, store_reward, and if applicable, terminate_episode'''
    def __init__(self, input_dimensions : [], action_space : int, 
            discrete_actions = True, action_constraints : [] = None, 
            terminal_penalty = 0.0, 
            *args, **kwargs):
        '''Assumes flat action-space vector'''
        self.input_dimensions = input_dimensions
        self.action_space = action_space
        self.discrete = discrete_actions
        if discrete_actions is False:
            assert(action_constraints is not None)
            assert(len(action_constraints) == 2) #low/high for values
            self.action_constraints = action_constraints
        self.terminal_penalty = terminal_penalty
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.terminal_history = []

    @abc.abstractmethod
    def step(self, obs, *args, **kwargs) -> np.array:
        self.state_history.append(obs)
        self.terminal_history.append(0)

    def store_reward(self, r):
        self.reward_history.append(r)

    def terminate_episode(self):
        self.terminal_history.append(1)
        self.reward_history.append(self.terminal_penalty)

    #def construct_experience(self, s_k, a_k, r_k, s_K, terminal, **kwargs):
    #    return [s_k, a_k, r_k, s_K, terminal, kwargs]
    #    #TODO: Wrap these in tf / pytorch tensors?

    #def save_experience(self, s_k, a_k, r_k, s_K, terminal, **kwargs): #TODO: Memory constraints for long-episode task??
    #    e = self.assemble_experience(s_k, a_k, r_k, s_K, terminal, kwargs) 
    #    self.exp_buffer.append(e) 

    def get_experience(self, k) -> []: 
        s = None
        a = None
        r = None
        s_K = None
        if k < len(self.state_history):
            s = self.state_history[k]
            terminal = self.terminal_history[k]
            if terminal:
                return [s, None, None, None, terminal]
            a = self.action_history[k]
            r = self.reward_history[k] #NOTE: we assume 1st action gets 0 reward
            s_K = self.state_history[k+1]
            return [s, a, r, s_K, terminal]
        else:
            raise KeyError
    
    def get_experience_structure(self):
        return ['s_k', 'a_k', 'r_k', 's_K', 'terminal', 'kwargs']
        
    @abc.abstractmethod
    def transform_observation(self, obs, target_size = None, normalize_pixels = True, **kwargs) -> np.ndarray:
        if type(obs) is list:
            obs = np.array(obs)
        if target_size is not None:
            obs.resize(target_size)
        if normalize_pizels == True:
            obs /= 256



class RandomAgent(Agent):
    def step(self, obs, *args, **kwargs) -> np.ndarray:
        if self.discrete_actions:
            one_hots = np.eye(self.action_space)
            return one_hots[np.random.choice(self.action_space, 1)]
        else:
            mins = self.action_constraints[0]
            maxs = self.action_constraints[1]
            return np.random.uniform(mins, maxs, (1, self.action_space))


class TFAgent(Agent):
    def __init__(self):
    ##NOTE:We simply feed the numpy transform_observation into feed_dict
    ##for TFAgent
        pass


class PyTorchAgent(Agent):
    def __init__(self):
        pass
    def transform_observation(self, obs, ) -> torch.tensor:
        '''For PyTorch, we can transfer the observation directly into
        a tensor/variable with this helper function. '''
        pass

##  Tensorflow Helper functions/classes

#NOTE using https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py as reference

def lrelu(inp, name=None): 
    #return tf.nn.relu(inp, name=name)  
    return tf.maximum(inp, 0.2 * inp) #LEAKY RELU for encoder portion 

def relu(inp, name=None):
    return tf.nn.relu(inp, name=name)

def sigmoid(inp, name=None):
    return tf.nn.sigmoid(inp, name=name)

def droupout(inp, keep_prob, name=None):
    return tf.nn.dropout(inp, keep_prob, name=name)

def batch_norm(batch_input, is_training):
    return tf.layers.batch_normalization(batch_input, axis=3, epsilon=1e-5, 
            momentum=0.1, training=is_training, 
            gamma_initializer=tf.random_normal_initializer(1.0, 0.02), name=name) #DIRECTLY from git repo

def generate_weights_glorot(shape, name=None):
    if name is None:
        name = 'W'
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer()) 

def generate_bias_constant(shape, c:float, name=None):
    if name is None:
        name = 'B'
    return tf.get_variable(name, l, initializer=tf.constant_initializer(c)) 

def linear(batch_input, W, B):
    return tf.add(tf.matmul(batch_input, W), B)



def create_mlp(self, batch_inp, indim, outdim, hdim : [] = [], activations : [] = [], initializer = None, batchnorm = False, bias_const = 0.1):
    assert(len(activations) == len(hdim) + 1)
    layers = []
    prev_size = indim
    inp = batch_inp
    for i in range(len(hdim)):
        with tf.variable_scope("linear_%s"%(i)):    
            w_shape = [None, inp.get_shape()[1], hdim[i]]
            b_shape = [None, hdim[i]]
            W = generate_weights_glorot(w_shape, "W%s"%(i))
            B = generate_bias_constant(b_shape, bias_const, "B%s"%(i))
            l = linear(inp, W, B)
            layers.append(l)
            if activations[i] is not None:
                if activations[i] == 'relu':
                    layers.append(relu(layers[-1], "relu_%s"%(i)))
                elif activations[i] == 'sig':
                    layers.append(sigmoid(layers[-1], "sig_%s"%(i)))
            if batchnorm:
                layers.append(batch_norm(layers[-1], "norm_%s"%(i)))
            inp = layers[-1]
    with tf.variable_scope("linear_%s"%(i+1)):
        i = i + 1
        w_shape = [None, inp.get_shape()[1], outdim]
        b_shape = [None, outdim]
        W = generate_weights_glorot(w_shape, "W%s"%(i))
        B = generate_bias_constant(b_shape, bias_const, "B%s"%(i))
        l = linear(inp, W, B)
        layers.append(l)
        if activations[i] is not None:
            if activations[i] == 'relu':
                layers.append(relu(layers[-1], "relu_%s"%(i)))
            elif activations[i] == 'sig':
                layers.append(sigmoid(layers[-1], "sig_%s"%(i)))
    return layers[-1]




##

##  Pytorch helper functions /classes

class PyTorchMLP(torch.nn.Module):
    def __init__(self, indim, outdim, hdim : [] = [], 
            activations : [] = [], initializer = None, batchnorm = False):
        self.indim = indim
        self.outdim = outdim
        self.hdim = hdim
        self.activations = activations
        assert(len(activations) == len(hdim) + 1)
        
        layers = []
        prev_size = indim
        for i in range(len(hdim)):
            linear = torch.nn.Linear(prev_size, hdim[i], bias = True)
            layers.append(linear)
            if activations[i] is not None:
                if activations[i] == 'relu':
                    layers.append(torch.nn.functional.relu)
                elif activations[i] == 'sig':
                    layers.append(torch.nn.functional.sigmoid)
            if batchnorm:
                layers.append(tf.nn.BatchNorm1d(hdim[i]))
        linear = torch.nn.Linear(prev_size, outdim, bias = True)
        if activations[i+1] is not None:
            if activations[i+1] == 'relu':
                layers.append(torch.nn.functional.relu)
            elif activations[i+1] == 'sig':
                layers.append(torch.nn.functional.sigmoid)
        if batchnorm:
            layers.append(tf.nn.BatchNorm1d(outdim))

        self.layers = tf.nn.ModuleList(layers)
        raise Exception("BITCH USE TENSORFLOW FIRST!")

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

##



