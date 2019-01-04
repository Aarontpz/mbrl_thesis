# Author: Aaron Parisi
# 11/20/18
import random
import abc
import numpy as np

import tensorflow as tf
import torch

class Agent:
    __metaclass__ = abc.ABCMeta
    '''Convention: Call step, store_reward, and if applicable, terminate_episode'''
    def __init__(self, input_dimensions : [], action_space : int, 
            discrete_actions = True, action_constraints : [] = None, 
            has_value_function = True, terminal_penalty = 0.0, 
            *args, **kwargs):
        '''Assumes flat action-space vector'''
        self.obs_mean = None
        self.obs_variance = None
        self.steps = 0

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
        self.action_score_history = []
        self.reward_history = []
        self.net_reward_history = []
        self.net_loss_history = [] #wtf is this loss
        self.terminal_history = []

        self.has_value_function = has_value_function
        self.value_history = [] #NOTE: this could be None

    @property
    @abc.abstractmethod
    def module(self):
        pass
    
    @module.setter
    @abc.abstractmethod
    def module(self, m):
        pass

    @abc.abstractmethod
    def step(self, obs, *args, **kwargs) -> np.array:
        self.steps += 1
        self.state_history.append(obs)
        self.terminal_history.append(0)
        action, value, normal = self.evaluate(obs, *args, **kwargs) #action may be ONE SHOT or continuous, values may be None
        self.action_history.append(action)

        if self.has_value_function:
            self.value_history.append(value)
        if self.discrete: 
            self.action_score_history.append(action)
        if not self.discrete: #TODO: make this more efficient BY OUTPUTTING NORMAL DIST INSTEAD
            action_score = normal.log_prob(action) #use constructed normal distribution to generate log prob
            self.action_score_history.append(action_score) 
            #print("ACTION: ", action)
            return action 

    def evaluate(self, obs, *args, **kwargs) -> (np.array, None, None):
        action_scores = None
        value = None
        normal = None

        if self.has_value_function:
            if self.discrete: #TODO: there could be a more pythonic way of doing this
                action_scores, value = self.module(obs)
            else:
                action_mu, action_sigma, value = self.module(obs)
        else:
            if self.discrete: #TODO: there could be a more pythonic way of doing this
                action_scores = self.module(obs)
            else:
                action_mu, action_sigma = self.module(obs)
        if self.discrete: 
            return action_scores, value, normal
        if not self.discrete:
            #print("Action mu: %s Sigma^2: %s" % (action_mu, action_sigma))
            action_covariance = torch.eye(self.action_space, 
                    device = self.device) * action_sigma.to(self.device) #create SPHERICAL covariance
            action_mu = action_mu.to(self.device) #TODO: this should naturally happen via the MLP
            normal = torch.distributions.MultivariateNormal(action_mu, action_covariance)
            try:
                action = normal.sample()
            except RunTimeError: #resample
                action = normal.sample() 
            #action_score = normal.log_prob(action)
            #print("ACTION: ", action)
            return action, value, normal

    def store_reward(self, r):
        self.reward_history.append(r)

    def terminate_episode(self):
        self.steps = 0
        self.terminal_history[-1] = 1
        self.reward_history[-1] = self.terminal_penalty
        avg_reward = sum(self.reward_history) / len(self.reward_history)
        self.net_reward_history.append(avg_reward)

    def reset_histories(self):
        self.state_history = []
        self.action_history = []
        self.action_score_history = []
        self.reward_history = []
        self.terminal_history = []
        self.value_history = []


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
    @abc.abstractmethod
    def preprocess(self, obs) -> np.ndarray:
        #TODO: add different means of pre-processing
        #assert(self.obs_mean is not None)
        #assert(self.obs_var is not None)
        if self.obs_mean is not None and self.obs_var is not None:
            pass

    @abc.abstractmethod
    def __call__(self, timestep):
        '''NOTE: This is SPECIFICALLY used to interface with dm_control.viewer'''
        pass


class RandomAgent(Agent):
    def step(self, obs, *args, **kwargs) -> np.ndarray:
        if self.discrete_actions:
            one_hots = np.eye(self.action_space)
            return one_hots[np.random.choice(self.action_space, 1)]
        else:
            mins = self.action_constraints[0]
            maxs = self.action_constraints[1]
            return np.random.uniform(mins, maxs, (1, self.action_space))

#raise Exception("Implement this, recreate their MPC paper")
class MPCAgent(Agent): 
    pass

class TFAgent(Agent):
    def __init__(self):
    ##NOTE:We simply feed the numpy transform_observation into feed_dict
    ##for TFAgent
        pass


class PyTorchAgent(Agent):
    def __init__(self, device, *args, **kwargs):
        super(PyTorchAgent, self).__init__(*args, **kwargs)
        self.device = device
        self._module = None
        self.obs_mean = torch.zeros(self.input_dimensions).to(device)
        self.obs_var = torch.zeros(self.input_dimensions).to(device)

    def step(self, obs) -> np.ndarray:
        '''Transforms observation, stores relevant data in replay buffers, then
        outputs action. 
        
        Outputs as np.ndarray because that's what the environment
        needs'''
        obs = self.transform_observation(obs).to(self.device)
        return super(PyTorchAgent, self).step(obs)
   
    def evaluate(self, obs, *args, **kwargs) -> torch.tensor:
        #online param calculation discussion: https://discuss.pytorch.org/t/normalization-of-input-data-to-qnetwork/14800/2
        obs = self.preprocess(obs)
        return super(PyTorchAgent, self).evaluate(obs, *args, **kwargs)

    def terminate_episode(self):
        self.obs_mean = torch.zeros(self.input_dimensions).to(self.device)
        self.obs_var = torch.zeros(self.input_dimensions).to(self.device)
        super(PyTorchAgent, self).terminate_episode()

    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, m):
        self._module = m

    def compute_normalization(self, obs):
        last_mean = self.obs_mean.clone()
        #print("Obs Mean: %s Step: %s Obs: %s" % (self.obs_mean, self.steps, obs))
        self.obs_mean += (obs - self.obs_mean)/self.steps
        mean_diff = (obs - last_mean)*(obs - self.obs_mean) 
        self.obs_var = torch.clamp(mean_diff/self.steps, min=1e-2)    

    def normalize(self, obs) -> torch.tensor:
        return (obs - self.obs_mean) / self.obs_var

    def preprocess(self, obs) -> torch.tensor:
        #TODO: add different means of pre-processing
        self.compute_normalization(obs)
        if self.obs_mean is not None and self.obs_var is not None:
            return self.normalize(obs)
        return obs

    def __call__(self, timestep):
        obs = timestep.observation
        action = self.step(obs).cpu().detach()
        if self.discrete: #TODO: This is currently only compatible with action_space = 1
            action_ind = max(range(len(action)), key=action.__getitem__) 
            action = [-1, 1][action_ind] 
        else: #continuous action space, currently mu + sigma
            pass            
        return action

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
    def __init__(self, device, indim, outdim, hdims : [] = [], 
            activations : [] = [], initializer = None, batchnorm = False):
        super(PyTorchMLP, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hdims = hdims
        self.activations = activations
        self.batchnorm = batchnorm
        self.device = device
        assert(len(activations) == len(hdims) + 1)
        
        layers = []
        prev_size = indim
        for i in range(len(hdims)):
            linear = torch.nn.Linear(prev_size, hdims[i], bias = True)
            layers.append(linear)
            if activations[i] is not None:
                if activations[i] == 'relu':
                    layers.append(torch.nn.LeakyReLU())
                elif activations[i] == 'sig':
                    layers.append(torch.nn.Sigmoid())
            if batchnorm:
                layers.append(tf.nn.BatchNorm1d(hdims[i]))
            prev_size = hdims[i]
        linear = torch.nn.Linear(prev_size, outdim, bias = True)
        layers.append(linear)
        final_ind = len(hdims)
        if activations[final_ind] is not None:
            if activations[final_ind] == 'relu':
                layers.append(torch.nn.LeakyReLU())
            elif activations[final_ind] == 'sig':
                layers.append(torch.nn.Sigmoid())
        if batchnorm:
            layers.append(tf.nn.BatchNorm1d(outdim))

        self.layers = torch.nn.ModuleList(layers)
        #raise Exception("BITCH USE TENSORFLOW FIRST!")


    def forward(self, x):
        for l in range(len(self.layers)):
            x = self.layers[l](x)
        return x

class PyTorchDiscreteACMLP(PyTorchMLP):
    '''Adds action / value heads to the end of an MLP constructed
    via PyTorchMLP '''
    def __init__(self, action_space, seperate_value_network = False,
            action_bias = True, value_bias = True,
            *args, **kwargs):
        self.action_space = action_space
        self.seperate_value_net = seperate_value_network
        super(PyTorchDiscreteACMLP, self).__init__(*args, **kwargs)
        if seperate_value_network:
            self.value_mlp = PyTorchMLP(*args, **kwargs)
        self.action_module = torch.nn.Linear(self.outdim, 
                action_space, bias = action_bias)
        self.value_module = torch.nn.Linear(self.outdim, 1, 
                bias = value_bias)
    
    def forward(self, x):
        mlp_out = super(PyTorchDiscreteACMLP, self).forward(x)
        #print("MLP OUT: ", mlp_out)
        actions = self.action_module(mlp_out) 
        action_scores = torch.nn.functional.softmax(actions, dim=-1)
        if self.seperate_value_net:
            mlp_out = self.value_mlp.forward(x)
        value = self.value_module(mlp_out)
        #print("ACTION: %s VALUE: %s" % (actions, value))
        return action_scores, value

class PyTorchContinuousGaussACMLP(PyTorchMLP):
    '''Adds action (mean, variance) / value heads 
    to the end of an MLP constructed via PyTorchMLP '''
    def __init__(self, action_space, seperate_value_network = True,
            action_bias = True, value_bias = True, sigma_head = False,
            *args, **kwargs):
        self.action_space = action_space
        self.seperate_value_net = seperate_value_network
        self.sigma_head = sigma_head
        super(PyTorchContinuousGaussACMLP, self).__init__(*args, **kwargs)
        if seperate_value_network:
            self.value_mlp = PyTorchMLP(*args, **kwargs)
        self.action_mu_module = torch.nn.Linear(self.outdim, 
                action_space, bias = action_bias)
        self.action_sigma_module = torch.nn.Linear(self.outdim,
                action_space, bias = action_bias)
        self.action_sigma_softplus = torch.nn.Softplus()
        self.value_module = torch.nn.Linear(self.outdim, 1, 
                bias = value_bias)
    
    def forward(self, x):
        mlp_out = super(PyTorchContinuousGaussACMLP, self).forward(x)
        #print("MLP OUT: ", mlp_out)
        action_mu = self.action_mu_module(mlp_out) 
        if self.sigma_head:
            action_sigma = self.action_sigma_module(mlp_out)
            action_sigma = self.action_sigma_softplus(action_sigma) #from 1602.01783 appendix 9
        else: #assume sigma is currently zeros
            action_sigma = torch.ones([1, self.action_space], dtype=torch.float32) * 0.0001
        if self.seperate_value_net:
            mlp_out = self.value_mlp.forward(x)
        value = self.value_module(mlp_out)
        #print("ACTION: %s VALUE: %s" % (actions, value))
        return action_mu, action_sigma, value

#class PyTorchDynamicsMLP(PyTorchMLP):
#    def __init__(self, state_dim, *args, **kwargs):
#        super(PyTorchDynamicsMLP, self).__init__(*args, **kwargs)
#        self.state_dim = state_dim



def EpsGreedyMLP(mlp_base, eps, eps_decay = 1e-3, eps_min = 0.0, action_constraints = None, 
        *args, **kwargs):
    class PyTorchEpsGreedyModule(mlp_base):
        def __init__(self, mlp_base, eps, eps_decay, eps_min, action_constraints = None, 
                *args, **kwargs):
            self.eps = eps
            self.decay = eps_decay
            self.eps_min = eps_min
            self.base = mlp_base
            self.action_constraints = action_constraints
            super(PyTorchEpsGreedyModule, self).__init__(*args, **kwargs)
    
        def update_eps(self):
            self.eps = max(self.eps_min, self.eps - self.decay)
            #print("EPS: ", self.eps)

        def forward(self, x):
            if self.base == PyTorchDiscreteACMLP: #TODO: can't assume value function exists...??
                action_score, values = super(PyTorchEpsGreedyModule,
                        self).forward(x)
                if random.random() < self.eps:
                    self.update_eps()
                    with torch.no_grad(): #no gradient for this
                        action = random.choice([0, self.action_space - 1])
                        action_score = torch.tensor( \
                                np.eye(self.action_space)[action],
                                    device = self.device).float()
                        #print("A: %s Score: %s" % (action, action_score))
                return action_score, values
            elif self.base == PyTorchContinuousGaussACMLP:
                action_mu, action_sigma, values = super(PyTorchEpsGreedyModule,
                        self).forward(x)
                if random.random() < self.eps: #randomize sigma values
                    self.update_eps()
                    mins = self.action_constraints[0]
                    maxs = self.action_constraints[1]
                    action_mu =  np.random.uniform(mins, maxs, (1, self.action_space))
                    #action_sigma = np.random.random_integers(1, high = 2, size = (1, self.action_space))
                    action_sigma = np.random.uniform(low = 0.000001, high = 3.0, size = (1, self.action_space))
                    action_mu = torch.as_tensor(action_mu).float()
                    action_sigma = torch.tensor(action_sigma).float()
                    #print("RANDOMIZING! Eps: ", self.eps)
                return action_mu, action_sigma, values 
                raise Exception("Figure this out?")

    return PyTorchEpsGreedyModule(mlp_base, eps, eps_decay, eps_min, action_constraints, *args, **kwargs)
##


class MPCAgent(Agent):
    '''Implements MPC for the MuJoCo environment, as specified
    in https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
    
    This DOES NOT imply that this agent relies on a model, however. 
    The paper itself states that, considering contact friction on
    multiple joints, fully-modelled ("smoothly" modelled) systems 
    are rather untrue to the physical outcome.  
    ''' #TODO: Verify / question the above statement
    pass



