# Author: Aaron Parisi
# 11/20/18
import os
import random
import abc
import numpy as np

import tensorflow as tf
import torch
from torch.autograd import Variable

import torch.multiprocessing as mp

import math

class Agent:
    __metaclass__ = abc.ABCMeta
    '''Convention: Call step, store_reward, and if applicable, terminate_episode'''
    def __init__(self, input_dimensions : [], action_space : int, 
            discrete_actions = True, action_constraints : [] = None, 
            has_value_function = True, terminal_penalty = 0.0, 
            policy_entropy = True, energy_penalty = True, 
            encode_inputs = False, encoder = None, 
            *args, **kwargs):
        '''Assumes flat action-space vector'''
        self.obs_mean = None
        self.obs_variance = None
        self.steps = 0

        self.input_dimensions = input_dimensions
        self.action_space = action_space
        self.discrete = discrete_actions
        if discrete_actions is False:
            #assert(action_constraints is not None)
            #assert(len(action_constraints) == 2) #low/high for values
            self.action_constraints = action_constraints
        self.terminal_penalty = terminal_penalty
        self.state_history = []
        self.action_history = []
        self.action_score_history = []
        self.reward_history = []
        self.net_reward_history = []
        self.net_loss_history = [] #wtf is this loss
        self.action_loss_history = []
        self.value_loss_history = []
        self.terminal_history = []
        
        if policy_entropy:
            self.policy_entropy_history = []
        if energy_penalty:
            self.energy_history = []

        self.encode_inputs = encode_inputs
        self.encoder = encoder

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
        if not self.encode_inputs: #otherwise, obs is already encoded...
            self.state_history.append(obs)
        self.terminal_history.append(0)
        action, value, normal = self.evaluate(obs, *args, **kwargs) #action may be ONE SHOT or continuous, values may be None
        self.action_history.append(action)

        if self.has_value_function:
            self.value_history.append(value)
        if not (hasattr(self, 'mpc') or hasattr(self, 'random')):
            if self.discrete: 
                self.action_score_history.append(action)
            if not self.discrete: #TODO: make this more efficient BY OUTPUTTING NORMAL DIST INSTEAD
                action_score = normal.log_prob(action) #use constructed normal distribution to generate log prob
                self.action_score_history.append(action_score) 
                #print("ACTION SCORE: ", action_score)
        return action 

    def evaluate(self, obs, *args, **kwargs) -> (np.array, None, None):
        action_scores = None
        value = None
        normal = None
        
        value_input = None
        if 'value_module_input' in kwargs.keys():
            value_input = kwargs['value_module_input']

        if self.has_value_function:
            if self.discrete: #TODO: there could be a more pythonic way of doing this
                action_scores, value = self.module(obs, value_input = value_input)
            else:
                action_mu, action_sigma, value = self.module.forward(obs, value_input = value_input)
        else:
            if self.discrete: #TODO: there could be a more pythonic way of doing this
                action_scores = self.module(obs, value_input)
            else:
                action_mu, action_sigma = self.module(obs, value_input)
        if self.discrete: 
            self.policy_entropy_history.append(self.get_policy_entropy(action_scores))
            return action_scores, value, normal
        if not self.discrete:
            #print("Action mu: %s \nSigma^2: %s" % (action_mu, action_sigma))
            action_covariance = torch.eye(self.action_space, 
                    device = self.device) * action_sigma.to(self.device) #create SPHERICAL covariance
            #action_covariance += torch.eye(self.action_space, device=self.device) * 1e-2
            action_mu = action_mu.to(self.device) #TODO: this should naturally happen via the MLP
            #print("Mu: %s \n Covariance: %s\n" % (action_mu, action_covariance))
            normal = torch.distributions.MultivariateNormal(action_mu, action_covariance)
            if hasattr(self, 'policy_entropy_history'):
                self.policy_entropy_history.append(self.get_policy_entropy(action_sigma))
            if hasattr(self, 'energy_history'):
                self.energy_history.append(self.get_energy_penalty(action_mu, R_mat = None))
            try:
                action = normal.sample()
            except: #resample
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
        if hasattr(self, 'policy_entropy_history'):
            self.policy_entropy_history = []
        if hasattr(self, 'energy_history'):
            self.energy_history = []

    def sample_action(self, obs) -> np.ndarray:
        if self.discrete:
            one_hots = np.eye(self.action_space)
            return one_hots[np.random.choice(self.action_space, 1)]
        else:
            mins = self.action_constraints[0]
            maxs = self.action_constraints[1]
            return np.random.uniform(mins, maxs, (1, self.action_space))


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
    def get_policy_entropy(self, s):
        pass

    @abc.abstractmethod
    def get_energy_penalty(self, a, R_mat = None):
        pass

    @abc.abstractmethod
    def encode_input(self, inp):
        '''Project / encode the input to some alternative space (as a preprocessing method)'''
        pass

    @abc.abstractmethod
    def transform_observation(self, obs, target_size = None, normalize_pixels = True, **kwargs) -> np.ndarray:
        if type(obs) is list:
            obs = np.array(obs)
        if target_size is not None:
            obs.resize(target_size)
        if normalize_pixels == True:
            obs /= 256
        return obs

    @abc.abstractmethod
    def preprocess(self, obs) -> np.ndarray:
        #TODO: add different means of pre-processing
        #assert(self.obs_mean is not None)
        #assert(self.obs_var is not None)
        if self.obs_mean is not None and self.obs_var is not None:
            pass

    @abc.abstractmethod
    def clone(self):
        return Agent(self.input_dimensions, self.action_space,
                self.discrete, self.action_constraints, self.has_value_function,
                self.terminal_penalty)

    @abc.abstractmethod
    def __call__(self, timestep):
        '''NOTE: This is SPECIFICALLY used to interface with dm_control.viewer'''
        pass


class RandomAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random = True
    
    def step(self, obs):
        obs = self.transform_observation(obs)
        return super(RandomAgent, self).step(obs)

    def evaluate(self, obs, *args, **kwargs):    
        return self.sample_action(obs), None, None

    def __call__(self, timestep):
        obs = timestep.observation
        action = self.step(obs)
        if self.discrete: #TODO: This is currently only compatible with action_space = 1
            action_ind = max(range(len(action)), key=action.__getitem__) 
            action = [-1, 1][action_ind] 
        else: #continuous action space, currently mu + sigma
            pass            
        return action

#raise Exception("Implement this, recreate their MPC paper")
class MPCAgent(Agent): 
    def __init__(self, horizon = 50, k_shoots = 1, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.k_shoots = k_shoots
        self.mpc = True
    
    def evaluate(self, obs, *args, **kwargs) -> (np.array, None, None):
        #print("MPC EVALUATE")
        traj = ()
        max_r = -float('inf')
        for k in range(self.k_shoots):
            states, actions, rewards = self.shoot(obs)
            if sum(rewards) > max_r:
                max_r = sum(rewards)
                traj = (states, actions, rewards)
        states, actions, rewards = traj
        #print("ACTIONS: ", actions[0])
        return actions[0], None, None #TODO: ACTION SCORES TOO?!

    def shoot(self, st) -> ([], [], []):
        states = []
        actions = []
        rewards = []
        st = st #wow
        for t in range(self.horizon): 
            at = self.sample_action(st)
            rt = self.reward(st, at)
            states.append(st)
            actions.append(at)
            if rt is None:
                rt = 0.0
            rewards.append(rt)
            st = self.predict_state(st, at)
        return states, actions, rewards

    @abc.abstractmethod
    def predict_state(self, st, at, *args, **kwargs) -> np.ndarray:
        '''Predict the next state based on (st, at) and additional arguments.'''
        pass

    @abc.abstractmethod
    def reward(self, st, at, *args, **kwargs):
        '''Compute reward function based on (st, at) and additional arguments
        Left entirely abstract because I'm confused.'''
        pass


class DMEnvMPCAgent(MPCAgent):
    '''The EnvironmentMPCAgent performs MPC using the observation and 
    reward signals from an environment it interacts with to guide its
    control.'''
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.state = None

    def predict_state(self, st, at, *args, **kwargs) -> np.ndarray:    
        self.set_state(st)
        timestep = self.env.step(at)
        return timestep.observation

    def reward(self, st, at, *args, **kwargs):
        self.set_state(st)
        timestep = self.env.step(at)
        return timestep.reward

    def set_state(self, st):
        assert(self.state is not None)
        self.env.physics.set_state(self.state)
        #st = self.transform_observation(st)
        #self.env.physics.set_state(st)

    def __call__(self, timestep):
        self.state = self.env.physics.get_state()
        obs = timestep.observation
        action = self.step(obs)
        if self.discrete: #TODO: This is currently only compatible with action_space = 1
            action_ind = max(range(len(action)), key=action.__getitem__) 
            action = [-1, 1][action_ind] 
        else: #continuous action space, currently mu + sigma
            pass            
        return action





class PyTorchAgent(Agent):
    def __init__(self, device, *args, **kwargs):
        super(PyTorchAgent, self).__init__(*args, **kwargs)
        self.device = device
        self._module = None
        self.obs_mean = torch.zeros(self.input_dimensions).to(device)
        self.obs_var = torch.zeros(self.input_dimensions).to(device)

    def step(self, obs, transform = True) -> np.ndarray:
        '''Transforms observation, stores relevant data in replay buffers, then
        outputs action. 
        
        Outputs as np.ndarray because that's what the environment
        needs'''
        #print("PyTorch Step")
        if transform:
            obs = self.transform_observation(obs).to(self.device)
        obs_clone = None
        if self.module.seperate_value_module_input == True:
            obs_clone = obs
        if self.encode_inputs:
            self.state_history.append(obs) #store ORIGINAL observation
            obs = self.encode_input(obs)
            detach = True
            if detach:
                obs = obs.detach()
        return super(PyTorchAgent, self).step(obs, value_module_input = obs_clone)
   
    def evaluate(self, obs, *args, **kwargs) -> torch.tensor:
        #online param calculation discussion: https://discuss.pytorch.org/t/normalization-of-input-data-to-qnetwork/14800/2
        #print("PyTorch Evaluate")
        #obs = self.preprocess(obs) 
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
    
    def get_policy_entropy(self, s):
        #raise Exception("Make this accomodate continuous space (use distribution.entropy)")
        if self.discrete:
            # s in this case is prob of taking action
            s = s.squeeze(0)
            log_prob = s.log()
            entropy = (log_prob * s).sum().to(self.device)
            print("Score: %s Entropy: %s"%(action_score, entropy))
            return entropy    
        else: #continuous, based on 1602.01783 section 9
            # s, in this case, is SIGMA    
            entropy = -0.5 * (torch.log(2*math.pi*s)+1)
            return entropy.squeeze(0)

    def get_energy_penalty(self, a, R_mat = None):
        '''Use policy network mu term to compute quadratic penalty
        associated with 'energy' expended in action a. (aT * R * a)'''
        action_size = self.action_space
        if R_mat is None:
            R_mat = torch.eye(action_size, device = self.device)
        penalty = torch.matmul(a.unsqueeze(0), R_mat)
        penalty = torch.matmul(penalty, a.unsqueeze(0).t())
        return penalty.squeeze(0).squeeze(0)

    def encode_input(self, inp):
        encoded = self.encoder.encode(inp) 
        return encoded

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
        #self.compute_normalization(obs)
        #if self.obs_mean is not None and self.obs_var is not None:
        #    return self.normalize(obs)
        return obs
    
    def clone(self) -> Agent:
        return PyTorchAgent(self.device, self.input_dimensions, self.action_space,
                self.discrete, self.action_constraints, self.has_value_function,
                self.terminal_penalty)

    def __call__(self, timestep):
        obs = timestep.observation
        action = self.step(obs).cpu().detach()
        if self.discrete: #TODO: This is currently only compatible with action_space = 1
            action_ind = max(range(len(action)), key=action.__getitem__) 
            action = [-1, 1][action_ind] 
        else: #continuous action space, currently mu + sigma
            pass            
        return action



class PyTorchMPCAgent(MPCAgent, PyTorchAgent):
    def __init__(self, num_processes = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_history = []
        self.shoots = []
        self.num_processes = num_processes
    
    #@property
    #def module(self):
    #    return self._module

    #@module.setter
    #def module(self, m):
    #    #if self.num_processes > 1:
    #    #    self._module = m.cpu()
    #    #else:
    #    self._module = m

    def evaluate(self, obs, *args, **kwargs) -> (np.array, None, None):
        if self.num_processes == 1: #single-process
            return super().evaluate(obs, *args, **kwargs)
        else:
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
            #self.module = self.module.cpu()
            self.module.share_memory() 


            traj = ()
            max_r = -float('inf')
   
            queue = mp.Queue()
            event = mp.Event()
            processes = []
            
            #obs = obs.cpu()
            
            pool = mp.Pool(self.num_processes)
            results = pool.map(self.shoot, [obs for i in range(self.num_processes)])
            print("Results: ", results)
            for k in range(self.k_shoots):
                #obs = obs.cpu()
            #    states, actions, rewards = pool.map(self.shoot, (obs,))
                if sum(rewards) > max_r:
                    max_r = sum(rewards)
                    traj = (states, actions, rewards)
            states, actions, rewards = traj
            pool.close()
            pool.join()
            #print("ACTIONS: ", actions[0])
            return actions[0], None, None #TODO: ACTION SCORES TOO?!

            #for k in range(self.k_shoots):
            #    results = queue.get()
            #    rewards = results['rewards']
            #    actions = results['actions']
            #    forward = results['forward'] #TODO: incorporate into self.shoots for training!
            #    #print("TRAJ: ", traj)
            #    #states, actions, rewards = traj
            #    if sum(rewards) > max_r:
            #        max_r = sum(rewards)
            #        traj = (forward, actions, rewards)
            
            obs = obs.cpu()
            obs.share_memory_()
            for k in range(self.k_shoots):
            #for k in range(self.num_processes):
                p = mp.Process(target=self.shoot_async, args=(obs,queue,event))
                p.start()
                processes.append(p)
                #states, actions, rewards = self.shoot(obs)
            for k in range(self.k_shoots):
                results = queue.get()
                rewards = results['rewards']
                actions = results['actions']
                forward = results['forward'] #TODO: incorporate into self.shoots for training!
                #print("TRAJ: ", traj)
                #states, actions, rewards = traj
                if sum(rewards) > max_r:
                    max_r = sum(rewards)
                    traj = (forward, actions, rewards)
            states, actions, forward = traj
            event.set()
            for p in processes:
                p.join()
            #print("ACTIONS: ", actions[0])
            #TODO: set self.module to cuda again?? or is this a permanent thing?
            return actions[0], None, None #TODO: ACTION SCORES TOO?!

    def shoot(self, st, gpu = True) -> ([], [], []):
        states = []
        actions = []
        rewards = []
        for t in range(self.horizon): 
            at = self.sample_action(st, gpu = gpu)
            rt = self.reward(st, at)
            states.append(st)
            actions.append(at)
            if rt is None:
                #print("NONE REWARD")
                rt = 0.0
            rewards.append(rt)
            st = self.predict_state(st, at)
        self.shoots.append(self.forward_history) 
        self.forward_history = []
        #print("Shoots: ", self.shoots)
        return states, actions, rewards

    def shoot_async(self, st, queue, event) -> None:
        states = []
        actions = []
        rewards = []
        
        forward_history = []
        for t in range(self.horizon): 
            at = self.sample_action(st, gpu = False)
            rt = self.reward(st, at)
            states.append(st)
            actions.append(at)
            if rt is None:
                #print("NONE REWARD")
                rt = 0.0
            rewards.append(rt)
            st, forward = self.predict_state(st, at, save = False)
            forward_history.append(forward)
        results = {'rewards':rewards, 'actions':actions,
            'forward':forward_history}
        #print("Results: ", results)
        queue.put(results)
        #print("Process %s waiting on event!" % (os.getpid()))
        event.wait()
        
    def predict_state(self, st, at, save = True, *args, **kwargs) -> np.ndarray:
        '''Predict the next state based on (st, at) and additional arguments.
        Makes use of PyTorch MLP, taking in st, at and computing f(st, at) where
        st+1 = st + f(st, at)
        '''
        #st = st.view(st.size(-1), -1)
        #st = st.view(-1, st.numel())
        #st = torch.t(st)
        #st = st.squeeze(0)
        #print("St: %s at: %s" % (st, at))
        inp = torch.cat((st, at), 0) 
        #inp = torch.cat( at), 0) 
        forward = self.module(inp)
        st_1 = st + forward
        if save:
            self.forward_history.append(forward.detach()) 
            return st_1
        else:
            return st_1, forward
    
    def reset_histories(self):
        super().reset_histories()
        self.forward_history = []
        self.shoots = []

    def sample_action(self, obs, gpu = True) -> np.ndarray:
        a = super().sample_action(obs)
        if gpu:
            return torch.tensor(a, device = self.device).float().squeeze(0)
        else:
            return torch.tensor(a).cpu().float().squeeze(0)
    
    def clone(self) -> Agent:
        return PyTorchMPCAgent(self.num_processes, self.horizon, self.k_shoots,
                self.device, self.input_dimensions, self.action_space,
                self.discrete, self.action_constraints, self.has_value_function,
                self.terminal_penalty)

    def __call__(self, timestep):
        obs = timestep.observation
        action = self.step(obs).cpu().detach()
        if self.discrete: #TODO: This is currently only compatible with action_space = 1
            action_ind = max(range(len(action)), key=action.__getitem__) 
            action = [-1, 1][action_ind] 
        else: #continuous action space, currently mu + sigma
            pass            
        return action



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


    def forward(self, x, value_input = None):
        for l in range(len(self.layers)):
            x = self.layers[l](x)
        return x


class PyTorchLinearAutoencoder(torch.nn.Module):
    def __init__(self, indim, device, depth, activations : [] = [], encoded_activations = [], 
            reduction_factor = 0.75, uniform_layers = False):
        '''Arguments:
            @activations: Determines the activations at each layer (?except the first and last layer?).
            @Depth: Determines the number of layers in the encoder / decoder layer.
            @Reduction_factor: Determines how much smaller / larger each layer is from the last. This
            also informs the size of the encoded space, which equals indim * floor((factor)^depth)'''
        super(PyTorchLinearAutoencoder, self).__init__()
        self.activations = activations
        self.encoded_activations = encoded_activations
        self.reduction = reduction_factor
        self.encoded_space = math.floor(indim * self.reduction ** depth)
        self.depth = depth
        print("DEPTH: ", self.depth)
        print("Indim: ", indim)
        self.device = device
        self.indim = indim
        #assert(len(activations) == len(encoder_layers + decoder_layers) + 1)
        
        layers = []
        prev_size = indim
        ## create encoder
        for i in range(depth):
            if uniform_layers:
                size = prev_size
            elif i < depth - 1:
                size = math.floor(prev_size * self.reduction) 
            if i >= depth - 1:
                size = math.floor(indim * self.reduction ** depth)
            linear = torch.nn.Linear(math.floor(prev_size), size, bias = True)
            layers.append(linear)
            functions = activations if i < depth - 1 else encoded_activations
            for a in functions:
                if a == 'relu':
                    layers.append(torch.nn.LeakyReLU())
                elif a == 'sig':
                    layers.append(torch.nn.Sigmoid())
            if not uniform_layers:
                prev_size = prev_size * self.reduction #necessary to carry floating point
        layers = [l.to(device) for l in layers]
        self.encoder = torch.nn.ModuleList(layers)
        ##create decoder
        layers = []
        for i in range(depth):
            if uniform_layers:
                if i == 0:
                    size = prev_size
                    prev_size = math.floor(indim * self.reduction ** depth)
                else:
                    size = indim
                    prev_size = size
            elif i < depth - 1:
                size = math.floor(prev_size / self.reduction) 
            else:
                size = indim
            linear = torch.nn.Linear(math.floor(prev_size), size, bias = True)
            layers.append(linear)
            functions = activations if i < depth - 1 else encoded_activations
            for a in activations:
                if i >= depth - 1: #output should HAVE NO NONLINEARITY?
                    break
                if a == 'relu':
                    layers.append(torch.nn.LeakyReLU())
                elif a == 'sig':
                    layers.append(torch.nn.Sigmoid())
            if not uniform_layers:
                prev_size = prev_size / self.reduction #necessary to carry floating point
        layers = [l.to(device) for l in layers]
        self.decoder = torch.nn.ModuleList(layers)
    
    def encode(self, x):
        if len(self.encoder) > 0:
            for l in range(len(self.encoder)):
                x = self.encoder[l](x)
            #print("Len: %s Encoded Space: %s" % (len(x), self.encoded_space))
            assert(len(x) == self.encoded_space)
        return x

    def decode(self, x):
        #print("Inp: ", len(x))
        #print("Encoded: ", self.encoded_space)
        if len(self.decoder) > 0:
            assert(len(x) == self.encoded_space)
            for l in range(len(self.decoder)):
                x = self.decoder[l](x)
        return x

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
            


class PyTorchLinearUnetEncoder(torch.nn.Module):
    def __init__(self, indim, device, depth, activations : [] = [], encoded_activations = [], 
            reduction_factor = 0.75):
        '''Arguments:
            @activations: Determines the activations at each layer (?except the first and last layer?).
            @Depth: Determines the number of layers in the encoder / decoder layer.
            @Reduction_factor: Determines how much smaller / larger each layer is from the last. This
            also informs the size of the encoded space, which equals indim * floor((factor)^depth)'''
        self.activations = activations
        self.encoded_activations = encoded_activations
        self.encoded_space = indim * math.floor(reduction ** depth)
        self.depth = depth
        self.reduction = reduction_factor
        self.device = device
        self.indim = indim
        #assert(len(activations) == len(encoder_layers + decoder_layers) + 1)

def LinearSAAutoencoder(encoder_base, state_size, action_size, forward_mlp,
        coupled_sa = False,
        forward_dynamics = False, *args, **kwargs):
    '''Construct an Autoencoder, encoding State/Action pairs into some space, 
    and decoding the same state/action pair from that space. 
    
    Arguments: 
        @coupled_sa: Indicates whether or not the State/Action encoding should be
        coupled or decoupled. 
        @forward_state: Determines whether or not the autoencoder also outputs
        x_t+1, computed from the (xt, at).
        @forward_dynamics: Determines whether or not the autoencoder encodes
        x_t+1 = xt + f(encoded(xt, at)), the next state is represented as 
        some additive term.
        @linear_forward: Imposes a limitation in which the forward_dynamics are
        computed linearly: x_t+1 = xt + A*encoded_state(xt, at) + B*encoded_action(xt, at)
        This limitation should be investigated for the potential of imposing linearity on the
        inner space.'''
    class PyTorchLinearSAAutoencoder(encoder_base):
        def __init__(self, state_size, action_size, forward_mlp, 
                coupled_sa = False, forward_dynamics = False, 
                *args, **kwargs):
            self.forward_dynamics = forward_dynamics
            self.coupled_sa = coupled_sa

            self.action_size = action_size
            self.state_size = state_size
            
            indim = state_size if not coupled_sa else state_size + action_size
            if coupled_sa:
                super().__init__(indim, *args, **kwargs)
            else:
                super().__init__(indim, *args, **kwargs) #treat encode, decode as SPECIFICALLY for state
                if action_size == 1: #identity module
                    a = [args[0], 0]
                    a.extend(args[2:])
                    print('ARGS: ', args)
                    print("A: ", a)
                    args = a
                #else:
                self.action_ae = encoder_base(action_size, *args, **kwargs) #create identical second encoder for actions
            
            self.forward_mlp = forward_mlp #LINEAR MLP structure if the desire is to constrain encoded space to linear function
             

        def forward(self, s, a):
            if self.coupled_sa:
                inp = torch.cat((s, a), 0) 
                decoded = super().forward(inp)
                #seperate s / a
                decoded_state, decoded_action = self.separate_decoded(decoded)
            else:
                decoded_state = super().forward(s)
                decoded_action = self.action_ae(a)
            return decoded_state, decoded_action    

        def separate_encoded(self, e):
            assert(self.coupled_sa)
            encoded_state_size = math.floor(self.state_size * self.reduction**self.depth)
            return e[:encoded_state_size], e[encoded_state_size:]
            #raise Exception('self.state_size * REDUCTION**DEPTH, self.action-size * REDUCTION**DEPTH')

        def separate_decoded(self, d):
            assert(self.coupled_sa)
            return d[:self.state_size], d[self.state_size:]

        def forward_encode(self, s, a, detach = False):
            '''Maps ENCODED s / a to an ENCODED s_t+1.'''
            if self.coupled_sa:
                inp = torch.cat((s, a), 0) 
                encoded = super().encode(inp)
                s_, a_ = self.separate_encoded(encoded) 
            else:
                s_ = super().encode(s)
                a_ = self.action_ae.encode(a)
            if detach:
                s_ = s_.detach()
                a_ = a_.detach()
            inp = torch.cat((s_, a_), 0) 
            f = self.forward_mlp(inp) 
            if self.forward_dynamics:
                return s_ + f
            else:
                return f

        def forward_predict(self, s, a, detach = True):
            '''Maps full s/a to s_t+1 (not encoded s_t+1)'''
            f = self.forward_encode(s, a, detach)      
            if self.coupled_sa:
                inp = torch.cat((f, a), 0) 
                #print("Len: %s Encoded Space: %s" % (len(inp), self.encoded_space))
                decoded = self.decode(inp)
                s_d, a_d = self.separate_decoded(decoded)
                return s_d
            else:
                decoded_state = super().decode(f)
                return decoded_state

    return PyTorchLinearSAAutoencoder(state_size, action_size, 
            forward_mlp, coupled_sa, forward_dynamics,
            *args, **kwargs)



class PyTorchDiscreteACMLP(PyTorchMLP):
    '''Adds action / value heads to the end of an MLP constructed
    via PyTorchMLP '''
    def __init__(self, action_space, seperate_value_module = None,
            seperate_value_module_input = False, 
            value_head = True, 
            action_bias = True, value_bias = True,
            *args, **kwargs):
        self.action_space = action_space
        super(PyTorchDiscreteACMLP, self).__init__(*args, **kwargs)
        self.seperate_value_module_input = seperate_value_module_input
        self.sigma_head = sigma_head
        if seperate_value_module is not False: #currently, this creates an IDENTICAL value function
            print("seperate_value_module")
            if seperate_value_module is None:
                print("self.value_mlp = PyTorchMLP")
                self.value_mlp = PyTorchMLP(*args, **kwargs)
            else:
                self.value_mlp = seperate_value_module
        if seperate_value_module is False or value_head is True: #we don't need this as an attribute, do we?
            self.value_module = torch.nn.Linear(self.outdim, 1, 
                    bias = value_bias)
        self.action_module = torch.nn.Linear(self.outdim, 
                action_space, bias = action_bias)
    
    def forward(self, x, value_input = None):
        mlp_out = super(PyTorchDiscreteACMLP, self).forward(x)
        #print("MLP OUT: ", mlp_out)
        actions = self.action_module(mlp_out) 
        action_scores = torch.nn.functional.softmax(actions, dim=-1)
        if self.value_mlp is not False:
            if self.seperate_value_module_input == True:
                mlp_out = self.value_mlp.forward(value_input)
            else:
                mlp_out = self.value_mlp.forward(x)
        if hasattr(self, 'value_module'):
            value = self.value_module(mlp_out)
        #print("ACTION: %s VALUE: %s" % (actions, value))
        return action_scores, value

class PyTorchContinuousGaussACMLP(PyTorchMLP):
    '''Adds action (mean, variance) / value heads 
    to the end of an MLP constructed via PyTorchMLP '''
    def __init__(self, action_space, seperate_value_module = None,
            seperate_value_module_input = False, value_head = True, 
            action_bias = True, value_bias = True, sigma_head = False,
            *args, **kwargs):
        self.action_space = action_space
        super(PyTorchContinuousGaussACMLP, self).__init__(*args, **kwargs)
        self.seperate_value_module_input = seperate_value_module_input
        self.sigma_head = sigma_head
        if seperate_value_module is not False: #currently, this creates an IDENTICAL value function
            print("seperate_value_module")
            if seperate_value_module is None:
                print("self.value_mlp = PyTorchMLP")
                self.value_mlp = PyTorchMLP(*args, **kwargs)
            else:
                self.value_mlp = seperate_value_module
        if seperate_value_module is False or value_head is True: #we don't need this as an attribute, do we?
            self.value_module = torch.nn.Linear(self.outdim, 1, 
                    bias = value_bias)
        self.action_mu_module = torch.nn.Linear(self.outdim, 
                action_space, bias = action_bias)
        self.action_sigma_module = torch.nn.Linear(self.outdim,
                1, bias = action_bias)
        self.action_sigma_softplus = torch.nn.Softplus()
    
    def forward(self, x, value_input = None):
        mlp_out = super(PyTorchContinuousGaussACMLP, self).forward(x)
        #print("MLP OUT: ", mlp_out)
        action_mu = self.action_mu_module(mlp_out) 
        if self.sigma_head:
            action_sigma = self.action_sigma_module(mlp_out)
            action_sigma = self.action_sigma_softplus(action_sigma) + 0.01 #from 1602.01783 appendix 9
        else: #assume sigma is currently zeros
            action_sigma = torch.ones([1, self.action_space], dtype=torch.float32) * 0.0001
        if hasattr(self, 'value_mlp'):
            if self.seperate_value_module_input == True:
                mlp_out = self.value_mlp.forward(value_input)
            else:
                mlp_out = self.value_mlp.forward(x)
        if hasattr(self, 'value_module'):
            value = self.value_module(mlp_out)
        else:
            assert(hasattr(self, 'value_mlp'))
            value = mlp_out #assuming value_mlp creates value
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

        def forward(self, x, value_input = None):
            if self.base == PyTorchDiscreteACMLP: #TODO: can't assume value function exists...??
                action_score, values = super(PyTorchEpsGreedyModule,
                        self).forward(x, value_input)
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
                        self).forward(x, value_input)
                if random.random() < self.eps: #randomize sigma values
                    self.update_eps()
                    #mins = self.action_constraints[0]
                    #maxs = self.action_constraints[1]
                    #action_mu =  np.random.uniform(mins, maxs, (1, self.action_space))
                    #action_sigma = np.random.random_integers(1, high = 2, size = (1, self.action_space))
                    #eps_sigma = np.random.uniform(low = 0.01, high = 3.0, size = (1, 1))
                    #action_mu = torch.as_tensor(action_mu).float()
                    #noise = action_sigma.clone().uniform_(0, 3)
                    noise = action_sigma.clone().uniform_(0, 3)
                    #print("Current sigma: ", action_sigma)
                    action_sigma += noise
                    #print("New Sigma: %s Eps: %s"%(action_sigma, self.eps))
                return action_mu, action_sigma, values 
                raise Exception("Figure this out?")

    return PyTorchEpsGreedyModule(mlp_base, eps, eps_decay, eps_min, action_constraints, *args, **kwargs)



##  Tensorflow Helper functions/classes

#NOTE using https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py as reference

class TFAgent(Agent):
    def __init__(self):
    ##NOTE:We simply feed the numpy transform_observation into feed_dict
    ##for TFAgent
        pass

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
