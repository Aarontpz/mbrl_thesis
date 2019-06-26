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


from ddp import *
from model import *
from module import *

import sklearn.neighbors as neighbors 
from sklearn import linear_model
from sklearn.cluster import MiniBatchKMeans, Birch


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
        #avg_reward = sum(self.reward_history) / len(self.reward_history)
        ind = [i for i in range(len(self.terminal_history)) if self.terminal_history[i] == 1] #get all terminated indices
        #print("Indices: ", ind)
        ind = 0 if len(ind) <= 1 else ind[-2] #get previous terminated indice
        #print("Index: ", ind)
        avg_reward = sum(self.reward_history[ind:])
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
            return np.random.uniform(mins, maxs, (self.action_space))


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
        print("ObservatioN: ", obs)
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
    
    def __call__(self, timestep):
        '''Default (inherited) __call__ is designed to work within dm_control suite, handling "timestep" from dm_control environment..'''
        obs = timestep.observation
        obs = self.transform_observation(obs)
        action = self.step(obs)
        if self.discrete: #TODO: This is currently only compatible with action_space = 1
            action_ind = max(range(len(action)), key=action.__getitem__) 
            action = [-1, 1][action_ind] 
        else: #continuous action space, currently mu + sigma
            pass            
        return action
    


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
        #if self.encode_inputs and # Ensure that inputs are encoded if necessary
        return super(PyTorchAgent, self).evaluate(obs, *args, **kwargs)

    def terminate_episode(self):
        if self.module.rec_size > 0:
            self.module.reset_states()
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


class DDPMPCAgent(MPCAgent):
    def __init__(self, mpc_ddp : ILQG, 
            reuse_shoots = True, eps_base = 1e-1, eps_min = 2e-2,
            eps_decay = 1e-7,
             *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.mpc_ddp = mpc_ddp
        self.reuse_shoots = reuse_shoots
        self.prev_states = []
        self.prev_actions = []
        self.reuse_ind = 0
        self.deviation_threshold = 1e-1

        self.eps = eps_base
        self.eps_base = eps_base
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def update_target(self, env):
        pass

    def evaluate(self, obs, *args, **kwargs) -> (np.array, None, None):
        if hasattr(self, 'env'):
            self.update_target(self.env) #for tasks with variable targets
        if random.random() < self.eps:
            self.eps = max(self.eps_min, self.eps - self.eps_decay)
            #print("EPS ACTION: ", self.sample_action(obs))
            #flush stored state/action pairs
            self.prev_states = []
            self.prev_actions = []
            return self.sample_action(obs), None, None
        else:
            if isinstance(self.mpc_ddp, SMC):
                _, action = self.mpc_ddp.step(obs) 
                return action.flatten(), None, None
            action = None
            if not self.reuse_shoots:
                return super().evaluate(obs, *args, **kwargs)
            else:
                if self.reuse_criterion(obs, *args, **kwargs) and (self.reuse_ind) < len(self.prev_actions):
                    #print("Reusing previous MPC trajectory!")
                    #print("Actions: ", self.prev_actions)
                    action = self.prev_actions[self.reuse_ind]
                else:
                    #print("Generating new MPC trajectory!")
                    self.reuse_ind = 0
                    traj = ()
                    max_r = -float('inf')
                    for k in range(self.k_shoots):
                        states, actions, rewards = self.shoot(obs)
                        if sum(rewards) > max_r:
                            max_r = sum(rewards)
                            traj = (states, actions, rewards)
                    states, actions, rewards = traj
                    self.prev_states = states
                    self.prev_actions = actions
                    action = actions[self.reuse_ind]
                self.reuse_ind += 1
            return action, None, None

    def reuse_criterion(self, obs, *args, **kwargs):
        #if not self.reuse_ind < len(self.prev_actions):
        #    return False
        #return np.linalg.norm(obs - self.prev_states[self.reuse_ind]) < self.deviation_threshold
        return True
        
    def shoot(self, st) -> ([], [], []):
        rewards = [0 for i in range(int(self.horizon/self.mpc_ddp.dt))]
        st = st #wow
        states, actions = self.mpc_ddp.step(st)
        #input("We step!")
        rewards[-1] = -1 * self.mpc_ddp.forward_cost(states, actions)
        #we seek to MAX rewards = MIN cost (-1 * COST)
        return states, actions, rewards

    def reward(self, X, U, *args, **kwargs):
        return self.mpc_ddp.forward_cost(X, U)

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

##  Pytorch helper functions /classes



## PyTorch-specific Model elements (from ilqg.py)
class PyTorchModel(Model):
    '''Wraps a Pytorch Module in a Model so it can be painlessly
    integrated with iLQG.'''
    def __init__(self, module : PyTorchMLP, dt, *args, **kwargs):
        self.module = module
        self.dt = dt

    @abc.abstractmethod
    def forward(self, xt, ut, dt, *args, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, xt):
        pass

    def forward_predict(self, xt, ut, dt):
        if type(xt) == np.ndarray:
            xt = torch.tensor(xt, requires_grad = True, device = self.module.device).float()
            xt = xt.unsqueeze(0)
            #xt = xt.transpose(0, 1)
        return xt + dt * self.forward(xt, ut, *self.module(xt, ut)) 
    #resnet style <3

class LocalModel(Model):
    '''LocalModel for local (linear) models, for use with DDP/SMC
    solutions, instead of global (neural network) nonlinear models. 
    '''
    def __init__(self, dt = 0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.models = {}

    @abc.abstractmethod
    def get_local_model(self, x):
        '''Return local model corresponding to state x, based on 
        local partitioning of state space.'''
        pass

    @abc.abstractmethod
    def fit(self, X):
        '''Fit data from state-space ("s" in dataset's samples, of form
        s, a, r, s_) in online fashion. This should produce new regions
        as necessary.'''
        pass

class ClusterLocalModel(LocalModel):
    '''Utilize online clustering algorithm (Birch clustering,
    which is proposed as an alternative to mini-batch k-means)
    to group points spatially.'''
    def __init__(self, neighbors = 0,
            n_clusters = 50,
            compute_labels = True,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        #self.cluster = Birch(threshold = threshold, 
        #        branching_factor = branching_factor,
        #        n_clusters = n_clusters, 
        #        compute_labels = compute_labels)
        self.cluster = MiniBatchKMeans(n_clusters = n_clusters)
        
        self.region_s = {} #label : [s]
        self.region_a = {} #label : [a]
        self.region_s_ = {} #label : [s_]
        
        self.max_points = 2000

        self.update_clusters = True
        
        self.neighbors = neighbors
        self.region_knn = {}

        self.prev_region = None

    def get_local_model(self, x):
        if x.shape[0] > x.shape[1]: #this is a next-level disgusting hack
            x = x.T
        #print("X shape (glm): ", x.shape)
        label = self.cluster.predict(x)[0]
        #print("LABEL: ", label)
        #print("Models: ", self.models.keys())
        if label in self.models.keys():
            return self.models[label]
        else:
            print("Invalid Label: ", label)
            return None

    def fit_cluster(self, X, X_, U = None):
        self.cluster.partial_fit(X) #online clustering step
        #self.cluster.partial_fit() #global clustering step


    def predict_cluster(self, X) -> np.ndarray:
        '''Predict cluster for each entry in given set of states X.'''
        labels = self.cluster.predict(X)
        return labels
    
    def update_cluster(self):
        '''We have "datasets" of s, a, s_ by regions. But as new data
        comes in those points may no longer truly belong to that region.
        This systematically (inefficiently) verifies the position of
        each element in each region.'''
        self.prev_region = None
        for r in self.region_s.keys(): #update each region's points 
            if len(self.region_s[r]) > self.max_points:
                num_remove = len(self.region_s[r]) - self.max_points
                print("Region %s Has %s too many points!" % (r, num_remove))
                self.region_s[r] = self.region_s[r][num_remove:]  
                self.region_s_[r] = self.region_s_[r][num_remove:]  
                self.region_a[r] = self.region_a[r][num_remove:]  
            if len(self.region_s[r]) > 0:
                s = np.array(self.region_s[r])
                if len(s.shape) < 2:
                    s = s[..., np.newaxis]
                labels = self.predict_cluster(s)
                for i in reversed(range(len(labels))): 
                    if labels[i] != r: #we move s, a, s_ to appropriate region
                        #print("I: %s R: %s Label: %s" % (i, r, labels[i]))
                        self.region_s[labels[i]].append(self.region_s[r].pop(i))
                        self.region_a[labels[i]].append(self.region_a[r].pop(i))
                        self.region_s_[labels[i]].append(self.region_s_[r].pop(i))
            if self.neighbors > 0 and len(self.region_s[r]) > self.neighbors:
                #print("Knn on : ", self.region_s[r])
                self.region_knn[r] = neighbors.KDTree(self.region_s[r])



    @abc.abstractmethod
    def fit_model(self, X, X_, U = None):
        pass     

    def fit(self, X, X_, U = None):
        '''Fit data into existing model, updating clusters and 
        models as necessary. If only least-squares approximation
        (via sklearn) is used, no need to run trainer in this step.
        ''' 
        self.fit_cluster(X, X_, U)
        labels = self.predict_cluster(X)
        #raise Exception("Update clusters NEEDS to be a param?")
        #raise Exception("KNN vs Cluster fitting??")
        if self.update_clusters is True:
            self.update_cluster()
        for i in range(len(X)): #add samples to each region's dicts
            self.region_s.setdefault(labels[i], []).append(X[i])
            self.region_a.setdefault(labels[i], []).append(U[i])
            self.region_s_.setdefault(labels[i], []).append(X_[i])
            #^ in order to store s and s' (t and t+1 states)
        #print("LABELS: ", labels)
        self.fit_model(X, X_, U)
        #TODO: re-sort / verify existing models after this step?
        #this should be all "Online", not brute-forcey
        return labels

class ForwardClusterLocalModel(ClusterLocalModel, GeneralSystemModel):
    '''Express system dynamics as locally "control affine" within regions
    determined by cluster local model.'''
    def __init__(self, f_shape, g_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_shape = f_shape
        self.g_shape = g_shape
        self.max_points = 4000

    
    def update(self, xt):
        if len(xt.shape) < 2:
            xt = xt.reshape(-1, xt.size)
        if len(self.models) == 0:
            #self.A = np.ndarray(self.a_shape)
            #self.B = np.ndarray(self.b_shape)
            self.initialize_model_free_arrays()
        else:
            #if xt.shape[1] > xt.shape[0]:
            #    xt = xt.T
            region = self.predict_cluster(xt.T)
            #print("PREV REGION: %s REGION: %s" % (self.prev_region, region))
            if self.prev_region is not None:
                if region == self.prev_region:
                    return self.f, self.g
            self.prev_region = region

            model = self.get_local_model(xt)
            if model is None: #TODO: temporary measure
                print("INVALID xt (no label / no matching model)")
                #raise Exception("REEEE fix this") #TODO TODO TODO TODO
                self.initialize_model_free_arrays()
            else:
                self.f, self.g = self.get_forward_model(model, xt)
                #print("f: %s" % (self.f))
                #input()
        return self.f, self.g


    def initialize_model_free_arrays(self):
        self.f = np.random.rand(*self.f_shape)
        self.g = np.random.rand(*self.g_shape)
        #self.B = np.ones(self.b_shape)

    @abc.abstractmethod
    def fit_model(self, X, X_, U = None):
        pass     

    @abc.abstractmethod
    def get_linear_model(self, model, xt = None, ut = None) -> (np.ndarray, np.ndarray):
        '''Retrieve local parameters from local model'''
        pass


class LinearClusterLocalModel(ClusterLocalModel, LinearSystemModel):
    '''Express system dynamics as locally linear within regions determined by
    cluster local model.'''
    def __init__(self, a_shape, b_shape,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.fit_b = True

    def update(self, xt):
        if len(xt.shape) < 2:
            xt = xt.reshape(-1, xt.size)
        if len(self.models) == 0:
            #self.A = np.ndarray(self.a_shape)
            #self.B = np.ndarray(self.b_shape)
            self.initialize_model_free_arrays()
        else:
            if xt.shape[1] > xt.shape[0]:
                xt = xt.T
            print("Xt: ", xt.shape)
            region = self.predict_cluster(xt.T)
            #print("PREV REGION: %s REGION: %s" % (self.prev_region, region))
            if self.prev_region is not None:
                if region == self.prev_region:
                    return self.A, self.B
            self.prev_region = region

            model = self.get_local_model(xt)
            if model is None: #TODO: temporary measure
                print("INVALID xt (no label / no matching model)")
                #raise Exception("REEEE fix this") #TODO TODO TODO TODO
                self.initialize_model_free_arrays()
            else:
                self.A, self.B = self.get_linear_model(model, xt)
        #self.A += np.eye(self.A.shape[0])
        #self.B += np.eye(self.A.shape[0], M=self.B.shape[1])
        #input()
        return self.A, self.B


    def initialize_model_free_arrays(self):
        self.A = np.random.rand(*self.a_shape)
        self.B = np.random.rand(*self.b_shape)
        #self.B = np.ones(self.b_shape)

    @abc.abstractmethod
    def fit_model(self, X, X_, U = None):
        pass     

    @abc.abstractmethod
    def get_linear_model(self, model, xt = None, ut = None) -> (np.ndarray, np.ndarray):
        '''Retrieve local parameters from local model'''
        pass

class PyTorchForwardClusterLocalModel(ForwardClusterLocalModel):
    '''Utilize online clustering algorithm (Birch clustering,
    which is proposed as an alternative to mini-batch k-means)
    to group points spatially, apply linear fit (SINGLE
    LAYER PERCEPTRON, representing linear function) 
    to each cluster in order to create piecewise-linear 
    model of system dynamics.'''
    def __init__(self, mlp_layer, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.mlp_layer = mlp_layer
        self.linear_g = True

    def get_forward_model(self, model, xt = None, ut = None):
        '''Assumes local models are "forward dynamics", where F 
        and G and calculated seperately and are represented by
        properly-sized matrices.'''
        f, g = model.forward(xt) 
        f = f.cpu().detach().numpy()
        if self.linear_g:
            g = model.du(xt, ut)
        else:
            g = g.cpu().detach().numpy()
        return f, g

    def get_local_model(self, x):
        if x.shape[0] > x.shape[1]: #this is a next-level disgusting hack
            x = x.T
        model = super().get_local_model(x)
        label = self.cluster.predict(x)[0]
        if self.neighbors > 0 and label in self.region_knn.keys(): #train model on local neighbors
            dist, ind = self.region_knn[label].query(x, k=self.neighbors)
            ind = ind[0] #nested list for some reason...
            X = [self.region_s[label][i] for i in ind]
            X_ = [self.region_s_[label][i] for i in ind]
            U = [self.region_a[label][i] for i in ind]
            model.fit(X, X_, U, num_iters = 5, lr=1e-1) #overfit on neighbors
        return model


    def construct_model(self):
        class PyTorchForwardModule(torch.nn.Module):
            def __init__(self, linear_g, mlp_layer, device, dt, obs_size, action_size):
                super().__init__()
                self.device = device
                
                self.mlp_layer = mlp_layer
                self.linear_g = linear_g
                if self.mlp_layer:
                    self.f_layer = PyTorchMLP(device, obs_size, 
                            obs_size, hdims = [200],
                            activations = [None, 'relu'], bias = True)
                else:
                    self.f_layer = torch.nn.Linear(obs_size, obs_size, bias = False)
                if self.mlp_layer and not self.linear_g:
                    self.g_layer = PyTorchMLP(device, obs_size,
                            obs_size * action_size, hdims = [100, 300,], 
                            #activations = ['relu', None])
                            activations = [None, None], bias = True)
                else:
                    self.g_layer = PyTorchMLP(device, action_size,
                            obs_size, hdims = [100,], 
                            activations = [None, None], bias = False)
                            #obs_size, hdims = [], 
                            #activations = [None], bias = False)
                    #self.g_layer = torch.nn.Linear(action_size, obs_size, bias = False)
                    
                self.f_layer.to(device)
                self.g_layer.to(device)
                self.dt = dt
                
                self.obs_size = obs_size
                self.action_size = action_size
                
                #self.optimizer = torch.optim.SGD(self.parameters(), lr = lr, momentum = 1e-4)
                self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3, betas = (0.9, 0.999))
                

            def forward(self, x, u=None):
                #print("u: ", u)
                if len(x.shape) > 1:
                    x = x.flatten()
                if self.device == torch.device('cuda'):
                    x = torch.tensor(x, device = self.device).float()
                #print("X: ", x.shape)
                #print("Weight: ", self.f_layer.layers[0].weight.shape)
                f = self.f_layer(x)
                if self.linear_g:
                    g = None
                else:
                    g = self.g_layer(x)
                    g = g.reshape((self.obs_size, self.action_size))
                return f, g

            def forward_predict(self, x, u = None):
                '''Return x' = x + dt(Ax + Bu)'''
                if self.device == torch.device('cuda'):
                    x = torch.tensor(x, device = self.device)
                    u = torch.tensor(u, device = self.device)
                f, g = self.forward(x, u)
                if self.linear_g:
                    g = self.g_layer(u)
                    forward = f + g
                else:
                    forward = f + torch.mv(g, u)
                #print("f: ", f)
                #print("g: ", g)
                #print("Forward: ", forward.shape)
                #return x + self.dt * (forward)
                return x + torch.mul(self.dt, (forward))
            
            def du(self, xt, u = None, create_graph = True):
                #dm_du = grad(self.b_layer, xt, create_graph = True)
                #dm_du = dm_du.detach().numpy()
                dm_du = np.eye(self.action_size)
                for l in (self.g_layer.layers):
                    #print("LAYER: ", l)
                    #print("Weight Shape", l.weight.shape) 
                    if self.device == torch.device('cuda'):
                        weight = l.weight.cpu().detach().numpy()
                    else:
                        weight = l.weight.detach().numpy()
                    dm_du = np.dot(weight, dm_du) 
                return dm_du
            
            def fit(self, X, X_, U, num_iters = 20, lr=2e-3):
                '''We're being sloppy here. This is a training step to
                allow this to work with the existing LocalLinearModel trainer
                '''
                criterion = torch.nn.MSELoss()
                for ii in range(num_iters): #we're overfitting...intentionally?
                    loss = torch.tensor([0.0], requires_grad = False).to(self.device)
                    for i in range(len(X)):
                        s_ = torch.tensor(X_[i]).float().to(self.device)
                        s = torch.tensor(X[i]).float()
                        a = torch.tensor(U[i]).float()
                        #print("s: %s \n a: %s s_: %s" % (s, a, s_))
                        prediction = self.forward_predict(s, a)
                        #print("prediction: ", prediction)
                        loss += criterion(s_, prediction)    
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()


        return PyTorchForwardModule(self.linear_g, self.mlp_layer, self.device, self.dt, self.f_shape[0], self.g_shape[1]) 
    
    def fit_model(self, X, X_, U = None):
        #Multiple samples here, instead of overfitting in model.fit?
        for r in self.region_s.keys(): #update each region's model
            print("Training region: %s" % (r))
            if len(self.region_s[r]) > 1: #cannot fit on empty/1 samples
                if r not in self.models.keys():
                    print("Constructing model!")
                    self.models[r] = self.construct_model()
                print("Training model!")
                for training_iter in range(8):
                    ##Randomly sample some points in region
                    num_samples = 64
                    indices = random.sample(range(len(self.region_s[r]) - 1), min(num_samples, len(self.region_s[r]) - 1))
                    sample_s = [self.region_s[r][i] for i in indices]
                    sample_a = [self.region_a[r][i] for i in indices]
                    sample_s_ = [self.region_s_[r][i] for i in indices]
                    #raise Exception("Only train on recent samples from each region? Otherwise, as regions DEVELOP, samples which no longer are representitive are left????")
                    self.models[r].fit(sample_s, sample_s_, sample_a, num_iters = 1, lr=1e-3) #TODO: use ALL points in reg.

class PyTorchLinearClusterLocalModel(LinearClusterLocalModel):
    '''Utilize online clustering algorithm (Birch clustering,
    which is proposed as an alternative to mini-batch k-means)
    to group points spatially, apply linear fit (SINGLE
    LAYER PERCEPTRON, representing linear function) 
    to each cluster in order to create piecewise-linear 
    model of system dynamics.'''
    def __init__(self, mlp_layer, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.mlp_layer = mlp_layer

    def get_linear_model(self, model, xt = None, ut = None):
        '''Assumes local models are "forward dynamics", where F 
        and G and calculated seperately and are represented by
        properly-sized matrices.'''
        if self.mlp_layer:
            xt = xt if xt is not None else np.zeros(self.a_shape[0])
            #print("GET LINEAR MODEL")
            #input()
            A = model.dx(xt)
            B = model.du(xt)
        else:
            #print("NOT MLP PLALERY")
            #input()
            A = model.a_layer.weight.detach().numpy()
            B = model.b_layer.weight.detach().numpy()
        return A,B 
    
    def get_local_model(self, x):
        if x.shape[0] > x.shape[1]: #this is a next-level disgusting hack
            x = x.T
        model = super().get_local_model(x)
        label = self.cluster.predict(x)[0]
        if self.neighbors > 0 and label in self.region_knn.keys(): #train model on local neighbors
            dist, ind = self.region_knn[label].query(x, k=self.neighbors)
            ind = ind[0] #nested list for some reason...
            X = [self.region_s[label][i] for i in ind]
            X_ = [self.region_s_[label][i] for i in ind]
            U = [self.region_a[label][i] for i in ind]
            model.fit(X, X_, U, num_iters = 5, lr=1e-1) #overfit on neighbors
        return model

    def construct_model(self):
        class PyTorchLinearModule(torch.nn.Module):
            def __init__(self, device, dt, obs_size, action_size, mlp_layer = True):
                super().__init__()
                self.device = device
                
                self.mlp_layer = mlp_layer
                if self.mlp_layer:
                    self.a_layer = PyTorchMLP(device, obs_size, 
                            obs_size, hdims = [150,],
                            activations = [None, None], bias = False)
                    self.b_layer = PyTorchMLP(device, action_size,
                            obs_size, hdims = [150,], 
                            #activations = ['relu', None])
                            activations = [None, None], bias = False)
                    #self.a_layer = PyTorchMLP(device, obs_size, 
                    #        obs_size, hdims = [100,],
                    #        activations = [None, None], bias = False)
                    #self.b_layer = PyTorchMLP(device, action_size,
                    #        obs_size, hdims = [100,], 
                    #        #activations = ['relu', None])
                    #        activations = [None, None], bias = False)
                else:
                    self.a_layer = torch.nn.Linear(obs_size, obs_size, bias = False)
                    self.b_layer = torch.nn.Linear(action_size, obs_size, bias = False)
                    
                self.a_layer.to(device)
                self.b_layer.to(device)
                self.dt = dt
                
                self.obs_size = obs_size
                self.action_size = action_size

            def forward(self, x, u):
                if self.device == torch.device('cuda'):
                    x = torch.tensor(x, device = self.device)
                    u = torch.tensor(u, device = self.device)
                ax = self.a_layer(x)
                bu = self.b_layer(u)
                return ax + bu

            def forward_predict(self, x, u):
                '''Return x' = x + dt(Ax + Bu)'''
                if self.device == torch.device('cuda'):
                    x = torch.tensor(x, device = self.device)
                #return x + self.dt * (self.forward(x, u))
                return x + torch.mul(self.dt, (self.forward(x, u)))
            
            def dx(self, xt, u = None, create_graph = True):
                #dm_dx = grad(self.a_layer, xt, create_graph = create_graph)
                #dm_dx = dm_dx.detach().numpy()
                #print("Xt Shape: ", xt.shape)
                dm_dx = np.eye(xt.shape[0])
                #for l in reversed(self.a_layer.layers):
                for l in (self.a_layer.layers):
                    #print("LAYER: ", l)
                    #print("Weight Shape", l.weight.shape) 
                    if self.device == torch.device('cuda'):
                        weight = l.weight.cpu().detach().numpy()
                    else:
                        weight = l.weight.detach().numpy()
                    dm_dx = np.dot(weight, dm_dx) 
                return dm_dx

            def du(self, xt, u = None, create_graph = True):
                #dm_du = grad(self.b_layer, xt, create_graph = True)
                #dm_du = dm_du.detach().numpy()
                dm_du = np.eye(self.action_size)
                for l in (self.b_layer.layers):
                    #print("LAYER: ", l)
                    #print("Weight Shape", l.weight.shape) 
                    if self.device == torch.device('cuda'):
                        weight = l.weight.cpu().detach().numpy()
                    else:
                        weight = l.weight.detach().numpy()
                    dm_du = np.dot(weight, dm_du) 
                return dm_du

            def fit(self, X, X_, U, num_iters = 20, lr=2e-3):
                '''We're being sloppy here. This is a training step to
                allow this to work with the existing LocalLinearModel trainer
                '''
                criterion = torch.nn.MSELoss()
                #optimizer = torch.optim.SGD(self.parameters(), lr = lr, momentum = 1e-4)
                optimizer = torch.optim.Adam(self.parameters(), lr = lr, betas = (0.9, 0.999))
                for ii in range(num_iters): #we're overfitting...intentionally?
                    loss = torch.tensor([0.0], requires_grad = False).to(self.device)
                    for i in range(len(X)):
                        s_ = torch.tensor(X_[i]).float().to(self.device)
                        s = torch.tensor(X[i]).float()
                        a = torch.tensor(U[i]).float()
                        prediction = self.forward_predict(s, a)
                        loss += criterion(s_, prediction)    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


        return PyTorchLinearModule(self.device, self.dt, self.a_shape[0], self.b_shape[1], self.mlp_layer) 
    
    def fit_model(self, X, X_, U = None):
        #Multiple samples here, instead of overfitting in model.fit?
        for r in self.region_s.keys(): #update each region's model
            print("Training region: %s" % (r))
            if len(self.region_s[r]) > 1: #cannot fit on empty/1 samples
                if r not in self.models.keys():
                    print("Constructing model!")
                    self.models[r] = self.construct_model()
                print("Training model!")
                for training_iter in range(5):
                    ##Randomly sample some points in region
                    num_samples = 64
                    indices = random.sample(range(len(self.region_s[r]) - 1), min(num_samples, len(self.region_s[r]) - 1))
                    sample_s = [self.region_s[r][i] for i in indices]
                    sample_a = [self.region_a[r][i] for i in indices]
                    sample_s_ = [self.region_s_[r][i] for i in indices]
                    #raise Exception("Only train on recent samples from each region? Otherwise, as regions DEVELOP, samples which no longer are representitive are left????")
                    self.models[r].fit(sample_s, sample_s_, sample_a, num_iters = 5, lr=5e-2) #TODO: use ALL points in reg.



class SKLearnLinearClusterLocalModel(LinearClusterLocalModel):
    '''Utilize online clustering algorithm (Birch clustering,
    which is proposed as an alternative to mini-batch k-means)
    to group points spatially, apply linear fit (via SKLearn) to each cluster
    in order to create piecewise-linear model of system dynamics.'''
    def get_linear_model(self, model, xt = None):
        #print("MODEL RETRIEVED!")
        augmented = model.coef_ #[A|B] coeff
        #print("Coef: ", self.A)
        if len(self.region_a) > 0:
            A = augmented[:self.a_shape[0], :self.a_shape[1]]
            B = augmented[:self.a_shape[0], self.a_shape[1]:]
        A += np.random.rand(*self.A.shape) * 1e-2
        B += np.random.rand(*self.B.shape) * 1e-2
        return A, B
    
    def fit_model(self, X, X_, U = None):
        for r in self.region_s.keys(): #update each region's model
            print("KEY: ", r)
            if len(self.region_s[r]) > 1: #cannot fit on empty/1 samples
                #self.models[r] = linear_model.LinearRegression(fit_intercept = False)
                self.models[r] = linear_model.LinearRegression(normalize = True)
                #print("State Samples shape: ", self.region_s[r].shape)
                #print("Action Samples shape: ", self.region_u[r].shape)
                #print("NextState Samples shape: ", self.region_s_[r].shape)
                system = []
                if self.fit_b:
                    assert(U is not None)
                    for i in range(len(self.region_s[r])):
                        system.append(np.concatenate((self.region_s[r][i], self.region_a[r][i])))
                else:
                    system = self.region_s[r]
                system = np.array(system)
                print("System shape: ", system.shape)
                print("System[-1]: ", system[-1])

                ##Randomly sample some points in region
                #print("Len: ", len(system))
                indices = random.sample(range(len(system) - 1), min(512, len(system) - 1))
                #print("INDICES: ", indices)
                sample_sa = system[indices]
                sample_s_ = [self.region_s_[r][i] for i in indices]
                #print("Sample s_: ", sample_s_)

                self.models[r].fit(sample_sa, sample_s_) #TODO: use ALL points in reg.
                #print("region r model: ", self.models[r])




class PyTorchForwardDynamicsModel(PyTorchModel, GeneralSystemModel):
    def forward(self, xt, ut, f, g):
        '''Operate on f_, g_ returned from ForwardDynamicsModule in order
        to compute x' = f + g*u, assuming the system is linear w.r.t
        controls'''
        if type(xt) == type(np.array([0])):
            xt = torch.tensor(xt, requires_grad = True, device = self.module.device).float()
            xt = xt.unsqueeze(0)
        if type(xt) == torch.Tensor and xt.size()[0] < xt.size()[1]:
            xt = xt.transpose(0, 1)
        if type(ut) == np.ndarray:
            ut = torch.tensor(ut, requires_grad = True, device = self.module.device).float()
            ut = ut.flatten()
            #ut = ut.unsqueeze(0)
            #ut = ut.transpose(0, 1)
        if self.module.linear_g:
            g = self.module.g_layer(ut)
            return f + g
        g = g.reshape(self.module.b_shape)
        gu = torch.mm(g, ut)
        return f + gu #s.t. xt+1 = xt + dt * (f+gu), linear w.r.t controls

    def update(self, xt):
        if type(xt) == type(np.array([0])):
            xt = torch.tensor(xt, requires_grad = True, device = self.module.device).float()
            xt = xt.flatten()
        f, g = self.module(xt)
        #raise Exception("Todo: conclude this for GeneralModels")
        self.f = f.cpu().detach().numpy()
        #self.f.resize(self.module.a_shape)
        if not self.module.linear_g:
            self.g = g.cpu().detach().numpy()
            self.g.resize(self.module.b_shape)
        else:
            self.g = self.module.du(xt)
        #print("f: ", self.f.shape)
        #print("g: ", self.g.shape)
        return self.f, self.g

class PyTorchLinearSystemModel(PyTorchModel, LinearSystemModel):
    '''Note: Output of PyTorchModel is FLATTENED A/B'''
    def forward(self, xt, ut, a_, b_):
        '''Operate on a_, b_ returned from LinearDynamicsModule in 
        order to compute x' = Ax + Bu'''
        self.update(xt)
        #return LinearSystemModel.__call__(self, xt, ut)
        A = a_.reshape(self.module.a_shape)
        B = b_.reshape(self.module.b_shape)
        #print("A: %s \nB: %s" % (A, B))
        if type(xt) == type(np.array([0])):
            xt = torch.tensor(xt, requires_grad = True, device = self.module.device).float()
            xt = xt.unsqueeze(0)
        if type(xt) == torch.Tensor and xt.size()[0] < xt.size()[1]:
            xt = xt.transpose(0, 1)
        if type(ut) == np.ndarray:
            ut = torch.tensor(ut, requires_grad = True, device = self.module.device).float()
            ut = ut.unsqueeze(0)
            ut = ut.transpose(0, 1)
        #print("A: ", A.shape)
        #print("xt: ", xt.shape)
        ax = torch.mm(A, xt)
        #if len(ax.shape) < 2:
        #    ax = ax.squeeze(0)
        bu = torch.mm(B, ut)
        #print("Ut: ", type(ut))
        #print("Ut: ", ut.shape)
        #print("A: ", A)
        #print("Xt: ", xt)
        #print("A*xt: ", (ax).shape)
        #print("B*ut: ", (bu).shape)
        #print("SUM: ", torch.mm(A, xt) + torch.mm(B, ut))
        return ax + bu #must build computational graph
        #return torch.mm(A, xt) + (b_ * ut) #must build computational graph

    def update(self, xt):
        a_, b_ = self.module(xt)
        self.A = a_.cpu().detach().numpy()
        self.B = b_.cpu().detach().numpy()
        self.A.resize(self.module.a_shape)
        self.B.resize(self.module.b_shape)
        #print("A: ", self.A.shape)
        #print("B: ", self.B.shape)
        return self.A, self.B

