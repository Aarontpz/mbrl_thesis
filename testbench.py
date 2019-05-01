#Author: Aaron Parisi
#3/11/19
import tensorflow as tf
import torch
import torch.optim as optim
from torch.autograd import Variable
import copy

import numpy as np
from agent import *
from trainer import *
from control_environment import *

from dm_control import suite
from dm_control import viewer

import gym

import matplotlib.pyplot as plt

import threading
import pickle
import collections

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--load", type=int, help="Load arguments/model in given run (requires lib-type, env-type, task-type, agent-type, and ITERATION corresponding to which 'milestone' of previous run to load)", default = 0)
parser.add_argument("--load-iteration", type=str, help="Load arguments/model in given run (requires lib-type, env-type, task-type, agent-type, and ITERATION corresponding to which 'milestone' of previous run to load)", default = None)
parser.add_argument("--save-rate", type=int, help="Iterations between saves", default = 500)
#parser.add_argument("--save-location", type=str, help="Directory to save run details to.")

parser.add_argument("--lib-type", type=str, help="Library type (DM, GYM, Control)", default='dm')
parser.add_argument("--env-type", type=str, help="Environment type (walker, cartpole, humanoid, swimmer,...)", default = 'walker')
parser.add_argument("--task-type", type=str, help="Environment-specific task (walk, stand, balance, swingup,...)", default = 'stand')

parser.add_argument("--agent-type", type=str, help="Specify agent algorithm (policy [policy gradient], mbrl [model-based RL])", default = 'policy')
parser.add_argument("--max-iterations", type=int, help="Maximum number of iterations to train / evaluate agent for.", default=5000)
parser.add_argument("--random-baseline", type=int, default = 0, help="Sample actions uniformly if agent is eps-greedy and this argument != 0")
parser.add_argument("--eps-base", type=float, default=5e-2)
parser.add_argument("--eps-min", type=float, default=1e-2)
parser.add_argument("--eps-decay", type=float, default=1e-7)

parser.add_argument("--maxmin-normalization", type=int, default = 0, help="Boolean for maxmin-normalization. True if != 0")

parser.add_argument("--dataset-recent-prob", type=float, default=0.5, help="Sets ratio of selecting recent samples vs stored samples.")

##** MBRL-specific arguments
parser.add_argument("--ddp-mode", type=str, help="Determines 'DDP' / control method (ismc, ilqg)", default = 'ismc')
parser.add_argument("--local-linear-model", type=str, help="Specifies local-linear model type (if any)", default = 'pytorch')
parser.add_argument("--local-clusters", type=int, help="Number of clusters for local model", default = 100)
parser.add_argument("--local-neighbors", type=int, help="Number of neighbors to overfit on for local model", default = 0)
parser.add_argument("--global-model", type=str, help="Specifies glocal model, if any", default = None)

parser.add_argument("--smc-switching-function", type=str, default='arctan')

##** PG-specific arguments
parser.add_argument("--gamma", type=float, default = 0.98, help="Reward discount factor for reinforcement learning methods")
parser.add_argument("--trainer-type", type=str, default = 'AC', help="Specify the policy-gradient algorithm to train policy network on (AC, PPO)")
parser.add_argument("--value-coeff", type=float, default = 9e-2, help="Coeff to multiply value loss by before training policy/value network.")
parser.add_argument("--entropy-coeff", type=float, default = 5e-3, help="Coeff to multiply entropy loss by before training policy/value network.")

parser.add_argument("--train-autoencoder", type=int, default = 0, help="If != 0, uses autoencoder to transform inputs before inputting to policy network.")

#important that we retain the ability to have differnet Policy/Value nets
parser.add_argument('--value-mlp-activations')
parser.add_argument('--value-mlp-hdims')
parser.add_argument('--value-mlp-outdim')
parser.add_argument('--value-lr', type=float, default=1e-3, help="NN Learning Rate")
parser.add_argument('--value-momentum', type=float, default=1e-4)


##** NN-specific arguments (agent INDEPENDENT, IMPORTANT!)
parser.add_argument('--mlp-activations', nargs='+', type=str, default=[None])
parser.add_argument('--mlp-hdims', nargs='*', type=int, default=[])
parser.add_argument('--mlp-outdim', type=int, default = 500)
parser.add_argument('--lr', type=float, default=1e-3, help="NN Learning Rate")
parser.add_argument('--momentum', type=float, default=1e-4)
#parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))

parser.add_argument('--sigma-head', type=int, default = 1,
        help="Gauss Policy Sigma Head vs constant sigma")


class DMEnvMPCWalkerAgent(DMEnvMPCAgent):
    def transform_observation(self, obs) -> Variable:
        '''Converts ordered-dictionary of position and velocity into
        1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
        orientation = obs['orientations']
        #vel = obs['velocity']
        #height = np.asarray([obs['height'],])
        #state = np.concatenate((orientation, vel, height))
        return orientation
class PyTorchMPCCartpoleAgent(PyTorchMPCAgent):
    def __init__(self, *args, **kwargs):
        global env
        super().__init__(*args, **kwargs)
        self.env = env

    def transform_observation(self, obs) -> Variable:
        '''Converts ordered-dictionary of position and velocity into
        1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
        pos = obs['position']
        vel = obs['velocity']
        state = np.concatenate((pos, vel))
        return Variable(torch.tensor(state).float(), requires_grad = False)

    def reward(self, st, at, *args, **kwargs):
        upright = (self.env.physics.pole_angle_cosine() + 1) / 2
        return upright

def create_dm_cartpole_agent(agent_base, *args, **kwargs):
    '''Helper function to implement "transform_observation", and, in the case of MPC agents,
    a reward function, for the cartpole environment.'''
    class DMCartpoleAgent(agent_base):
        def transform_observation(self, obs) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            if type(obs) == type(collections.OrderedDict()):
                pos = obs['position']
                vel = obs['velocity']
                state = np.concatenate((pos, vel))
            #elif type(obs) == type(np.zeros(0)):
            else:
                state = obs
            return Variable(torch.tensor(state).float(), requires_grad = False)
    
        def reward(self, st, at, *args, **kwargs):
            upright = (self.env.physics.pole_angle_cosine() + 1) / 2
            return upright

        #def get_observation_size(self):
        #    return 
    return DMCartpoleAgent(*args, **kwargs) 


def create_dm_walker_agent(agent_base, *args, **kwargs):
    '''Helper function to implement "transform_observation", and, in the case of MPC agents,
    a reward function, for the walker2d environment.'''
    class DMWalkerAgent(agent_base):
        def transform_observation(self, obs) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            if type(obs) == type(collections.OrderedDict()):
                orientation = obs['orientations']
                vel = obs['velocity']
                height = np.asarray([obs['height'],])
                state = np.concatenate((orientation, vel, height))
            elif type(obs) == type(np.zeros(0)):
                state = obs
            #elif isinstance(obs, type(torch.tensor)):
            elif type(obs) == type(torch.tensor([0])):
                return obs
            return Variable(torch.tensor(state).float(), requires_grad = True)
    
        def reward(self, st, at, *args, **kwargs):
            global TASK_NAME #GROOOOOOSSSSSSSTHH
            if TASK_NAME in ['walk', 'run']:
                return env.physics.horizontal_velocity() #TODO: this is only valid for walk / run tasks
            else: 
                return 0.0
    
    return DMWalkerAgent(*args, **kwargs)


def create_dm_acrobot_agent(agent_base, *args, **kwargs):
    '''Helper function to implement "transform_observation", and, in the case of MPC agents,
    a reward function, for the acrobot environment.'''
    class DMAcrobotAgent(agent_base):
        def transform_observation(self, obs) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            if type(obs) == type(collections.OrderedDict()):
                orientation = obs['orientations']
                vel = obs['velocity']
                state = np.concatenate((orientation, vel))
            elif type(obs) == type(np.zeros(0)):
                state = obs
            #elif isinstance(obs, type(torch.tensor)):
            elif type(obs) == type(torch.tensor([0])):
                return obs
            return Variable(torch.tensor(state).float(), requires_grad = True)
    
        def reward(self, st, at, *args, **kwargs):
            return 0.0
    return DMAcrobotAgent(*args, **kwargs)

def create_gym_walker_agent(agent_base, *args, **kwargs):
    '''Helper function to implement "transform_observation", and, in the case of MPC agents,
    a reward function, for the walker2d environment.'''
    class GymWalkerAgent(agent_base):
        def transform_observation(self, obs) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            return Variable(torch.tensor(obs).float(), requires_grad = True)
    
        def reward(self, st, at, *args, **kwargs):
            raise Exception("Not implemented for Gym")
    
    return GymWalkerAgent(*args, **kwargs)

def create_dm_humanoid_agent(agent_base, *args, **kwargs):
    '''Helper function to implement "transform_observation", and, in the case of MPC agents,
    a reward function, for the walker2d environment.'''
    class DMHumanoidAgent(agent_base):
        def transform_observation(self, obs, extra_info = True) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            if type(obs) == type(collections.OrderedDict()):
                angles = obs['joint_angles']
                extremities = obs['extremities']
                velocity = obs['velocity']
                state = np.concatenate((angles, extremities, velocity))
                if extra_info:
                    com_velocity = obs['com_velocity']
                    #head_height = obs['head_height']
                    head_height = np.asarray([obs['head_height'],])
                    torso_vertical = obs['torso_vertical']
                    state = np.concatenate((state, com_velocity, head_height, torso_vertical))
                state = Variable(torch.tensor(state).float(), requires_grad = True)
            elif type(obs) in [type(np.zeros(0)),]: 
                state = Variable(torch.tensor(obs).float(), requires_grad = True)
            elif type(obs) in [type(torch.tensor([0])),]:
                state = obs
            #return Variable(torch.tensor(state).float(), requires_grad = True)
            return state
    
        def reward(self, st, at, *args, **kwargs):
            global TASK_NAME #GROOOOOOSSSSSSSTHH
            if TASK_NAME in ['walk', 'run']:
                return env.physics.horizontal_velocity() #TODO: this is only valid for walk / run tasks
            else: 
                return 0.0
    
    return DMHumanoidAgent(*args, **kwargs)

def create_numpy_agent(agent_base, *args, **kwargs):
    class NumpyAgent(agent_base):
        def transform_observation(self, obs) -> Variable:
            assert(type(obs) == type(np.zeros(0)))
            return Variable(torch.tensor(obs).float(), requires_grad = False)
    return NumpyAgent(*args, **kwargs)


def create_ddp_dm_humanoid_agent(agent_base, *args, **kwargs):
    '''Transform_observation transforms observation into Numpy array. '''
    class DDPHumanoidAgent(agent_base):
        def transform_observation(self, obs, extra_info = True) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            if type(obs) == type(collections.OrderedDict()):
                angles = obs['joint_angles']
                extremities = obs['extremities']
                velocity = obs['velocity']
                state = np.concatenate((angles, extremities, velocity))
                if extra_info:
                    com_velocity = obs['com_velocity']
                    #head_height = obs['head_height']
                    head_height = np.asarray([obs['head_height'],])
                    torso_vertical = obs['torso_vertical']
                    state = np.concatenate((state, com_velocity, head_height, torso_vertical))
                #print("STATE HEAD_HEIGHT: ", state[63])
                #print("STATE COM_VEL: ", state[60:63])
                #print("STATE TORSO_VERTICAL: ", state[64:])
                #print("Head: ", obs['head_height'])
                #print("com_vel: ", obs['com_velocity'])
                #print("torso_vel: ", obs['torso_vertical'])
            elif type(obs) in [type(np.zeros(0)), type(torch.tensor([0]))]:
                state = obs
            #return Variable(torch.tensor(state).float(), requires_grad = True)
            return state
    return DDPHumanoidAgent(*args, **kwargs)
def create_ddp_dm_walker_agent(agent_base, *args, **kwargs):
    '''Transform_observation transforms observation into Numpy array. '''
    class DDP_DMWalkerAgent(agent_base):
        def transform_observation(self, obs) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            if type(obs) == type(collections.OrderedDict()):
                orientation = obs['orientations']
                vel = obs['velocity']
                height = np.asarray([obs['height'],])
                state = np.concatenate((orientation, vel, height))
            elif type(obs) == type(np.zeros(0)):
                state = obs
            #return Variable(torch.tensor(state).float(), requires_grad = True)
            return state
    
        def reward(self, st, at, *args, **kwargs):
            global TASK_NAME #GROOOOOOSSSSSSSTHH
            if TASK_NAME in ['walk', 'run']:
                return env.physics.horizontal_velocity() #TODO: this is only valid for walk / run tasks
            else: 
                return 0.0
    
    return DDP_DMWalkerAgent(*args, **kwargs)

def create_ddp_dm_cartpole_agent(agent_base, *args, **kwargs):
    '''Transform_observation transforms observation into Numpy array. '''
    class DDP_DMCartpoleAgent(agent_base):
        def transform_observation(self, obs) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            if type(obs) == type(collections.OrderedDict()):
                pos = obs['position']
                vel = obs['velocity']
                state = np.concatenate((pos, vel))
            elif type(obs) == type(np.zeros(0)):
                state = obs
            #return Variable(torch.tensor(state).float(), requires_grad = False)
            return state
    
        def reward(self, st, at, *args, **kwargs):
            upright = (self.env.physics.pole_angle_cosine() + 1) / 2
            return upright

    return DDP_DMCartpoleAgent(*args, **kwargs) 

def create_ddp_dm_acrobot_agent(env, agent_base, *args, **kwargs):
    '''Helper function to implement "transform_observation", and, in the case of MPC agents,
    a reward function, for the acrobot environment.'''
    class DMAcrobotAgent(agent_base):
        def transform_observation(self, obs) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            if type(obs) == type(collections.OrderedDict()):
                orientation = obs['orientations']
                vel = obs['velocity']
                state = np.concatenate((orientation, vel))
                if True: #add extra "tip" information for DDP acrobot,
                        #or this is basically impossible
                    tip = self.env.physics.named.data.site_xpos['target']
                    state = np.concatenate((state, tip))
            elif type(obs) == type(np.zeros(0)):
                state = obs
            return state
    
        def reward(self, st, at, *args, **kwargs):
            return 0.0
    agent = DMAcrobotAgent(*args, **kwargs)
    agent.env = env #necessary for DDP methods
    return agent

#NOTE: I really need to clean this initialization code up :(
def create_ddp_agent(lib_type, env_type, env, agent_base = DDPMPCAgent, *args, **kwargs):
    agent = None
    if lib_type == 'dm':
        if env_type == 'humanoid':
            agent = create_ddp_dm_humanoid_agent(agent_base, *args, **kwargs)   
        elif env_type == 'walker':
            agent = create_ddp_dm_walker_agent(DDPMPCAgent, *args, **kwargs) 
        elif env_type == 'acrobot':
            agent = create_ddp_dm_acrobot_agent(env, DDPMPCAgent, *args, **kwargs) #requires access to additional state in order to function
        elif env_type == 'cartpole':
            agent = create_ddp_dm_cartpole_agent(DDPMPCAgent, *args, **kwargs) 
    else:
        return create_numpy_agent(agent_base, *args, **kwargs)
    return agent

def create_agent(agent_base, lib_type = 'dm', env_type = 'walker', *args, **kwargs):
    agent = None
    if lib_type == 'gym':
        if env_type == 'walker':
            return create_gym_walker_agent(agent_base, *args, **kwargs)
        elif env_type == 'cartpole':
            return create_gym_cartpole_agent(agent_base, *args, **kwargs)
    elif lib_type == 'dm':
        if env_type == 'walker':
            return create_dm_walker_agent(agent_base, *args, **kwargs)
        elif env_type == 'cartpole':
            return create_dm_cartpole_agent(agent_base, *args, **kwargs)
        elif env_type == 'acrobot':
            return create_dm_acrobot_agent(agent_base, *args, **kwargs)
        elif env_type == 'humanoid':
            return create_dm_humanoid_agent(agent_base, *args, **kwargs)
    else:
        return create_numpy_agent(agent_base, *args, **kwargs)
    return agent

def initialize_dm_environment(env_type, task_name, agent_type, *args, **kwargs):
    env = suite.load(domain_name = env_type, task_name = task_name)  
    tmp_env = suite.load(domain_name = env_type, task_name = task_name)  
    action_space = env.action_spec()
    obs_space = env.observation_spec()
    if env_type == 'walker':
        obs_size = obs_space['orientations'].shape[0] + obs_space['velocity'].shape[0] + 1 #+1 for height
    if env_type == 'acrobot':
        obs_size = obs_space['orientations'].shape[0] + obs_space['velocity'].shape[0]
        if agent_type in ['mbrl']: #add "tip" information. Just the tip
            obs_size += env.physics.named.data.site_xpos['tip'].shape[0]
    elif env_type == 'cartpole':
        obs_size = obs_space['position'].shape[0] + obs_space['velocity'].shape[0]
    elif env_type == 'humanoid':
        obs_size = obs_space['joint_angles'].shape[0] + obs_space['extremities'].shape[0]
        obs_size = obs_size + obs_space['velocity'].shape[0]
        obs_size = obs_size + obs_space['com_velocity'].shape[0]
        obs_size = obs_size + 1 #+1 for height
        obs_size = obs_size + obs_space['torso_vertical'].shape[0]

    action_size = action_space.shape[0] 
    #action_size = 2
    action_constraints = [action_space.minimum, action_space.maximum]
    obs_space = [1, obs_size]
    return env, tmp_env, obs_space, obs_size, action_size, action_constraints


def initialize_gym_environment(env_type, task_name, *args, **kwargs):
    if env_type == 'walker':
        env_string = 'Walker2d-v2'    
    elif env_type == 'cartpole':
        env_string = 'CartPole-v0'    
    env = gym.make(env_string)
    tmp_env = gym.make(env_string)
    #print("Action space: ", env.action_space)
    action_size = env.action_space.shape[0]
    action_constraints = [env.action_space.low, env.action_space.high]
    env.reset()
    obs, reward, done, info = env.step(1)
    obs_size = obs.size
    obs_space = [1, obs_size]
    return env, tmp_env, obs_space, obs_size, action_size, action_constraints

def initialize_control_environment(env_type, task_name, *args, **kwargs):
    env = retrieve_control_environment(env_type, **kwargs)
    tmp_env = retrieve_control_environment(env_type, **kwargs)
    action_size = env.get_action_size()
    action_constraints = env.get_action_constraints()
    obs_size = env.get_observation_size()
    obs_space = [1, obs_size]
    return env, tmp_env, obs_space, obs_size, action_size, action_constraints

def initialize_environment(lib_type, env_type, task_name, agent_type, *args, **kwargs):
    ''' 
        @args: 

        @returns:
            * env - Control environment
            * tmp_env - Copy of control environment, if applicable (else None)
            * obs_space - array representing shape of input space for RL environment
            * obs_size - scalar representing total size (vector elements) of observation
            * action_size - scalar representing # of actions for RL environment
            * action_constraints - (optional, else None) constraints for action outputs.
    '''
    if lib_type == 'dm':
        return initialize_dm_environment(env_type, task_name, agent_type, *args, **kwargs)
    elif lib_type == 'gym':
        return initialize_gym_environment(env_type, task_name, *args, **kwargs)
    elif lib_type == 'control':
        return initialize_control_environment(env_type, task_name, *args, **kwargs)



def create_pytorch_mpc_agent(env_type, lib_type, 
             
    env, obs_size, action_size, action_constraints,
    mlp_hdims, mlp_activations, 
    num_processes = 1,
    horizon = 40, 
    k_shoots = 20,
    discrete_actions = False, 
    has_value_function = False) -> (Agent, Trainer): 
    agent = create_agent(PyTorchMPCAgent, lib_type, env_type, 
                NUM_PROCESSES, HORIZON, K_SHOOTS,  #num_processes, #horizon, k_shoots
                device, 
                [1, obs_size], action_size, discrete_actions = DISCRETE_AGENT, 
                action_constraints = action_constraints, has_value_function = False)
    if env_type == 'walker':
        random_agent = RandomWalkerAgent([1, obs_size], action_size, 
                discrete_actions = DISCRETE_AGENT, 
                action_constraints = action_constraints, has_value_function = False)
    elif env_type == 'cartpole':
        random_agent = RandomCartpoleAgent([1, obs_size], action_size, 
                discrete_actions = DISCRETE_AGENT, 
                action_constraints = action_constraints, has_value_function = False)

    print("This reward is only valid for walk / run tasks!")
    agent.module = PyTorchMLP(device, obs_size + action_size, obs_size, 
            hdims = mlp_hdims, activations = mlp_activations, 
            initializer = mlp_initializer).to(device)    
    lr = 1.0e-3
    ADAM_BETAS = (0.9, 0.999)
    optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)
    trainer = PyTorchNeuralDynamicsMPCTrainer(agent, random_agent, optimizer,
            512, 1.0, 0.05, 0.5, 700, 700, #batch_size, starting rand, rand_decay, rand min, max steps, max iter        
            device, value_coeff, entropy_coeff,
            agent, env, optimizer, replay = replay_iterations, max_traj_len = max_traj_len, gamma = args.gamma,
            num_episodes = EPISODES_BEFORE_TRAINING) 
    return agent, trainer

def create_pytorch_policy_agent(env, obs_size, 
        action_size, action_constraints,
        mlp_hdims, mlp_activations, mlp_outdim, mlp_base,
        lr = 1e-3, adam_betas = (0.9, 0.999), momentum = 1e-3, 
        discrete_actions = False, 
        has_value_function = False) -> (Agent, Trainer): 
    mlp_indim = obs_size
    #mlp_outdim = mlp_indim * WIDENING_CONST #based on state size (approximation)
    print("MLP INDIM: %s HDIM: %s OUTDIM: %s " % (obs_size, mlp_hdims, mlp_outdim))
    print("MLP ACTIVATIONS: ", mlp_activations)
    agent = create_agent(PyTorchAgent, args.lib_type, args.env_type, device, [1, obs_size], 
                action_size, discrete_actions = DISCRETE_AGENT, 
                action_constraints = action_constraints, has_value_function = True,
                terminal_penalty = 0.0, 
                policy_entropy_history = True, energy_penalty_history = True)
    agent.module = EpsGreedyMLP(mlp_base, EPS, EPS_DECAY, EPS_MIN, action_constraints, 
            action_size, 
            seperate_value_module = None, seperate_value_module_input = False,
            value_head = True,
            action_bias = True, value_bias = True, sigma_head = sigma_head, 
            device = device, indim = obs_size, outdim = mlp_outdim, hdims = mlp_hdims,
            activations = mlp_activations, initializer = mlp_initializer).to(device)
    
    optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)
    #optimizer = optim.SGD(agent.module.parameters(), lr = lr, momentum = MOMENTUM)
    scheduler = None
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 300, gamma = 0.85)
    if args.trainer_type == 'AC':
        trainer = PyTorchACTrainer(args.value_coeff, args.entropy_coeff, 
                0, 0, #entropy_coeff, energy_loss
                device, optimizer, scheduler,   
                agent, env, 
                replay = replay_iterations, max_traj_len = max_traj_len, gamma = args.gamma,
                num_episodes = EPISODES_BEFORE_TRAINING) 
    elif args.trainer_type == 'PPO':
        trainer = PyTorchPPOTrainer(args.value_coeff, args.entropy_coeff, 
                0, 0, #entropy_coeff, energy_loss
                device, optimizer, scheduler, 
                agent, env, 
                replay = replay_iterations, max_traj_len = max_traj_len, gamma = args.gamma,
                num_episodes = EPISODES_BEFORE_TRAINING) 
    return agent, trainer

def create_pytorch_agnostic_mbrl(cost, 
        ddp, reuse_shoots, 
        eps_base, eps_min, eps_decay,
        dataset, 
        #system_model, system_model_args, system_model_kwargs, 
        dt, horizon, k_shoots,
        batch_size, collect_forward_loss,
        replays, criterion,
        lib_type, env_type, 
        env, obs_size, action_size, action_constraints,
        mlp_hdims, mlp_activations, 
        lr = 1e-3, adam_betas = (0.9, 0.999), momentum = 1e-3, 
        discrete_actions = False, 
        has_value_function = False, ) -> (Agent, Trainer): 
    #NOTE: I'm going insane with this pseudoOOP it's compulsive right now D:
    #NOTE: I really need to clean this initialization code up :(
    agent = create_ddp_agent(lib_type, env_type, env, DDPMPCAgent, ddp, 
                    reuse_shoots, 
                    eps_base, eps_min, eps_decay, 
                    horizon, k_shoots, 
                    [1, obs_size], 
                    action_size, discrete_actions = DISCRETE_AGENT, 
                    action_constraints = action_constraints, 
                    has_value_function = True,
                    terminal_penalty = 0.0, 
                    policy_entropy_history = True, 
                    energy_penalty_history = True)
    if hasattr(agent.mpc_ddp.model, 'module'):
        criterion = torch.nn.MSELoss() 
        module = agent.mpc_ddp.model.module
        optimizer = optim.Adam(module.parameters(), lr = lr, betas = ADAM_BETAS)
        #optimizer = optim.SGD(module.parameters(), lr = lr, momentum = MOMENTUM)
        scheduler = None
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 300, gamma = 0.85)

        trainer = PyTorchDynamicsTrainer(system_model, 
                dataset, 
                criterion,
                batch_size, collect_forward_loss, device,
                optimizer, scheduler = None, agent = agent, env = env,
                replay = replays, max_traj_len = None, gamma=0.98, 
                num_episodes = 1) 
    else: #local models?
        #if isinstance(system_model, SKLearnLinearClusterLocalModel):
        trainer = LocalDynamicsTrainer(system_model, 
                dataset, 
                agent = agent, env = env,
                replay = replays, max_traj_len = None, gamma=0.98, 
                num_episodes = 1) 
        #elif isinstance(system_model, PyTorchLinearClusterLocalModel):
        #in the case of local-linear models relying on
        #non-PyTorch methods. This code is getting unmanageable. 
    return agent, trainer
        


def launch_viewer(env, agent):
    try:
        viewer.launch(env, policy = agent)
    except:
        pass


def launch_best_agent(env, agent):
    try:
        pass
    except: 
        pass

def console(env, agent, lock, lib_type = 'dm', env_type = 'walker', encoder = None):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    #agent.device = device
    #agent.module.to(device)
    while True: 
        input()
        with lock:
            cmd = input('>')
            if cmd.lower() == 'v': #view with thread locked?
                if lib_type == 'control':
                    print("INVALID OPTION!")
                    break
                print("VIEWING!")
                ## We create a clone of the agent (to preserve the training agent's history) 
                encoder_clone = None
                encode_inputs = encoder is not None
                if encode_inputs:
                    encoder_clone = copy.deepcopy(encoder).to(device)
                    clone = create_agent(PyTorchAgent, args.lib_type, env_type, device, [1, obs_size], 
                        action_size, discrete_actions = DISCRETE_AGENT, 
                        action_constraints = action_constraints, has_value_function = True, 
                        encode_inputs = encode_inputs, encoder = encoder_clone)
                elif (issubclass(type(agent), DDPMPCAgent)):
                    clone = copy.deepcopy(agent)
                    #clone.cost = copy.deepcopy(agent.cost)
                    print("Clone: ", clone)
                    if hasattr(clone.mpc_ddp.model, 'module'):
                        clone.mpc_ddp.model.module = copy.deepcopy(agent.mpc_ddp.model.module).to(device)
                        clone.mpc_ddp.model.module.device = device
                    launch_viewer(env, clone)    
                else:
                    clone = create_agent(PyTorchAgent, args.lib_type, env_type, device, [1, obs_size], 
                        action_size, discrete_actions = DISCRETE_AGENT, 
                        action_constraints = action_constraints, has_value_function = True, 
                        encode_inputs = None, encoder = None)
                    module = copy.deepcopy(agent.module)
                    #agent.module = None
                    #clone = copy.deepcopy(agent)
                    clone.module = module
                    clone.module.to(device)
                    launch_viewer(env, clone)    
                print("RESUMING!")

   
def get_max_min_path(LIB_TYPE = 'dm', ENV_TYPE = 'walker', 
        norm_dir = './norm'):
    if not os.path.exists(norm_dir):
        os.makedirs(norm_dir)
    return os.path.join(norm_dir, LIB_TYPE + '_' + ENV_TYPE)

def get_max_min_filename(LIB_TYPE = 'dm', ENV_TYPE = 'walker', 
        norm_dir = './norm'):
    return get_max_min_path(LIB_TYPE, ENV_TYPE, norm_dir) + '.pkl'

def retrieve_max_min(mx_shape, mn_shape, LIB_TYPE = 'dm', 
        ENV_TYPE = 'walker',
        norm_dir = 'norm') -> (np.ndarray, np.ndarray):
    filename = get_max_min_filename(LIB_TYPE, ENV_TYPE, norm_dir)
    if not os.path.exists(filename):
        mx = np.ones(mx_shape) * (-1) * float('inf') #s.t. anything is larger
        mn = np.ones(mx_shape) * float('inf') #such that ANYTHING is smaller
    else:
        with open(filename, 'rb') as f:
            mx_mn = pickle.load(f)
        mx = mx_mn['mx']
        mn = mx_mn['mn']
    return mx, mn

def store_max_min(mx, mn, LIB_TYPE = 'dm', ENV_TYPE = 'walker',
        norm_dir = 'norm'):
    filename = get_max_min_filename(LIB_TYPE, ENV_TYPE, norm_dir)
    curr_mx, curr_mn = retrieve_max_min(mx.shape, mn.shape, LIB_TYPE, ENV_TYPE,
            norm_dir)
    mx = np.maximum(curr_mx, mx) #in order to prevent parallel processes overwriting
    mn = np.minimum(curr_mn, mn)
    mx_mn = {'mx':mx, 'mn':mn}
    with open(filename, 'wb') as f:
        pickle.dump(mx_mn, f, pickle.HIGHEST_PROTOCOL)


def normalize_max_min(observation : np.ndarray, 
        mx : np.ndarray, mn : np.ndarray):
    if mn[0] >= float('inf'):
        return observation
    return (observation - mn) / (mx - mn)

def update_max_min(observation : np.ndarray, 
        mx : np.ndarray, mn : np.ndarray):
    mx = np.maximum(mx, observation)
    mn = np.minimum(mn, observation)
    return mx, mn

def save_pytorch_module(module, optimizer, filepath):
    '''https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/23'''
    state_dict = module.module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict,
        filepath)
    #torch.save({
    #    'state_dict':state_dict,
    #    'optimizer':optim},
    #    filepath)


LIB = 'pytorch'
MAX_TIMESTEPS = 100000

WIDENING_CONST = 20 #indim * WIDENING_CONST = hidden layer size
mlp_initializer = None
DISCRETE_AGENT = False




DISPLAY_HISTORY = True
DISPLAY_AV_LOSS = True

FULL_EPISODE = True
MAXIMUM_TRAJECTORY_LENGTH = MAX_TIMESTEPS
SMALL_TRAJECTORY_LENGTH = 200

if FULL_EPISODE:
    max_traj_len = MAXIMUM_TRAJECTORY_LENGTH
    EPISODES_BEFORE_TRAINING = 1 #so we benefit from reusing sampled trajectories with PPO / TRPO
    replay_iterations = EPISODES_BEFORE_TRAINING #approximate based on episode length 
else:
    max_traj_len = SMALL_TRAJECTORY_LENGTH
    EPISODES_BEFORE_TRAINING = 5
    replay_iterations = 30 #approximate based on episode length 






ADAM_BETAS = (0.9, 0.999)
PRETRAINED = False #this has lost its meaning, remove?!
RUN_ANYWAYS = True


MA_LEN = -1
MA_LEN = 15

if __name__ == '__main__':
    args = parser.parse_args() 
    dirname = os.getcwd()
    taskdir = os.path.join(dirname, 'history/%s_%s'%(args.env_type, args.task_type))
    if not os.path.exists(taskdir): 
        os.mkdir(taskdir)
    if args.load != 0:  #load previous run HERE
        run = tmp_args.load #corresponds to "[agent_type]_run" dir
        loaddir = os.path.join(taskdir, '%s_%s'%(args.agent_type, run))
        argspath = os.path.join(rundir, 'args')
        if os.path.exists(loaddir):
            rundir = loaddir
            num_runs = run #so we don't attempt to create rundir again
            with open(argspath, 'rb') as f:
                args = pickle.load(f)
        else:
            raise Exception("Directory/Run to load does not exist!")
    else:
        runs = [dI for dI in os.listdir(taskdir) if os.path.isdir(os.path.join(taskdir,dI)) and args.agent_type in dI]
        num_runs = len(runs)
        rundir = os.path.join(taskdir, '%s_%s'%(args.agent_type, num_runs))
    print("Rundir: ", rundir)
    print("TASKDIR: ", taskdir)
    print("DIRNAME: ", dirname)


    print("Lib: ", args.lib_type)
    print("Env: ", args.env_type)
    print("Task: ", args.task_type)
    if args.random_baseline != 0:
        EPS = 1e0
        EPS_MIN = 1e0
        EPS_DECAY = 1e-6
    else:
        EPS = args.eps_base
        EPS_MIN = args.eps_min
        EPS_DECAY = args.eps_decay
    print("Eps: %s Eps min: %s Eps decay: %s" % (EPS, EPS_MIN, EPS_DECAY))
    kwargs = {}
    if args.lib_type == 'control':
        if args.env_type == 'rossler':
            ENV_KWARGS = {'noisy_init' : True, 'ts' : 0.001, 'interval' : 10}
            args.task_type = 'point'
        elif args.env_type == 'cartpole':

            ENV_KWARGS = {'noisy_init' : True, 'ts' : 0.001, 'interval' : 10}
            args.task_type = 'point'
        elif args.env_type == 'inverted':
            ENV_KWARGS = {'noisy_init' : True, 'friction' : 0.001, 'ts' : 0.001, 'interval' : 5, 
                    'target':np.array([0, 0])}
            args.task_type = 'point'
        kwargs = ENV_KWARGS
    env, tmp_env, obs_space, obs_size, action_size, action_constraints = initialize_environment(args.lib_type, args.env_type, args.task_type, args.agent_type, **kwargs)
    
    sigma_head = True if args.sigma_head else False
    print("Sigma Head: ", sigma_head)

    mlp_hdims = args.mlp_hdims
    mlp_activations = args.mlp_activations
    mlp_outdim = args.mlp_outdim



    MA = 0
    averages = []

    i = 0
    step = 0
    timestep = None
    trainer = None 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.load != 0:
        print("Loading agent from: %s" % (args.load))
    else:
        print("Creating agent!")
        #device = torch.device("cpu") 
         
        if args.agent_type == 'mpc': 
            agent, trainer = create_pytorch_mpc_agent(env_type, lib_type, 
                    env, obs_size, action_size, action_constraints,
                    mlp_hdims, mlp_activations, 
                    num_processes = 1,
                    horizon = 40, 
                    k_shoots = 20,
                    discrete_actions = DISCRETE_AGENT, 
                    has_value_function = False) 
            trainer.step() #RUN PRETRAINING STEP
            PRETRAINED = True

        #pertinent from pg 9 of 1708.02596: "...ADAM optimizer lr = 0.001, batch size 512.
        #prior to training both the inputs and outputs in the dataset were preprocessed to have
        #zero mean, std dev = 1"
        #TODO TODO:^this could include a value function too?! SEPERATE VALUE FUNCTION
        # to benefit from MPC AND model-free agent sampling, since value functions are ubiquitous

        elif args.agent_type == 'policy':
            #raise Exception("Static vs Variable variance, anneal variance to avoid agent converging to whacky actions. Penalize magnitude of action?")
            #mlp_activations = [None, 'relu', None] #+1 for outdim activation, remember extra action/value modules
            #mlp_hdims = [obs_size * WIDENING_CONST, obs_size * WIDENING_CONST] 
            if DISCRETE_AGENT:
                mlp_base = PyTorchDiscreteACMLP
            else:
                mlp_base = PyTorchContinuousGaussACMLP
            agent, trainer = create_pytorch_policy_agent(env, obs_size, 
                action_size, action_constraints,
                mlp_hdims, mlp_activations, mlp_outdim, mlp_base,
                lr = 1e-3, adam_betas = (0.9, 0.999), momentum = 1e-3, 
                discrete_actions = False, 
                has_value_function = False) 

            print("AGENT MODULE (pre-autoencoder?)", agent.module)
            ## Autoencoder addition
            autoencoder_base = PyTorchLinearAutoencoder
            autoencoder = None
            autoencoder_trainer = None
            #AE_BATCHES = None
            TRAIN_FORWARD = True
            TRAIN_ACTION = True

            mlp_outdim = None
            DEPTH = 2
            UNIFORM_LAYERS = True
            REDUCTION_FACTOR = 0.7
            COUPLED_SA = False #have S/A feed into same encoded space or not
            PREAGENT = True
            PREAGENT_VALUE_FUNC = True #(True, None) or False
            PREAGENT_VALUE_INPUT = not (PREAGENT_VALUE_FUNC in (None, False))
            FORWARD_DYNAMICS = False #False reflects potential for linear transformation
            LINEAR_FORWARD = False #imposes linear function on forward dynamics
            AE_ACTIVATIONS = ['relu']
            ENCODED_ACTIVATIONS = []

            ae_lr = 0.5e-3
            ae_ADAM_BETAS = (0.9, 0.999)
            ae_MOMENTUM = 1e-3
            ae_MOMENTUM = 0

            AE_BATCHES = 64
            AE_REPLAYS = 10
            #DATASET_RECENT_PROB = 0.5
            if action_size > 1:
                forward_indim = math.floor(obs_size * REDUCTION_FACTOR**DEPTH) + math.floor(action_size * REDUCTION_FACTOR**DEPTH)
            else:
                forward_indim = math.floor(obs_size * REDUCTION_FACTOR**DEPTH) + 1
            #TODO: ^ this works...in both cases (coupled_sa or not)
            forward_outdim = obs_size * (REDUCTION_FACTOR**DEPTH)
            forward_hdims = [obs_size * WIDENING_CONST] 
            forward_activations = ['relu', None] #+1 for outdim activation, remember extra action/value modules
            
            if LINEAR_FORWARD:
                raise Exception("Reduce to the sum of TWO matrices representing \
                        Ax + Bu where x=state and u=action")
                forward_mlp = PyTorchMLP(device, forward_indim, forward_outdim, 
                        hdims = forward_hdims, activations = forward_activations, 
                        initializer = mlp_initializer).to(device)
            else:
                forward_mlp = PyTorchMLP(device, forward_indim, math.floor(forward_outdim), 
                        hdims = forward_hdims, activations = forward_activations, 
                        initializer = mlp_initializer).to(device)   

            if args.train_autoencoder != 0:
                raise Exception("Add autoencoder-specific args, such as coupled_sa, reduction_factor, widening_const, etc")

                AUTOENCODER_DATASET = Dataset(aggregate_examples = False, shuffle = True)
                #AUTOENCODER_DATASET = DAgger(recent_prob = DATASET_RECENT_PROB, aggregate_examples = False, shuffle = True)
                autoencoder = LinearSAAutoencoder(autoencoder_base, 
                        obs_size, action_size, forward_mlp, COUPLED_SA, FORWARD_DYNAMICS,
                        device, DEPTH, AE_ACTIVATIONS, ENCODED_ACTIVATIONS, REDUCTION_FACTOR,
                        UNIFORM_LAYERS)
                print("AUTOENCODER: ", autoencoder)
                ae_optimizer = optim.Adam(autoencoder.parameters(), lr = ae_lr, betas = ae_ADAM_BETAS)
                #ae_optimizer = optim.SGD(autoencoder.parameters(), lr = ae_lr, momentum = ae_MOMENTUM)
                ae_scheduler = None
                #ae_scheduler = torch.optim.lr_scheduler.StepLR(ae_optimizer, step_size = 300, gamma = 0.85)
                autoencoder_trainer = PyTorchSAAutoencoderTrainer(
                        autoencoder, AUTOENCODER_DATASET, AE_BATCHES, TRAIN_FORWARD, TRAIN_ACTION,
                        device, ae_optimizer, ae_scheduler, 
                        agent, env, 
                        replay = AE_REPLAYS, max_traj_len = max_traj_len, gamma = args.gamma,
                        num_episodes = EPISODES_BEFORE_TRAINING)
                if PREAGENT and not COUPLED_SA:
                    print("Feeding ENCODED state information into PolicyGradient agent!")
                    value_module = None
                    if PREAGENT_VALUE_FUNC == True:
                        value_module = torch.nn.Sequential(agent.module.value_mlp, #copy for safekeeping :)
                                agent.module.value_module)
                        #value_module = agent.module.value_mlp
                    mlp_indim = math.floor(obs_size * REDUCTION_FACTOR**DEPTH)  
                    agent.module = EpsGreedyMLP(mlp_base, EPS, EPS_DECAY, EPS_MIN, [], 
                            action_size, 
                            seperate_value_module = value_module, 
                            seperate_value_module_input = PREAGENT_VALUE_INPUT,
                            value_head = not PREAGENT_VALUE_FUNC, 
                            action_bias = True, value_bias = True, sigma_head = sigma_head, 
                            device = device, indim = mlp_indim, outdim = action_size, hdims = mlp_hdims,
                            activations = mlp_activations, initializer = mlp_initializer,
                            ).to(device)
                    agent.encode_inputs = True
                    agent.encoder = autoencoder
                    print("Post-encoder Module: ", agent.module)
                    optimizer = optim.Adam(agent.module.parameters(), lr = args.lr, betas = ADAM_BETAS)
                    #optimizer = optim.SGD(agent.module.parameters(), lr = lr, momentum = MOMENTUM)
                    scheduler = None
                    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.85)
                    trainer.opt = optimizer
                    trainer.scheduler = scheduler
                else:
                    agent.encode_inputs = False
        
        elif args.agent_type == 'mbrl':
            BATCH_SIZE = 64 
            COLLECT_FORWARD_LOSS = False
            DT = 1e-2 
            if args.lib_type == 'dm':
                DT = env.control_timestep()
            
            print("Local Linear Model: ", args.local_linear_model)
            print("Global Model: ", args.global_model)
            dataset = DAgger(recent_prob = args.dataset_recent_prob, aggregate_examples = False, shuffle = True)
            if args.local_linear_model is not None:
                dataset = Dataset(aggregate_examples = False, shuffle = True)
            if args.global_model == 'linear':
                pytorch_class = PyTorchLinearSystemDynamicsLinearModule
                pytorch_model = PyTorchLinearSystemModel 
            else:
                pytorch_class = PyTorchForwardDynamicsLinearModule
                pytorch_model = PyTorchForwardDynamicsModel 

            
            ## SMC-SPECIFIC ARGS
            SURFACE_BASE = None
            #surface = np.concatenate([np.eye(obs_size) for i in range(action_size)])
            #surface = np.eye(obs_size, M=action_size)
            surface = np.ones([obs_size, action_size]) * 0.5

            ## DDP-SPECIFIC ARGS 
            LQG_FULL_ITERATIONS = True
            LAMB_FACTOR = 10
            LAMB_MAX = 1000
            DDP_INIT = 0.0
            HORIZON = 0.5e-1
            DDP_MAX_ITERATIONS = 2
            K_SHOOTS = 1
            UPDATE_DDP_MODEL = True
            ILQG_SMC = False
            REUSE_SHOOTS = False if args.ddp_mode == 'ilqg' else False

            if args.local_linear_model == 'sklearn': 
                system_model = SKLearnLinearClusterLocalModel(
                        (obs_size, obs_size), (obs_size, action_size),
                        n_clusters = args.local_clusters,
                        compute_labels = True, 
                        dt = DT)
            elif args.local_linear_model == 'pytorch': 
                system_model = PyTorchLinearClusterLocalModel(device,
                        (obs_size, obs_size), (obs_size, action_size),
                        n_clusters = args.local_clusters, 
                        neighbors = args.local_neighbors,
                        compute_labels = True, 
                        dt = DT)
            else:
                if pytorch_class == PyTorchForwardDynamicsLinearModule:
                    pytorch_module = pytorch_class((obs_size,  obs_size), (obs_size, action_size), device = device, indim = obs_size, outdim = mlp_outdim, hdims = mlp_hdims,
                        activations = mlp_activations, initializer = mlp_initializer).to(device)
                elif pytorch_class == PyTorchLinearSystemDynamicsLinearModule:
                    pytorch_module = pytorch_class((obs_size,  obs_size), (obs_size, action_size), device = device, indim = obs_size, outdim = mlp_outdim, hdims = mlp_hdims,
                        activations = mlp_activations, initializer = mlp_initializer).to(device)
                system_model = pytorch_model(pytorch_module, DT) 

            #TODO TODO: Neural Network generation of quadratic cost
            #function, using RL
            cost = None
            if args.lib_type == 'dm':
                target_inds = []
                if args.env_type == 'humanoid':
                    target = np.zeros(obs_size)
                    head_height_ind = 60 + 3 #61 from angle+extrem+vel
                    #torso_vertx_ind = 0
                    #torso_verty_ind = 0
                    torso_vertz_ind = 60 + 4
                    com_velx_ind = 60
                    #com_vely_ind = 0
                    #com_velz_ind = 0

                    target_head_height = 1.4
                    target_vertz_val = 0.9 #MINIMAL upright value
                    target_com_velx_val = 0#TRY to have it run in straight line
                    target[head_height_ind] = target_head_height
                    target[torso_vertz_ind] = target_vertz_val #TODO: confirm this
                    target[com_velx_ind] = target_com_velx_val
                    target_inds = [head_height_ind, torso_vertz_ind, com_velx_ind]
                    
                    if args.task_type == 'walk':
                        target_com_velx_val = 1
                    elif args.task_type == 'run':
                        target_com_velx_val = 10
                    Q = np.eye(obs_size) * 1e8
                    Qf = Q
                    R = np.eye(action_size) * 1e3
                if args.env_type == 'walker':
                    #cost is manually set as walker height and
                    #center-of-mass horizontal velocity, conditioned
                    #on task type
                    print("ENV TYPE IS WALKER")
                    target = np.zeros(obs_size)
                    upright_ind = 0 #1st "orientation" corresponds
                    height_ind = 14 #height field corresponds
                    if args.task_type == 'stand':
                        target[height_ind] = 2.0
                        target[upright_ind] = 1.0 #TODO: confirm this
                        target_inds = [height_ind, upright_ind]
                    else:
                        raise Exception("Extract COM for this env.")
                        #com_vel_ind = 30 #NO obs directly corresponds to com velocity
                        #target_inds = [height_ind, upright_ind, com_vel_ind]
                    Q = np.eye(obs_size) * 1e8
                    Qf = Q
                    R = np.eye(action_size) * 1e3
                if args.env_type == 'cartpole':
                    #cost is manually set as deviation from pole upright
                    #position.
                    print("ENV TYPE IS CARTPOLE")
                    target = np.zeros(obs_size)
                    theta_ind = 1 #based on pole-angle cosine, rep theta
                    target_theta = -0.0 #target pole position
                    target[theta_ind] = target_theta
                    target_inds = [theta_ind]
                    Q = np.eye(obs_size) * 1e8
                    Qf = Q
                    R = np.eye(action_size) * 1e3
                if args.env_type == 'acrobot':
                    #cost is manually set as deviation from upright position
                    print("ENV TYPE IS ACROBOT")
                    target = np.zeros(obs_size)
                    tip_target_start = 6
                    tip_target = env.physics.named.data.site_xpos['target']
                    target[tip_target_start:] = tip_target
                    target_inds = [tip_target_start + i for i in range(tip_target.shape[0])]
                    #target_inds = [tip_target_start + (tip_target.shape[0]) - 1]
                    Q = np.eye(obs_size) * 1e8
                    Qf = Q
                    R = np.eye(action_size) * 1e3
            elif args.lib_type == 'control':
                if args.env_type == 'inverted':
                    #cost is manually set as deviation from pole upright
                    #position.
                    print("ENV TYPE IS INVERTED PENDULUM (control environment)")
                    target = np.zeros(obs_size)
                    theta_ind = 0 #based on pole-angle cosine, rep theta
                    target_theta = 0.0 #target pole position
                    target[theta_ind] = target_theta
                    target_inds = [theta_ind]
                    Q = np.eye(obs_size) * 1e2
                    Qf = Q * 1e2
                    R = np.eye(action_size) * 1e1
            print("TARGET: ", target)
            if len(target.shape) < 2:
                target = target[..., np.newaxis]
            priority_cost = True
            if priority_cost:
                for i in range(obs_size):
                    if i not in target_inds:
                        Q[i][i] = np.sqrt(Q[i][i])
            else:
                for i in range(obs_size):
                    if i not in target_inds:
                        Q[i][i] = 0
            class DiffFunc:
                def __init__(self, target_inds):
                    self.inds = target_inds
                def __call__(self, t, x):
                    x_ = np.zeros(x.shape)
                    for i in self.inds:
                        x_[i] = x[i] - t[i]
                    return x_
            #diff_func = DiffFunc(target_inds)
            diff_func = lambda t,x:x - t 
            print("Q: ", Q)
            cost = LQC(Q, R, Qf, target = target, 
                    diff_func = diff_func)
            if args.lib_type == 'control':
                env.cost = cost #to get meaningful reward data
            ddp = None
            if args.ddp_mode == 'ilqg':
                ddp = ILQG(LQG_FULL_ITERATIONS,
                        LAMB_FACTOR, LAMB_MAX, DDP_INIT,
                        obs_space, obs_size,
                        [1, action_size], action_size,
                        system_model, cost,
                        None, 
                        action_constraints, 
                        horizon = int(HORIZON/DT), dt = DT, 
                        max_iterations = DDP_MAX_ITERATIONS,
                        eps = 1e-2,
                        update_model = UPDATE_DDP_MODEL
                        )
                if ILQG_SMC == True:
                    print("Surface Function: ", surface)
                    smc = SMC(surface,
                            target, diff_func,
                            args.smc_switching_function,
                            obs_space, obs_size,
                            [1, action_size], action_size,
                            system_model, cost,
                            None, 
                            action_constraints, 
                            horizon = int(HORIZON/DT), dt = DT, 
                            max_iterations = DDP_MAX_ITERATIONS,
                            eps = 1e-2,
                            update_model = UPDATE_DDP_MODEL
                            )
                    ddp.set_smc(smc)

            elif args.ddp_mode == 'ismc':
                #The idea is to make surface non-uniform (singular) 
                #across control variables, but also to construct
                #a generally good surface (target * 1e1 to focus emphasis
                #on target variables)
                target_vars = np.zeros((obs_size, 1)) #for guiding SMC
                for i in target_inds:
                    target_vars[i] = 1
                surface = surface + target_vars * 5e0 + np.eye(obs_size, M = action_size) * 1e-2
                #surface = surface + target * 1e1
                print("Obs Size: ", obs_size)
                print("Action Size: ", action_size)
                print("Surface Function: ", surface)
                ddp = SMC(surface,
                        target, diff_func,
                        args.smc_switching_function,
                        obs_space, obs_size,
                        [1, action_size], action_size,
                        system_model, cost,
                        None, 
                        action_constraints, 
                        horizon = int(HORIZON/DT), dt = DT, 
                        max_iterations = DDP_MAX_ITERATIONS,
                        eps = 1e-2,
                        update_model = UPDATE_DDP_MODEL
                        )
            
            
            agent, trainer =  create_pytorch_agnostic_mbrl(cost, 
                ddp, REUSE_SHOOTS,
                EPS, EPS_MIN, EPS_DECAY, 
                dataset,
                DT, HORIZON, K_SHOOTS,
                BATCH_SIZE, COLLECT_FORWARD_LOSS, 
                replay_iterations, None,
                args.lib_type, args.env_type,
                env, obs_size, action_size, action_constraints,
                mlp_hdims, mlp_activations, 
                lr = args.lr, adam_betas = ADAM_BETAS, momentum = args.momentum, 
                discrete_actions = False, 
                has_value_function = False) 
    


    if args.maxmin_normalization: 
        print(get_max_min_path(args.lib_type, args.env_type))
        mx, mn = retrieve_max_min([obs_size],[obs_size],args.lib_type, args.env_type, norm_dir = 'norm')
        print("CURRENT MX: %s \n MN: %s \n" % (mx, mn))
        new_mx, new_mn = mx, mn #copy them for updating
    input()
    
    ## RUN AGENT / TRAINING
    if (not PRETRAINED) or (RUN_ANYWAYS):
        ## set up listener for user input
        lock = threading.Lock()
        console_args = (tmp_env, agent, lock, args.lib_type, args.env_type, agent.encoder)
        console_thread = threading.Thread(target = console, args = console_args)
        console_thread.daemon = True
        console_thread.start()
        ##
        i = 0
        while i < args.max_iterations: #and error > threshold
            print("ITERATION: ", i)
            # Exploration / evaluation step
            for episode in range(EPISODES_BEFORE_TRAINING):
                if args.lib_type == 'dm':
                    timestep = env.reset()        
                    while not timestep.last() and step < MAX_TIMESTEPS:
                        reward = timestep.reward
                        if reward is None:
                            reward = 0.0
                        #print("TIMESTEP %s: %s" % (step, timestep))
                        observation = timestep.observation
                        observation = agent.transform_observation(observation)
                        #print("OBS: ", observation)
                        if args.maxmin_normalization:
                            if type(observation) == torch.Tensor:
                                observation = normalize_max_min(observation.detach().cpu().numpy(), mx, mn)
                            else:
                                observation = normalize_max_min(observation, mx, mn)
                            #print("Observation:", observation)
                        action = agent.step(observation)
                        if type(action) is torch.Tensor:
                            action = action.cpu().numpy()
                        #print("Observation: ", observation)
                        #action = agent(timestep)
                        print("Action: ", action)
                        agent.store_reward(reward)
                        timestep = env.step(action)
                        #print("Reward: %s" % (timestep.reward))
                elif args.lib_type == 'gym':
                    env.reset()
                    observation, reward, done, info = env.step(env.action_space.sample())
                    while not done and step < MAX_TIMESTEPS:
                        if args.maxmin_normalization:
                            observation = agent.transform_observation(observation)
                            observation = normalize_max_min(observation.detach().cpu().numpy(), mx, mn)
                        action = agent.step(observation).cpu().numpy()
                        env.render()
                        observation, reward, done, info = env.step(action)
                        agent.store_reward(reward)
                elif args.lib_type == 'control':
                    env.reset()
                    while not env.episode_is_done() and step < MAX_TIMESTEPS:
                        obs = env.get_state()
                        action = agent.step(obs)
                        if isinstance(action, torch.Tensor):
                            action = agent.step(obs).cpu().numpy()
                        env.step(action)
                        reward = env.get_reward()
                        agent.store_reward(reward)

                if args.maxmin_normalization:
                    new_mx, new_mn = update_max_min(observation, new_mx, new_mn)
                agent.terminate_episode() #oops forgot this lmao
            # Update step
            if not PRETRAINED:
                trainer.step()
                print("Agent Net Loss: ", agent.net_loss_history[-1])
            if args.train_autoencoder != 0:
                autoencoder_trainer.step()
                autoencoder_trainer.plot_loss_histories()
            if i % 50 == 0 and i > 0: #periodic tasks
                print("Performing Periodic Tasks")
                if args.maxmin_normalization: 
                    print("(stored) max: %s\n min: %s\n"%(new_mx, new_mn))
                    store_max_min(new_mx, new_mn, args.lib_type, args.env_type, norm_dir = 'norm') 
            
            if args.lib_type == 'control':
                if 'mb' in args.agent_type: #look for model-based. Disgusting
                    env.generate_vector_field_plot()
                    env.generate_vector_field_plot(agent.mpc_ddp.model)

            if i % args.save_rate == 0 and i > 0: #perform milestone save
                #create directory(s) (task / agent)
                runs = [dI for dI in os.listdir(taskdir) if os.path.isdir(os.path.join(taskdir,dI)) and args.agent_type in dI]
                cur_runs = len(runs)

                modelpath = os.path.join(rundir, 'model_%s'%(i))
                historypath = os.path.join(rundir, 'history_%s'%(i))
                argspath = os.path.join(rundir, 'args')
                if num_runs == cur_runs: #make rundir
                    if not os.path.exists(rundir): 
                        os.mkdir(rundir)
                    else:
                        raise Exception("Rundir already exists?!")
                with open(modelpath, 'wb') as f:
                    if args.agent_type == 'mbrl':
                        #pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)
                        pickle.dump(agent.mpc_ddp.model.cluster, f, pickle.HIGHEST_PROTOCOL) #WEW
                    elif args.agent_type  == 'policy':
                        save_pytorch_module(agent, trainer.opt, f)
                with open(historypath, 'wb') as f:
                    history_dict = {'reward':agent.net_reward_history, 'averages':averages}
                    pickle.dump(history_dict, f, pickle.HIGHEST_PROTOCOL)
                if not os.path.exists(argspath):
                    with open(argspath, 'wb') as f:
                        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

            agent.reset_histories()
            if args.agent_type == 'policy' and not PRETRAINED:
                print("Agent Net Action loss: ", agent.value_loss_history[i])
                print("Agent Net Value loss: ", agent.action_loss_history[i])
            print("Agent Net Reward: ", agent.net_reward_history[-1])
            #i += EPISODES_BEFORE_TRAINING 
            if DISPLAY_HISTORY is True:
                try:
                    if args.lib_type == 'control' and args.train_autoencoder and i > 5:
                        #TODO: encapsulate this AE testing...and all else
                        #we now display forward dynamics to compare with 
                        #control trajectory
                        env.generate_state_history_plot()
                        #shift in predicted states, plot for side-by-side
                        #comparison
                        print("Compare with Autoencoder results!")
                        ae_history = [] 
                        for j in range(len(env.state_history)): 
                            s = env.state_history[j]
                            s = torch.tensor(s).to(device).float()
                            a = torch.tensor(np.ones(1)).to(device).float()
                            forward = autoencoder.forward_predict(s, a)
                            ae_history.append(forward)
                        #print("HISTORY: ", ae_history)
                        env.generate_state_history_plot(ae_history)
                        #plt.pause(0.01)
                        #plt.clf()

                            
                    plt.figure(1)
                    plt.subplot(2, 1, 1)
                    #plt.title("Algorithm:%s \n\
                    #        Activations: %s  Hdims: %s Outdims: %s\n\
                    #        lr=%s betas=%s eps=%s min_eps=%s eps_decay=%s\n\
                    #        gamma = %s"\
                    #        %(TRAINER_TYPE, mlp_activations,
                    #        mlp_hdims, mlp_outdim, 
                    #        lr, ADAM_BETAS, EPS, EPS_MIN, EPS_DECAY, GAMMA))
                    plt.title("Agent reward / loss history")
                    #graph.set_xdata(range(len(total_reward_history)))
                    #graph.set_ydata([r for r in total_reward_history])
                    #plt.scatter(range(len(total_reward_history)), [r.numpy()[0] for r in total_reward_history])
                    plt.xlim(0, len(agent.net_reward_history))
                    plt.ylim(min(agent.net_reward_history) - min(agent.net_reward_history) / 2, max(agent.net_reward_history) + max(agent.net_reward_history) / 2)
                    plt.ylabel("Net \n Reward")
                    plt.scatter(range(len(agent.net_reward_history)), [r for r in agent.net_reward_history], s=1.5, c='b')
                    if MA_LEN > 0 and len(agent.net_reward_history) > 0:
                        MA += agent.net_reward_history[-1]
                        val = MA #in order to divide
                        if i >= MA_LEN - 1: 
                            MA -= agent.net_reward_history[i - MA_LEN]
                            val = val / MA_LEN
                        else:
                            val = val / (i + 1)
                        averages.append(val)
                        #plt.plot(np.convolve(np.ones((MA_LEN,)), agent.net_reward_history, mode=m)) #alternative modes: 'full', 'same', 'valid'
                        plt.plot(range(len(agent.net_reward_history)), averages, '#FF4500') 
                        print("MA Reward: ", val)
                    plt.subplot(2, 1, 2)
                    plt.ylabel("Net \n Loss")
                    if type(agent.net_loss_history[0]) == float:
                        plt.scatter(range(len(agent.net_loss_history)), [r for r in agent.net_loss_history], s=1.5)
                    else:
                        plt.scatter(range(len(agent.net_loss_history)), [r.numpy()[0] for r in agent.net_loss_history], s=1.5)
                    if DISPLAY_AV_LOSS is True:
                        plt.figure(2)
                        plt.clf()
                        plt.subplot(2, 1, 1)
                        plt.ylabel("Action Loss")
                        plt.scatter(range(len(agent.action_loss_history)), [l.numpy() for l in agent.action_loss_history], s=0.1)
                        plt.subplot(2, 1, 2)
                        plt.ylabel("Value Loss")
                        plt.xlabel("Time")
                        plt.scatter(range(len(agent.value_loss_history)), [l.numpy() for l in agent.value_loss_history], s=0.1)
                        plt.draw()
                    plt.pause(0.01)
                except Exception as e:
                    print("ERROR: ", e)
            i += 1
    else:
        launch_viewer(env, agent)
