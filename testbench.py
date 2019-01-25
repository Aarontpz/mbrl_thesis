import tensorflow as tf
import torch
import torch.optim as optim
from torch.autograd import Variable
import copy

import numpy as np
from agent import *
from trainer import *

from dm_control import suite
from dm_control import viewer

import gym

import matplotlib.pyplot as plt

import threading

VIEWING = False

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
            pos = obs['position']
            vel = obs['velocity']
            state = np.concatenate((pos, vel))
            return Variable(torch.tensor(state).float(), requires_grad = False)
    
        def reward(self, st, at, *args, **kwargs):
            upright = (self.env.physics.pole_angle_cosine() + 1) / 2
            return upright
    return DMCartpoleAgent(*args, **kwargs) 


def create_dm_walker_agent(agent_base, *args, **kwargs):
    '''Helper function to implement "transform_observation", and, in the case of MPC agents,
    a reward function, for the walker2d environment.'''
    class DMWalkerAgent(agent_base):
        def transform_observation(self, obs) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            orientation = obs['orientations']
            vel = obs['velocity']
            height = np.asarray([obs['height'],])
            state = np.concatenate((orientation, vel, height))
            return Variable(torch.tensor(state).float(), requires_grad = True)
    
        def reward(self, st, at, *args, **kwargs):
            global TASK_NAME #GROOOOOOSSSSSSSTHH
            if TASK_NAME in ['walk', 'run']:
                return env.physics.horizontal_velocity() #TODO: this is only valid for walk / run tasks
            else: 
                return 0.0
    
    return DMWalkerAgent(*args, **kwargs)

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
    return agent

#class TFVisionCartpoleAgent(TFAgent): #but...would we WANT this??

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

def console(env, agent, lock, lib_type = 'dm', env_type = 'walker'):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    #agent.device = device
    #agent.module.to(device)
    while True: 
        input()
        with lock:
            cmd = input('>')
            if cmd.lower() == 'v': #view with thread locked?
                print("VIEWING!")
                ## We create a clone of the agent (to preserve the training agent's history) 
                clone = create_agent(PyTorchAgent, LIB_TYPE, env_type, device, [1, obs_size], 
                        action_size, discrete_actions = DISCRETE_AGENT, 
                        action_constraints = action_constraints, has_value_function = True)
                #clone = agent.clone()
                clone.module = copy.deepcopy(agent.module)
                clone.module.to(device)
                launch_viewer(env, clone)    
                print("RESUMING!")

    



LIB = 'pytorch'
MAX_ITERATIONS = 10000
MAX_TIMESTEPS = 100000
VIEW_END = True

mlp_outdim = 200 #based on state size (approximation)
mlp_hdims = [200] 
#mlp_activations = ['relu', 'relu'] #+1 for outdim activation, remember extra action/value modules
mlp_activations = [None, 'relu'] #+1 for outdim activation, remember extra action/value modules
#mlp_outdim = 200 #based on state size (approximation)
#mlp_hdims = []
#mlp_activations = ['relu'] #+1 for outdim activation, remember extra action/value modules
mlp_initializer = None
DISCRETE_AGENT = False


lr = 1.0e-4
ADAM_BETAS = (0.9, 0.999)
MOMENTUM = 1e-3
MOMENTUM = 0
entropy_coeff = 10e-4
#entropy_coeff = 0 
ENTROPY_BONUS = False
value_coeff = 5e-1


DISPLAY_HISTORY = True
DISPLAY_AV_LOSS = False

PRETRAINED = False
FULL_EPISODE = True
MAXIMUM_TRAJECTORY_LENGTH = MAX_TIMESTEPS
SMALL_TRAJECTORY_LENGTH = 100

if FULL_EPISODE:
    max_traj_len = MAXIMUM_TRAJECTORY_LENGTH
    EPISODES_BEFORE_TRAINING = 1 #so we benefit from reusing sampled trajectories with PPO / TRPO
    replay_iterations = EPISODES_BEFORE_TRAINING #approximate based on episode length 
else:
    max_traj_len = SMALL_TRAJECTORY_LENGTH
    EPISODES_BEFORE_TRAINING = 5
    replay_iterations = 30 #approximate based on episode length 

LIB_TYPE = 'dm'
#LIB_TYPE = 'gym'

AGENT_TYPE = 'mpc'
AGENT_TYPE = 'policy'

TRAINER_TYPE = 'AC'
TRAINER_TYPE = 'PPO'

EPS = 0.5e-1
EPS_MIN = 2e-2
EPS_DECAY = 1e-6
GAMMA = 0.95
ENV_TYPE = 'walker'
TASK_NAME = 'run'
#TASK_NAME = 'walk'
#TASK_NAME = 'stand'

#EPS = 1e-1
#EPS_MIN = 2e-2
#EPS_DECAY = 1e-6
#GAMMA = 0.95
#ENV_TYPE = 'cartpole'
#TASK_NAME = 'swingup'
#TASK_NAME = 'balance'
if __name__ == '__main__':
    #raise Exception("It is time...for...asynchronous methods. I think. Investigate??")
    #raise Exception("It is time...for...preprocessing. I think. INVESTIGATE?!")
    #raise Exception("It is time...for...minibatches (vectorized) training. I think. INVESTIGATE?!")
    if LIB_TYPE == 'dm':
        env = suite.load(domain_name = ENV_TYPE, task_name = TASK_NAME)  
        tmp_env = suite.load(domain_name = ENV_TYPE, task_name = TASK_NAME)  
        action_space = env.action_spec()
        obs_space = env.observation_spec()

        if ENV_TYPE == 'walker':
            obs_size = obs_space['orientations'].shape[0] + obs_space['velocity'].shape[0] + 1 #+1 for height
        elif ENV_TYPE == 'cartpole':
            obs_size = obs_space['position'].shape[0] + obs_space['velocity'].shape[0]

        action_size = action_space.shape[0] 
        #action_size = 2
        action_constraints = [action_space.minimum, action_space.maximum]

        print("Action Space: %s \n Observation Space: %s\n" % (action_size, obs_size))
        print("Agent IS: Discrete: %s; Traj Length: %s; Replays: %s" % (DISCRETE_AGENT,
            max_traj_len, replay_iterations))
        print("Trainer Type: %s" % (TRAINER_TYPE))
        print("MLP ACTIVATIONS: ", mlp_activations)
    elif LIB_TYPE == 'gym':
        if ENV_TYPE == 'walker':
            env_string = 'Walker2d-v2'    
        elif ENV_TYPE == 'cartpole':
            env_string = 'CartPole-v0'    
        env = gym.make(env_string)
        tmp_env = gym.make(env_string)
        action_size = env.action_space.shape[0]
        action_constraints = [env.action_space.low, env.action_space.high]
        env.reset()
        obs, reward, done, info = env.step(1)
        obs_size = obs.size

    i = 0
    step = 0
    timestep = None
    trainer = None 
    if LIB == 'pytorch': #TODO : encapsulate this in a runner
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu") 
         
        if AGENT_TYPE == 'mpc': 
            NUM_PROCESSES = 2
            HORIZON = 40
            K_SHOOTS = 20
            #mlp_outdim = 500 #based on state size (approximation)
            mlp_hdims = [500, 500] 
            mlp_activations = ['relu', 'relu', None] #+1 for outdim activation, remember extra action/value modules
            agent = create_agent(PyTorchMPCAgent, LIB_TYPE, ENV_TYPE, 
                        NUM_PROCESSES, HORIZON, K_SHOOTS,  #num_processes, #horizon, k_shoots
                        device, 
                        [1, obs_size], action_size, discrete_actions = DISCRETE_AGENT, 
                        action_constraints = action_constraints, has_value_function = False)
            if ENV_TYPE == 'walker':
                random_agent = RandomWalkerAgent([1, obs_size], action_size, 
                        discrete_actions = DISCRETE_AGENT, 
                        action_constraints = action_constraints, has_value_function = False)
            elif ENV_TYPE == 'cartpole':
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
                    agent, env, optimizer, replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA,
                    num_episodes = EPISODES_BEFORE_TRAINING) 
            trainer.step() #RUN PRETRAINING STEP
            PRETRAINED = True

        #pertinent from pg 9 of 1708.02596: "...ADAM optimizer lr = 0.001, batch size 512.
        #prior to training both the inputs and outputs in the dataset were preprocessed to have
        #zero mean, std dev = 1"
        #TODO TODO:^this could include a value function too?! SEPERATE VALUE FUNCTION
        # to benefit from MPC AND model-free agent sampling, since value functions are ubiquitous

        elif AGENT_TYPE == 'policy':
            agent = create_agent(PyTorchAgent, LIB_TYPE, ENV_TYPE, device, [1, obs_size], 
                        action_size, discrete_actions = DISCRETE_AGENT, 
                        action_constraints = action_constraints, has_value_function = True)
            if DISCRETE_AGENT:
                mlp_base = PyTorchDiscreteACMLP
            else:
                mlp_base = PyTorchContinuousGaussACMLP
            agent.module = EpsGreedyMLP(mlp_base, EPS, EPS_DECAY, EPS_MIN, action_constraints, 
                    action_size, seperate_value_network = True, 
                    action_bias = True, value_bias = True, sigma_head = True, 
                    device = device, indim = obs_size, outdim = mlp_outdim, hdims = mlp_hdims,
                    activations = mlp_activations, initializer = mlp_initializer).to(device)
            
            #optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)
            optimizer = optim.SGD(agent.module.parameters(), lr = lr, momentum = MOMENTUM)
            scheduler = None
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.85)
            if TRAINER_TYPE == 'AC':
                trainer = PyTorchACTrainer(value_coeff, entropy_coeff, ENTROPY_BONUS,
                        device, optimizer, scheduler,   
                        agent, env, 
                        replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA,
                        num_episodes = EPISODES_BEFORE_TRAINING) 
            elif TRAINER_TYPE == 'PPO':
                trainer = PyTorchPPOTrainer(value_coeff, entropy_coeff, ENTROPY_BONUS, 
                        device, optimizer, scheduler, 
                        agent, env, 
                        replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA,
                        num_episodes = EPISODES_BEFORE_TRAINING) 

        ## RUN AGENT / TRAINING
        if not PRETRAINED:
            ## set up listener for user input
            lock = threading.Lock()
            console_args = (tmp_env, agent, lock, LIB_TYPE, ENV_TYPE)
            console_thread = threading.Thread(target = console, args = console_args)
            console_thread.daemon = True
            console_thread.start()
            ##
            while i < MAX_ITERATIONS: #and error > threshold
                print("ITERATION: ", i)
                # Exploration / evaluation step
                for episode in range(EPISODES_BEFORE_TRAINING):
                    if LIB_TYPE == 'dm':
                        timestep = env.reset()        
                        while not timestep.last() and step < MAX_TIMESTEPS:
                            reward = timestep.reward
                            if reward is None:
                                reward = 0.0
                            #print("TIMESTEP %s: %s" % (step, timestep))
                            observation = timestep.observation
                            #print("Observation: ", observation)
                            action = agent(timestep)
                            #print("Action: ", action)
                            agent.store_reward(reward)
                            timestep = env.step(action)
                            #print("Reward: %s" % (timestep.reward))
                            step += 1
                    elif LIB_TYPE == 'gym':
                        env.reset()
                        observation, reward, done, info = env.step(env.action_space.sample())
                        while not done and step < MAX_TIMESTEPS:
                            action = agent.step(observation).cpu().numpy()
                            env.render()
                            observation, reward, done, info = env.step(action)
                            agent.store_reward(reward)
                            step += 1
                    step = 0
                    agent.terminate_episode() #oops forgot this lmao
                # Update step
                if not PRETRAINED:
                    trainer.step()
                    print("Agent Net Loss: ", agent.net_loss_history[i])

                agent.reset_histories()
                print("Agent Net Reward: ", agent.net_reward_history[i])
                #i += EPISODES_BEFORE_TRAINING 
                if DISPLAY_HISTORY is True:
                    plt.figure(1)
                    plt.subplot(2, 1, 1)
                    plt.title("Algorithm:%s \n\
                            Activations: %s  Hdims: %s Outdims: %s\n\
                            lr=%s betas=%s eps=%s min_eps=%s eps_decay=%s\n\
                            gamma = %s"\
                            %(TRAINER_TYPE, mlp_activations,
                            mlp_hdims, mlp_outdim, 
                            lr, ADAM_BETAS, EPS, EPS_MIN, EPS_DECAY, GAMMA))
                    #graph.set_xdata(range(len(total_reward_history)))
                    #graph.set_ydata([r for r in total_reward_history])
                    #plt.scatter(range(len(total_reward_history)), [r.numpy()[0] for r in total_reward_history])
                    plt.xlim(0, len(agent.net_reward_history))
                    plt.ylim(0, max(agent.net_reward_history)+1)
                    plt.ylabel("Net \n Reward")
                    plt.scatter(range(len(agent.net_reward_history)), [r for r in agent.net_reward_history], s=1.0)
                    plt.subplot(2, 1, 2)
                    plt.ylabel("Net \n Loss")
                    plt.scatter(range(len(agent.net_loss_history)), [r.numpy()[0] for r in agent.net_loss_history], s=1.0)
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
                i += 1
        else:
            launch_viewer(env, agent)

    #TODO: COMPARE VS RANDOM AGENT!!!

    elif LIB == 'tf': #TODO: encapsulate this in a runner
        ## set up listener for user input
        console_args = (env, agent, lock)
        console_thread = threading.Thread(target = console, args = console_args)
        console_thread.daemon = True
        console_thread.start()
        ##
        ## RUN AGENT / TRAINING
    
    if VIEW_END:
        launch_viewer(env, agent)
