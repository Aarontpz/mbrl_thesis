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


import matplotlib.pyplot as plt

import threading

VIEWING = False
class TFStateCartpoleAgent(TFAgent):
    def __init__():
        pass


class PyTorchStateWalkerAgent(PyTorchAgent):
    def transform_observation(self, obs) -> Variable:
        '''Converts ordered-dictionary of position and velocity into
        1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
        orientation = obs['orientations']
        vel = obs['velocity']
        height = np.asarray([obs['height'],])
        state = np.concatenate((orientation, vel, height))
        return Variable(torch.tensor(state).float(), requires_grad = True)

class DMEnvMPCWalkerAgent(DMEnvMPCAgent):
    def transform_observation(self, obs) -> Variable:
        '''Converts ordered-dictionary of position and velocity into
        1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
        orientation = obs['orientations']
        #vel = obs['velocity']
        #height = np.asarray([obs['height'],])
        #state = np.concatenate((orientation, vel, height))
        return orientation


class PyTorchMPCWalkerAgent(PyTorchMPCAgent):
    def transform_observation(self, obs) -> Variable:
        '''Converts ordered-dictionary of position and velocity into
        1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
        orientation = obs['orientations']
        vel = obs['velocity']
        height = np.asarray([obs['height'],])
        state = np.concatenate((orientation, vel, height))
        return Variable(torch.tensor(state).float(), requires_grad = True)

    def reward(self, st, at, *args, **kwargs):
        pass

class RandomWalkerAgent(RandomAgent):
    def transform_observation(self, obs) -> Variable:
        '''Converts ordered-dictionary of position and velocity into
        1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
        orientation = obs['orientations']
        vel = obs['velocity']
        height = np.asarray([obs['height'],])
        state = np.concatenate((orientation, vel, height))
        return Variable(torch.tensor(state).float(), requires_grad = True)



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

def console(env, agent, lock):
    while True: 
        input()
        with lock:
            cmd = input('>')
            if cmd.lower() == 'v': #view with thread locked?
                print("VIEWING!")
                ## We create a clone of the agent (to preserve the training agent's history) 
                clone = PyTorchStateWalkerAgent(device, [1, obs_size], action_size, discrete_actions = DISCRETE_AGENT, 
                        action_constraints = action_constraints, has_value_function = True) 
                clone.module = copy.deepcopy(agent.module)
                launch_viewer(env, clone)    
                print("RESUMING!")

    



LIB = 'pytorch'
MAX_ITERATIONS = 10000
MAX_TIMESTEPS = 100000
VIEW_END = True

EPS = 0.08
EPS_MIN = 0.01
EPS_DECAY = 1e-6

mlp_outdim = 100 #based on state size (approximation)
mlp_hdims = [200]
mlp_activations = ['relu', None] #+1 for outdim activation, remember extra action/value modules
#mlp_activations = [None, 'relu'] #+1 for outdim activation, remember extra action/value modules
#mlp_outdim = 200 #based on state size (approximation)
#mlp_hdims = []
#mlp_activations = ['relu'] #+1 for outdim activation, remember extra action/value modules
mlp_initializer = None
DISCRETE_AGENT = False
FULL_EPISODE = True
GAMMA = 0.99

MAXIMUM_TRAJECTORY_LENGTH = MAX_TIMESTEPS
SMALL_TRAJECTORY_LENGTH = 100

lr = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
entropy_coeff = 0.0
value_coeff = 0.1

TRAINER_TYPE = 'AC'
#TRAINER_TYPE = 'PPO'

DISPLAY_HISTORY = True

PRETRAINED = False

if FULL_EPISODE:
    max_traj_len = MAXIMUM_TRAJECTORY_LENGTH
    EPISODES_BEFORE_TRAINING = 3 #so we benefit from reusing sampled trajectories with PPO / TRPO
    replay_iterations = EPISODES_BEFORE_TRAINING #approximate based on episode length 
else:
    max_traj_len = SMALL_TRAJECTORY_LENGTH
    EPISODES_BEFORE_TRAINING = 20
    replay_iterations = 30 #approximate based on episode length 


if __name__ == '__main__':
    #raise Exception("It is time...for...asynchronous methods. I think. Investigate??")
    #raise Exception("It is time...for...preprocessing. I think. INVESTIGATE?!")
    #raise Exception("It is time...for...minibatches (vectorized) training. I think. INVESTIGATE?!")
    env = suite.load(domain_name = 'walker', task_name = 'walk')  
    tmp_env = suite.load(domain_name = 'walker', task_name = 'walk')  
    action_space = env.action_spec()
    obs_space = env.observation_spec()

    obs_size = obs_space['orientations'].shape[0] + obs_space['velocity'].shape[0] + 1 #+1 for height
    action_size = action_space.shape[0] 
    #action_size = 2
    action_constraints = [action_space.minimum, action_space.maximum]

    print("Action Space: %s \n Observation Space: %s\n" % (action_size, obs_size))
    print("Agent IS: Discrete: %s; Traj Length: %s; Replays: %s" % (DISCRETE_AGENT,
        max_traj_len, replay_iterations))
    print("Trainer Type: %s" % (TRAINER_TYPE))
    print("MLP ACTIVATIONS: ", mlp_activations)
    i = 0
    step = 0
    timestep = None
    trainer = None 
    if LIB == 'pytorch': #TODO : encapsulate this in a runner
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu") 
        random_agent = RandomWalkerAgent([1, obs_size], action_size, 
                discrete_actions = DISCRETE_AGENT, 
                action_constraints = action_constraints, has_value_function = False)
         
        #env_clone = suite.load(domain_name = 'walker', task_name = 'stand')  
        #agent = PyTorchMPCWalkerAgent(10, 1000,  #horizon, k_shoots
        #        device, 
        #        [1, obs_size], action_size, discrete_actions = DISCRETE_AGENT, 
        #        action_constraints = action_constraints, has_value_function = False)
        
    
        agent = PyTorchMPCWalkerAgent(30, 30,  #horizon, k_shoots
                device, 
                [1, obs_size], action_size, discrete_actions = DISCRETE_AGENT, 
                action_constraints = action_constraints, has_value_function = False)
        agent.module = PyTorchMLP(device, obs_size + action_size, obs_size, 
                hdims = [500, 500], activations = ['relu', 'relu', None], 
                initializer = mlp_initializer).to(device)    
        lr = 1.0e-3
        ADAM_BETAS = (0.9, 0.999)
        optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)
        trainer = PyTorchNeuralDynamicsMPCTrainer(agent, random_agent, 
                512, 1.0, 0.05, 0.5, 10, 100, #batch_size, starting rand, rand_decay, rand min, max steps, max iter        
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

        #agent = PyTorchStateWalkerAgent(device, [1, obs_size], action_size, discrete_actions = DISCRETE_AGENT, 
        #        action_constraints = action_constraints, has_value_function = True)
        #if DISCRETE_AGENT:
        #    mlp_base = PyTorchDiscreteACMLP
        #else:
        #    mlp_base = PyTorchContinuousGaussACMLP
        #agent.module = EpsGreedyMLP(mlp_base, EPS, EPS_DECAY, EPS_MIN, action_constraints, 
        #        action_size, seperate_value_network = True, 
        #        action_bias = True, value_bias = True, sigma_head = True, 
        #        device = device, indim = obs_size, outdim = mlp_outdim, hdims = mlp_hdims,
        #        activations = mlp_activations, initializer = mlp_initializer).to(device)
        #
        #optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)
        #if TRAINER_TYPE == 'AC':
        #    trainer = PyTorchACTrainer(device, value_coeff, entropy_coeff,
        #            agent, env, optimizer, replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA,
        #            num_episodes = EPISODES_BEFORE_TRAINING) 
        #elif TRAINER_TYPE == 'PPO':
        #    trainer = PyTorchPPOTrainer(device, value_coeff, entropy_coeff,
        #            agent, env, optimizer, replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA,
        #            num_episodes = EPISODES_BEFORE_TRAINING) 

        ## set up listener for user input
        lock = threading.Lock()
        console_args = (tmp_env, agent, lock)
        console_thread = threading.Thread(target = console, args = console_args)
        console_thread.daemon = True
        console_thread.start()
        ##
        ## RUN AGENT / TRAINING
        if not PRETRAINED:
            while i < MAX_ITERATIONS: #and error > threshold
                print("ITERATION: ", i)
                # Exploration / evaluation step
                for episode in range(EPISODES_BEFORE_TRAINING):
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
                    #plt.figure(2)
                    #plt.clf()
                    #plt.subplot(2, 1, 1)
                    #plt.ylabel("Action Loss")
                    #plt.scatter(range(len(action_loss_history)), [l.numpy()[0] for l in action_loss_history], s=0.1)
                    #plt.subplot(2, 1, 2)
                    #plt.ylabel("Value Loss")
                    #plt.xlabel("Time")
                    #plt.scatter(range(len(value_loss_history)), [l.numpy()[0] for l in value_loss_history], s=0.1)
                    #plt.draw()
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
