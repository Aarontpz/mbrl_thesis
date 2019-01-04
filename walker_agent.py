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
EPS_DECAY = 1e-8

mlp_outdim = 50 #based on state size (approximation)
mlp_hdims = [200]
mlp_activations = [None, 'relu'] #+1 for outdim activation, remember extra action/value modules
mlp_initializer = None
DISCRETE_AGENT = False
FULL_EPISODE = True

MAXIMUM_TRAJECTORY_LENGTH = MAX_TIMESTEPS
SMALL_TRAJECTORY_LENGTH = 100

lr = 1.0e-4
ADAM_BETAS = (0.9, 0.999)
entropy_coeff = 0.0
value_coeff = 0.1

if FULL_EPISODE:
    max_traj_len = MAXIMUM_TRAJECTORY_LENGTH
    EPISODES_BEFORE_TRAINING = 5 #so we benefit from reusing sampled trajectories with PPO / TRPO
    replay_iterations = EPISODES_BEFORE_TRAINING #approximate based on episode length 
else:
    max_traj_len = SMALL_TRAJECTORY_LENGTH
    EPISODES_BEFORE_TRAINING = 20
    replay_iterations = 30 #approximate based on episode length 


GAMMA = 0.96
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
    i = 0
    step = 0
    timestep = None
    
    if LIB == 'pytorch': #TODO : encapsulate this in a runner
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = PyTorchStateWalkerAgent(device, [1, obs_size], action_size, discrete_actions = DISCRETE_AGENT, 
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
            
        optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)

        #trainer = PyTorchACTrainer(device, value_coeff, entropy_coeff,
        #        agent, env, optimizer, replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA,
        #        num_episodes = EPISODES_BEFORE_TRAINING) 
        trainer = PyTorchPPOTrainer(device, value_coeff, entropy_coeff,
                agent, env, optimizer, replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA,
                num_episodes = EPISODES_BEFORE_TRAINING) 

        ## set up listener for user input
        lock = threading.Lock()
        console_args = (tmp_env, agent, lock)
        console_thread = threading.Thread(target = console, args = console_args)
        console_thread.daemon = True
        console_thread.start()
        ##
        ## RUN AGENT / TRAINING
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
                    action = agent(timestep)
                    agent.store_reward(reward)
                    timestep = env.step(action)
                    #print("Reward: %s" % (timestep.reward))
                    step += 1
                step = 0
                agent.terminate_episode() #oops forgot this lmao
            # Update step
            trainer.step()
            agent.reset_histories()
            print("Agent Net Reward: ", agent.net_reward_history[i])
            print("Agent Net Loss: ", agent.net_loss_history[i])
            #i += EPISODES_BEFORE_TRAINING 
            i += 1
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
