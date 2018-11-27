import tensorflow as tf
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from agent import *
from trainer import *

from dm_control import suite
from dm_control import viewer


class TFStateCartpoleAgent(TFAgent):
    def __init__():
        pass


class PyTorchStateCartpoleAgent(PyTorchAgent):
    def transform_observation(self, obs) -> Variable:
        '''Converts ordered-dictionary of position and velocity into
        1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
        pos = obs['position']
        vel = obs['velocity']
        state = np.concatenate((pos, vel))
        return Variable(torch.tensor(state).float(), requires_grad = True)


#class TFVisionCartpoleAgent(TFAgent): #but...would we WANT this??


LIB = 'pytorch'
MAX_ITERATIONS = 100
MAX_TIMESTEPS = 100000
VIEW = True

MLP_EPS = 0.1
mlp_outdim = 5 #based on state size (approximation)
mlp_hdims = [7, 10]
mlp_activations = ['relu', 'relu', None] #+1 for outdim activation, remember extra action/value modules
mlp_initializer = None
DISCRETE_AGENT = True


lr = 1.0e-4
ADAM_BETAS = (0.9, 0.999)
entropy_coeff = 1.0


EPISODES_BEFORE_TRAINING = 5
replay_iterations = 20 #approximate based on episode length 
max_traj_len = 30
GAMMA = 0.98
if __name__ == '__main__':
    env = suite.load(domain_name = 'cartpole', task_name = 'swingup')  
    action_space = env.action_spec()
    obs_space = env.observation_spec()

    obs_size = obs_space['position'].shape[0] + obs_space['velocity'].shape[0]
    #action_size = action_space.shape[0] 
    action_size = 2
    action_constraints = [action_space.minimum, action_space.maximum]
    print("Action Space: %s \n Observation Space: %s\n" % (action_size, obs_size))
    i = 0
    step = 0
    timestep = None
    if LIB == 'pytorch': #TODO : encapsulate this in a runner
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = PyTorchStateCartpoleAgent(device, [1, obs_size], action_size, discrete_actions = False, #TODO: d_a = DISCRETE_AGENT?
                action_constraints = action_constraints, has_value_function = True)
    
        #agent.module = PyTorchDiscreteACMLP(action_size, action_bias = True, value_bias = True,
        #        device = device, indim = obs_size, outdim = mlp_outdim, hdims = mlp_hdims,
        #        activations=  mlp_activations, initializer = mlp_initializer).to(device)
        if DISCRETE_AGENT:
            agent.module = EpsGreedyMLP(MLP_EPS, PyTorchDiscreteACMLP,
                    action_size, action_bias = True, value_bias = True,
                    device = device, indim = obs_size, outdim = mlp_outdim, hdims = mlp_hdims,
                    activations=  mlp_activations, initializer = mlp_initializer).to(device)
        
        optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)

        trainer = PyTorchDiscreteACTrainer(device, entropy_coeff,
                agent, env, optimizer, replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA) 
        while i < MAX_ITERATIONS: #and error > threshold
            # Exploration / evaluation step
            for episode in range(EPISODES_BEFORE_TRAINING):
                timestep = env.reset()        
                while not timestep.last() and step < MAX_TIMESTEPS:
                    reward = timestep.reward
                    if reward is None:
                        reward = 0.0
                    print("%s TIMESTEP %s: %s" % (i, step, timestep))
                    observation = timestep.observation
                    action = agent.step(observation).cpu().detach()
                    #discrete solution
                    if DISCRETE_AGENT:
                        action_ind = max(range(len(action)), key=action.__getitem__)
                        action = [-1, 1][action_ind]
                    #continuous solution
                    agent.store_reward(reward)
                    print("ACTION: ", action)
                    #action = agent(timestep)
                    timestep = env.step(action)
                    #print("Reward: %s" % (timestep.reward))
                    step += 1
                step = 0
            # Update step
            #input("This is where the magic happens") 
            #TODO: FIX NAN issue (reward = None fucking up backpropogation)
            #TODO: FIX this so it calls step after some number of episodes
            #instead of after each episode
            #TODO: the versiono f AC being used currently is...likely meant for 
            #discrete action spaces. This is less of an issue now with a = 1
            #but WILL beocme an inssue in the future
            #TODO: discounted_reward estimate should contain V(sT) in order to 
            #bootstrap...?
            trainer.step()
            agent.reset_histories()
            #i += EPISODES_BEFORE_TRAINING 
            i += 1
    #TODO: COMPARE VS RANDOM AGENT!!!

    elif LIB == 'tf': #TODO: encapsulate this in a runner
        pass
    
    
    if VIEW:
        viewer.launch(env, policy = agent)

