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
    def __init__(self, device, *args, **kwargs):
        super(PyTorchStateCartpoleAgent, self).__init__(*args, **kwargs)
        self.device = device
        self._module = None

    def step(self, obs) -> np.ndarray:
        '''Transforms observation, stores relevant data in replay buffers, then
        outputs action. 
        
        Outputs as np.ndarray because that's what the environment
        needs'''
        obs = self.transform_observation(obs).to(device)
        return super(PyTorchStateCartpoleAgent, self).step(obs)

    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, m):
        self._module = m

    def transform_observation(self, obs) -> Variable:
        '''Converts ordered-dictionary of position and velocity into
        1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
        pos = obs['position']
        vel = obs['velocity']
        state = np.concatenate((pos, vel))
        return Variable(torch.tensor(state).float(), requires_grad = True)

#class TFVisionCartpoleAgent(TFAgent): #but...would we WANT this??


LIB = 'pytorch'
MAX_ITERATIONS = 10
MAX_TIMESTEPS = 1000
view = True

mlp_outdim = 5 #based on state size (approximation)
mlp_hdims = [7, 10]
mlp_activations = ['relu', 'relu', None] #+1 for outdim activation, remember extra action/value modules
mlp_initializer = None

lr = 1.0e-4
ADAM_BETAS = (0.9, 0.999)
entropy_coeff = 1.0


episodes_before_replay = 5
replay_iterations = 20 #approximate based on episode length 
max_traj_len = 30
GAMMA = 0.98
if __name__ == '__main__':
    env = suite.load(domain_name = 'cartpole', task_name = 'swingup')  
    action_space = env.action_spec()
    obs_space = env.observation_spec()

    obs_size = obs_space['position'].shape[0] + obs_space['velocity'].shape[0]
    action_size = action_space.shape[0] 
    action_constraints = [action_space.minimum, action_space.maximum]
    print("Action Space: %s \n Observation Space: %s\n" % (action_size, obs_size))
    i = 0
    step = 0
    timestep = None
    if LIB == 'pytorch': #TODO : encapsulate this in a runner
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = PyTorchStateCartpoleAgent(device, [1, obs_size], action_size, discrete_actions = False,
                action_constraints = action_constraints, has_value_function = True)
    
        agent.module = PyTorchACMLP(action_size, action_bias = True, value_bias = True,
                device = device, indim = obs_size, outdim = mlp_outdim, hdims = mlp_hdims,
                activations=  mlp_activations, initializer = mlp_initializer).to(device)
        
        optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)

        trainer = PyTorchACTrainer(device, replay_iterations, entropy_coeff,
                agent, env, optimizer, replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA) 
        while i < MAX_ITERATIONS: #and error > threshold
            # Exploration / evaluation step
            timestep = env.reset()        
            while not timestep.last() and step < MAX_TIMESTEPS:
                if step > 0:
                    reward = timestep.reward
                else: #SKIP THIS?
                    reward = 0.0
                print("TIMESTEP %s: %s" % (i, timestep))
                observation = timestep.observation
                action = agent.step(observation).cpu().detach()
                #action = agent(timestep)
                timestep = env.step(action)
                print("Reward: %s" % (timestep.reward))
                step += 1
            # Update step
            #input("This is where the magic happens") 
            #TODO: FIX NAN issue (reward = None fucking up backpropogation)
            #TODO: FIX this so it calls step after some number of episodes
            #instead of after each episode
            #TODO: the versiono f AC being used currently is...likely meant for 
            #discrete action spaces. This is less of an issue now with a = 1
            #but WILL beocme an inssue in the future
            trainer.step()
            step = 0
            i += 1
    #TODO: COMPARE VS RANDOM AGENT!!!

    elif LIB == 'tf': #TODO: encapsulate this in a runner
        pass
    viewer.launch(env, policy = agent)

