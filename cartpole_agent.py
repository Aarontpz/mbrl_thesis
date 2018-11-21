import tensorflow as tf
import torch
import numpy as np
from agent import *
from trainer import *

from dm_control import suite
from dm_control import viewer


class TFStateCartpoleAgent(TFAgent):
    def __init__():
        pass


class PyTorchStateCartpoleAgent(PyTorchAgent):
    def __init__(self, *args, **kwargs):
        super(PyTorchStateCartpoleAgent, self).__init__(*args, **kwargs)
        self._module = None
    
    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, m):
        self._module = m

#class TFVisionCartpoleAgent(TFAgent): #but...would we WANT this??


LIB = 'pytorch'
iterations = 1000
view = True

mlp_outdim = 5 #based on state size (approximation)
mlp_hdims = [7, 10]
mlp_activations = ['relu', 'relu', 'relu'] #+1 for outdim activation, remember extra action/value modules
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


    if LIB == 'pytorch': #TODO : encapsulate this in a runner
        agent = PyTorchStateCartpoleAgent([1, obs_size], action_size, discrete_actions = False,
                action_constraints = action_constraints, has_value_function = True)
    
        agent.module = PyTorchACMLP(action_size, action_bias = True, value_bias = True,
                indim = obs_size, outdim = mlp_outdim, hdims = mlp_hdims,
                activations=  mlp_activations, initializer = mlp_initializer)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)

        trainer = PyTorchACTrainer(device, replay_iterations = replay_iterations, entropy_coeff = ent_coeff,
                agent, env, optimizer, replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA) 


    elif LIB == 'tf': #TODO: encapsulate this in a runner
        pass


