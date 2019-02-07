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
            if type(obs) == type(collections.OrderedDict()):
                pos = obs['position']
                vel = obs['velocity']
                state = np.concatenate((pos, vel))
            elif type(obs) == type(np.zeros(0)):
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

def create_dm_humanoid_agent(agent_base, *args, **kwargs):
    '''Helper function to implement "transform_observation", and, in the case of MPC agents,
    a reward function, for the walker2d environment.'''
    class DMHumanoidAgent(agent_base):
        def transform_observation(self, obs, extra_info = False) -> Variable:
            '''Converts ordered-dictionary of position and velocity into
            1D tensor. DOES NOT NORMALIZE, CURRENTLY'''
            if type(obs) == type(collections.OrderedDict()):
                angles = obs['joint_angles']
                extremities = obs['extremities']
                velocity = obs['velocity']
                state = np.concatenate((angles, extremities, velocity))
                if extra_info:
                    com_velocity = obs['com_velocity']
                    head_height = obs['head_height']
                    torso_vertical = obs['torso_vertical']
                    state = np.concatenate((state, com_velocity, head_height, torso_vertical))
            elif type(obs) == type(np.zeros(0)):
                state = obs
            return Variable(torch.tensor(state).float(), requires_grad = True)
    
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
        elif env_type == 'humanoid':
            return create_dm_humanoid_agent(agent_base, *args, **kwargs)
    else:
        return create_numpy_agent(agent_base, *args, **kwargs)
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
                clone = create_agent(PyTorchAgent, LIB_TYPE, env_type, device, [1, obs_size], 
                        action_size, discrete_actions = DISCRETE_AGENT, 
                        action_constraints = action_constraints, has_value_function = True, 
                        encode_inputs = encode_inputs, encoder = encoder_clone)
                #clone = agent.clone()
                clone.module = copy.deepcopy(agent.module)
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

LIB = 'pytorch'
MAX_ITERATIONS = 10000
MAX_TIMESTEPS = 100000
VIEW_END = True

#mlp_outdim = 200 #based on state size (approximation)
#mlp_hdims = [200] 
#mlp_activations = ['relu', 'relu'] #+1 for outdim activation, remember extra action/value modules
#mlp_activations = [None, 'relu'] #+1 for outdim activation, remember extra action/value modules
#mlp_outdim = 200 #based on state size (approximation)
#mlp_hdims = []
#mlp_activations = ['relu'] #+1 for outdim activation, remember extra action/value modules
WIDENING_CONST = 2 #indim * WIDENING_CONST = hidden layer size
mlp_initializer = None
DISCRETE_AGENT = False




DISPLAY_HISTORY = True
DISPLAY_AV_LOSS = True

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




PRETRAINED = True
RUN_ANYWAYS = True

#LIB_TYPE = 'dm'
#LIB_TYPE = 'gym'
LIB_TYPE = 'control'

#AGENT_TYPE = 'mpc'
AGENT_TYPE = 'policy'

TRAINER_TYPE = 'AC'
#TRAINER_TYPE = 'PPO'
lr = 0.5e-3
ADAM_BETAS = (0.9, 0.999)
MOMENTUM = 1e-3
#MOMENTUM = 0
entropy_coeff = 5e-3
#entropy_coeff = 0 
ENTROPY_BONUS = False
value_coeff = 5e-2

#energy_penalty_coeff = 5e-2 #HIGH, if energy is a consideration
energy_penalty_coeff = 5e-4 #low, if energy isn't a consideration
#energy_penalty_coeff = 0 #zero, to not penalize at all

EPS = 1.0e-1
EPS_MIN = 1e-2
EPS_DECAY = 1e-8
GAMMA = 0.98
ENV_TYPE = 'humanoid'
#TASK_NAME = 'run'
#TASK_NAME = 'walk'
TASK_NAME = 'stand'

EPS = 0.5e-1
EPS_MIN = 2e-2
EPS_DECAY = 1e-6
GAMMA = 0.98
ENV_TYPE = 'walker'
#TASK_NAME = 'run'
TASK_NAME = 'walk'
#TASK_NAME = 'stand'

#EPS = 0.7e-1
#EPS_MIN = 2e-2
#EPS_DECAY = 1e-7
#GAMMA = 0.98
#ENV_TYPE = 'cartpole'
#TASK_NAME = 'swingup'
#TASK_NAME = 'balance'

#CONTROL ENVIRONMENTS
EPS = 0.5e-1
EPS_MIN = 2e-2
EPS_DECAY = 1e-6
GAMMA = 0.98
ENV_TYPE = 'rossler'
ENV_KWARGS = {'noisy_init' : True, 'ts' : 0.0001, 'interval' : 10}
TASK_NAME = 'point'

MA_LEN = -1
MA_LEN = 20

MAXMIN_NORMALIZATION = False
TRAIN_AUTOENCODER = True
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
        elif ENV_TYPE == 'humanoid':
            obs_size = obs_space['joint_angles'].shape[0] + obs_space['extremities'].shape[0]
            obs_size = obs_size + obs_space['velocity'].shape[0]#+1 for height

        action_size = action_space.shape[0] 
        #action_size = 2
        action_constraints = [action_space.minimum, action_space.maximum]

        print("Action Space: %s \n Observation Space: %s\n" % (action_size, obs_size))
        print("Agent IS: Discrete: %s; Traj Length: %s; Replays: %s" % (DISCRETE_AGENT,
            max_traj_len, replay_iterations))
        print("Trainer Type: %s" % (TRAINER_TYPE))
    elif LIB_TYPE == 'gym':
        if ENV_TYPE == 'walker':
            env_string = 'Walker2d-v2'    
        elif ENV_TYPE == 'cartpole':
            env_string = 'CartPole-v0'    
        env = gym.make(env_string)
        tmp_env = gym.make(env_string)
        #print("Action space: ", env.action_space)
        action_size = env.action_space.shape[0]
        action_constraints = [env.action_space.low, env.action_space.high]
        env.reset()
        obs, reward, done, info = env.step(1)
        obs_size = obs.size

    elif LIB_TYPE == 'control':
        env = retrieve_control_environment(ENV_TYPE, **ENV_KWARGS)
        tmp_env = retrieve_control_environment(ENV_TYPE, **ENV_KWARGS)
        action_size = env.get_action_size()
        action_constraints = env.get_action_constraints()
        obs_size = env.get_observation_size()
    
    MA = 0
    averages = []

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
            mlp_indim = obs_size
            mlp_activations = [None, None, 'relu'] #+1 for outdim activation, remember extra action/value modules
            mlp_hdims = [mlp_indim * WIDENING_CONST, mlp_indim,] 
            mlp_outdim = mlp_indim * WIDENING_CONST #based on state size (approximation)
            print("MLP INDIM: %s HDIM: %s OUTDIM: %s " % (obs_size, mlp_hdims, mlp_outdim))
            print("MLP ACTIVATIONS: ", mlp_activations)
            agent = create_agent(PyTorchAgent, LIB_TYPE, ENV_TYPE, device, [1, obs_size], 
                        action_size, discrete_actions = DISCRETE_AGENT, 
                        action_constraints = action_constraints, has_value_function = True,
                        terminal_penalty = 0.0, 
                        policy_entropy_history = True, energy_penalty_history = True)
            if DISCRETE_AGENT:
                mlp_base = PyTorchDiscreteACMLP
            else:
                mlp_base = PyTorchContinuousGaussACMLP
            agent.module = EpsGreedyMLP(mlp_base, EPS, EPS_DECAY, EPS_MIN, action_constraints, 
                    action_size, 
                    seperate_value_module = None, seperate_value_module_input = False,
                    value_head = True,
                    action_bias = True, value_bias = True, sigma_head = True, 
                    device = device, indim = obs_size, outdim = mlp_outdim, hdims = mlp_hdims,
                    activations = mlp_activations, initializer = mlp_initializer).to(device)
            
            optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)
            #optimizer = optim.SGD(agent.module.parameters(), lr = lr, momentum = MOMENTUM)
            scheduler = None
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 300, gamma = 0.85)
            if TRAINER_TYPE == 'AC':
                trainer = PyTorchACTrainer(value_coeff, entropy_coeff, ENTROPY_BONUS, energy_penalty_coeff,
                        device, optimizer, scheduler,   
                        agent, env, 
                        replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA,
                        num_episodes = EPISODES_BEFORE_TRAINING) 
            elif TRAINER_TYPE == 'PPO':
                trainer = PyTorchPPOTrainer(value_coeff, entropy_coeff, ENTROPY_BONUS, energy_penalty_coeff,
                        device, optimizer, scheduler, 
                        agent, env, 
                        replay = replay_iterations, max_traj_len = max_traj_len, gamma = GAMMA,
                        num_episodes = EPISODES_BEFORE_TRAINING) 
            print("AGENT MODULE (pre-autoencoder?)", agent.module)
            ## Autoencoder addition
            autoencoder_base = PyTorchLinearAutoencoder
            autoencoder = None
            autoencoder_trainer = None
            #AE_BATCHES = None
            TRAIN_FORWARD = True
            TRAIN_ACTION = True

            mlp_outdim = None
            DEPTH = 3
            UNIFORM_LAYERS = False
            REDUCTION_FACTOR = 0.8
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
            AE_REPLAYS = 20
            DATASET_RECENT_PROB = 0.75
            if action_size > 1:
                mlp_indim = math.floor(obs_size * REDUCTION_FACTOR**DEPTH) + math.floor(action_size * REDUCTION_FACTOR**DEPTH)
            else:
                mlp_indim = math.floor(obs_size * REDUCTION_FACTOR**DEPTH) + 1
            #TODO: ^ this works...in both cases (coupled_sa or not)
            mlp_outdim = obs_size * (REDUCTION_FACTOR**DEPTH)
            mlp_hdims = [mlp_indim * WIDENING_CONST, mlp_indim * WIDENING_CONST,] 
            mlp_activations = ['relu', 'relu', None] #+1 for outdim activation, remember extra action/value modules
            
            if LINEAR_FORWARD:
                raise Exception("Reduce to the sum of TWO matrices representing \
                        Ax + Bu where x=state and u=action")
                forward_mlp = PyTorchMLP(device, mlp_indim, mlp_outdim, 
                        hdims = mlp_hdims, activations = mlp_activations, 
                        initializer = mlp_initializer).to(device)
            else:
                forward_mlp = PyTorchMLP(device, mlp_indim, math.floor(mlp_outdim), 
                        hdims = mlp_hdims, activations = mlp_activations, 
                        initializer = mlp_initializer).to(device)   

            if TRAIN_AUTOENCODER == True:

                #AUTOENCODER_DATASET = Dataset(aggregate_examples = True, shuffle = True)
                AUTOENCODER_DATASET = DAgger(recent_prob = DATASET_RECENT_PROB, aggregate_examples = False, shuffle = True)
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
                        replay = AE_REPLAYS, max_traj_len = max_traj_len, gamma = GAMMA,
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
                            action_bias = True, value_bias = True, sigma_head = True, 
                            device = device, indim = mlp_indim, outdim = action_size, hdims = mlp_hdims,
                            activations = mlp_activations, initializer = mlp_initializer,
                            ).to(device)
                    agent.encode_inputs = True
                    agent.encoder = autoencoder
                    print("Post-encoder Module: ", agent.module)
                    optimizer = optim.Adam(agent.module.parameters(), lr = lr, betas = ADAM_BETAS)
                    #optimizer = optim.SGD(agent.module.parameters(), lr = lr, momentum = MOMENTUM)
                    scheduler = None
                    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.85)
                    trainer.opt = optimizer
                    trainer.scheduler = scheduler
                else:
                    agent.encode_inputs = False

        if MAXMIN_NORMALIZATION: 
            print(get_max_min_path(LIB_TYPE, ENV_TYPE))
            mx, mn = retrieve_max_min([obs_size],[obs_size],LIB_TYPE, ENV_TYPE, norm_dir = 'norm')
            print("CURRENT MX: %s \n MN: %s \n" % (mx, mn))
            new_mx, new_mn = mx, mn #copy them for updating
        
        ## RUN AGENT / TRAINING
        if (not PRETRAINED) or (RUN_ANYWAYS):
            ## set up listener for user input
            lock = threading.Lock()
            console_args = (tmp_env, agent, lock, LIB_TYPE, ENV_TYPE, agent.encoder)
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
                            if MAXMIN_NORMALIZATION:
                                observation = agent.transform_observation(observation)
                                observation = normalize_max_min(observation.detach().cpu().numpy(), mx, mn)
                                #print("Observation:", observation)
                            action = agent.step(observation).cpu().numpy()
                            #print("Observation: ", observation)
                            #action = agent(timestep)
                            #print("Action: ", action)
                            agent.store_reward(reward)
                            timestep = env.step(action)
                            #print("Reward: %s" % (timestep.reward))
                    elif LIB_TYPE == 'gym':
                        env.reset()
                        observation, reward, done, info = env.step(env.action_space.sample())
                        while not done and step < MAX_TIMESTEPS:
                            if MAXMIN_NORMALIZATION:
                                observation = agent.transform_observation(observation)
                                observation = normalize_max_min(observation.detach().cpu().numpy(), mx, mn)
                            action = agent.step(observation).cpu().numpy()
                            env.render()
                            observation, reward, done, info = env.step(action)
                            agent.store_reward(reward)
                    elif LIB_TYPE == 'control':
                        env.reset()
                        while not env.episode_is_done() and step < MAX_TIMESTEPS:
                            obs = env.get_state()
                            action = agent.step(obs).cpu().numpy()
                            env.step(action)
                            reward = env.get_reward()
                            agent.store_reward(reward)

                    if MAXMIN_NORMALIZATION:
                        new_mx, new_mn = update_max_min(observation, new_mx, new_mn)
                    agent.terminate_episode() #oops forgot this lmao
                # Update step
                if not PRETRAINED:
                    trainer.step()
                    print("Agent Net Loss: ", agent.net_loss_history[i])
                if TRAIN_AUTOENCODER == True:
                    autoencoder_trainer.step()
                    autoencoder_trainer.plot_loss_histories()
                if i % 50 == 0: #periodic tasks
                    print("Performing Periodic Tasks")
                    if MAXMIN_NORMALIZATION: 
                        print("(stored) max: %s\n min: %s\n"%(new_mx, new_mn))
                        store_max_min(new_mx, new_mn, LIB_TYPE, ENV_TYPE, norm_dir = 'norm') 
                            

                agent.reset_histories()
                if AGENT_TYPE == 'policy' and not PRETRAINED:
                    print("Agent Net Action loss: ", agent.value_loss_history[i])
                    print("Agent Net Value loss: ", agent.action_loss_history[i])
                print("Agent Net Reward: ", agent.net_reward_history[i])
                #i += EPISODES_BEFORE_TRAINING 
                if DISPLAY_HISTORY is True:
                    if LIB_TYPE == 'control' and TRAIN_AUTOENCODER:
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
                    plt.ylim(0, max(agent.net_reward_history) + max(agent.net_reward_history) / 2)
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
                    plt.subplot(2, 1, 2)
                    plt.ylabel("Net \n Loss")
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
