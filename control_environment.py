import abc

import numpy as np
import scipy.integrate as spi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Environment:
    __metaclass__ = abc.ABCMeta
    def __init__(self,):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step(self, action) -> np.ndarray: 
        pass

    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_reward(self) -> int:
        pass

    def episode_is_done(self) -> bool:
        return False

    @abc.abstractmethod
    def get_episode_length(self):
        return 1

    @abc.abstractmethod
    def get_observation_size(self) -> int:
        pass
    
    @abc.abstractmethod
    def get_action_size(self) -> int:
        pass
    
    @abc.abstractmethod
    def get_action_constraints(self) -> [np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def generate_plots(self):
        pass

class ModelledEnvironment(Environment):
    @abc.abstractmethod
    def dx(self, x, *args) -> np.ndarray:
        pass
    
    def get_state(self) -> np.ndarray:
        return self.state

class ControlEnvironment(ModelledEnvironment):
    def __init__(self, mode = 'point', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode

    @abc.abstractmethod
    def dx(self, x, u, *args) -> np.ndarray:
        pass

    def set_control_mode(self, m):
        '''@args: 
            m: 'traj' or 'point' for the control objective'''
        self.mode = m

    @abc.abstractmethod
    def set_target_trajectory():
        pass

    @abc.abstractmethod
    def set_target_point():
        pass

class RosslerEnvironment(ModelledEnvironment):
    '''Actions don't affect this system, it is purely to test
    and compare approaches to modelling nonlinear systems online.  '''
    #NOTE: This may be unfair, since Rossler is chaotic and thus ANY
    #deviations have profound long-term impacts on the trajectories
    def __init__(self, initial_state = None, ts = 0.0001, 
            interval = 36.87, 
            noisy_init = True, 
            *args, **kwargs):
        self.noisy_init = noisy_init
        self.ts = ts
        self.interval = interval 
        if initial_state:
            self.state = initial_state
        else:
            self.state = self.get_initial_state(noise = self.noisy_init)
        self.state_history = []
        
        self.steps = 0
    
    def reset(self):
        self.state = self.get_initial_state(noise = self.noisy_init)
        self.state_history = []
        self.steps = 0

    def step(self, action):
        #if dx had an action, we would pass that into the spi.odeint args
        #and/or call the function generator between passes, and make
        #interval = sample frequency instead of transient-period
        
        #timesteps = np.linspace(0, self.interval, 1/self.ts)
        #traj = spi.RK23() #Runge-Kutta method
        #traj = spi.odeint(self.dx, self.state, timesteps, args = (self,))
        #self.state_history = traj

        state = self.state.copy()
        state += (self.dx(state) * self.interval*self.ts)
        self.steps += self.interval * self.ts
        self.state_history.append(state.copy())
        self.state = state
        #print('STATE', state)
        #print('STEPS: ', self.steps)
         
    @abc.abstractmethod
    def dx(self, x, *args) -> np.ndarray:
        '''x' = dx * x + bu * u; u = 0 vector though'''
        dx = np.array([10*(x[1] - x[0]), 
            x[0]*(28-x[2])-x[1], x[0]*x[1]-(8/3)*x[2]])
        return dx 

    def get_initial_state(self, noise = False):
        x = 2.9
        y = -1.3
        z = 25
        s = np.array([x, y, z])
        if noise:
            s += np.random.uniform(low=0, high=0.5, size=3)
        return s

    def episode_is_done(self):
        if self.steps >= self.interval:
            return True
        return False
    
    def get_observation_size(self):
        return 3

    def get_action_size(self):
        return 1

    def get_action_constraints(self):
        return None

    def generate_plots(self):
        self.generate_state_history_plot()

    def generate_state_history_plot(self, history = None):
        if not hasattr(self, 'state_fig'):
            self.state_fig = plt.figure()
        if history is None:
            history = self.state_history
            fig = self.state_fig 
        else:
            if not hasattr(self, 'secondary_state_fig'):
                self.secondary_state_fig = plt.figure()
            fig = self.secondary_state_fig
        x = [s[0] for s in history]
        y = [s[1] for s in history]
        z = [s[2] for s in history]
        #print('State history: ', self.state_history)
        #print("X: ", x)
        #ax = self.state_fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x,y,z, label='parametric curve')
        plt.draw()
        plt.pause(0.01)

    def get_reward(self):
        return 0

def retrieve_control_environment(env_type = 'rossler', *args, **kwargs):
    env = None
    if env_type == 'rossler':
        env = RosslerEnvironment(*args, **kwargs)

    if env is None:
        raise Exception("%s was not a recognized control environment type..." % (env_type))
    return env

if __name__ == '__main__':
    env = retrieve_control_environment('rossler')
    env.reset()
    while not env.episode_is_done():
        env.step(None)
    env.generate_plots()




