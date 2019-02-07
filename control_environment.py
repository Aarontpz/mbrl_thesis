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
    def step(self, action) -> np.ndarray: 
        pass

    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_reward(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def get_observation_size(self) -> int:
        pass
    
    @abc.abstractmethod
    def get_action_size(self) -> int:
        pass
    
    @abc.abstractmethod
    def get_action_constraints(self) -> int:
        pass

    @abc.abstractmethod
    def generate_plots(self):
        pass

class ModelledEnvironment(Environment):
    @abc.abstractmethod
    def dx(self, x, *args) -> np.ndarray:
        pass

class ControlEnvironment(ModelledEnvironment):
    @abc.abstractmethod
    def dx(self, x, u, *args) -> np.ndarray:
        pass

class RosslerEnvironment(ModelledEnvironment):
    '''Actions don't affect this system, it is purely to test
    and compare approaches to modelling nonlinear systems online.'''
    def __init__(self, ts = 0.00001, noisy_init = False, *args, **kwargs):
        self.noisy_init = noisy_init
        self.ts = ts
        
        self.state_history = []

    
    def reset(self):
        self.state = self.get_initial_state(noise = self.noisy_init)

        self.state_history = []
        #TODO: reset figures?

    def step(self, action):
        #TODO: expirement with just...doing this discretely? (additively)
        interval = 36.87
        timesteps = np.linspace(0, interval, 1/self.ts)
        #if dx had an action, we would pass that into the spi.odeint args
        #and/or call the function generator between passes, and make
        #interval = sample frequency instead of transient-period
        
        #traj = spi.RK23() #Runge-Kutta method
        #traj = spi.odeint(self.dx, self.state, timesteps, args = (self,))
        #self.state_history = traj

        traj = []
        state = self.state.copy()
        for t in timesteps:
            state += (self.dx(state) * interval*self.ts)
            traj.append(state.copy())
            #print('STATE', state)
        #self.state = state
        traj = np.array(traj)
        self.state_history = traj
         
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

    def get_action_size(self):
        return 1

    def get_action_constraints(self):
        return None

    def generate_plots(self):
        self.generate_state_history_plot()

    def generate_state_history_plot(self):
        x = [s[0] for s in self.state_history]
        y = [s[1] for s in self.state_history]
        z = [s[2] for s in self.state_history]
        print('STate history: ', self.state_history)
        #print("X: ", x)
        if not hasattr(self, 'state_fig'):
            self.state_fig = plt.figure()
        #ax = self.state_fig.gca(projection='3d')
        ax = self.state_fig.add_subplot(111, projection='3d')
        ax.plot(x,y,z, label='parametric curve')
        plt.show()
        #plt.pause(0.01)

        

def retrieve_control_environment(env_type = 'rossler'):
    env = None
    if env_type == 'rossler':
        env = RosslerEnvironment()

    if env is None:
        raise Exception("%s was not a recognized control environment type..." % (env_type))
    return env

if __name__ == '__main__':
    env = retrieve_control_environment('rossler')
    env.reset()
    env.step(None)
    env.generate_plots()




