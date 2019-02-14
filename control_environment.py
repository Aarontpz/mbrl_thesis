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
    def __init__(self, mode = 'point', target = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        if target is None:
            self.set_target(np.zeros())
        self.set_target(target)

    @abc.abstractmethod
    def dx(self, x, u, *args) -> np.ndarray:
        pass
    @abc.abstractmethod
    def d_dx(self, x, u=None, r=None, *args):
        pass
    @abc.abstractmethod
    def d_du(self, x, u=None, r=None, *args):
        pass

    def set_control_mode(self, m):
        '''@args: 
            m: 'traj' or 'point' for the control objective'''
        self.mode = m

    def set_target(self, target):
        if self.mode == 'point':
            self.set_target_point(target)
        elif self.mode == 'traj':
            self.set_target_trajectory(target)

    def set_target_trajectory(self, target):
        self.target_traj = target

    def set_target_point(self, target):
        self.target_point = target
    
    def get_target(self) -> np.ndarray:
        if self.mode == 'point':
            return self.target_point
        elif self.mode == 'traj':
            return self.target_traj

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

class InvertedPendulumEnvironment(ControlEnvironment):
    def __init__(self, initial_state = None, ts = 0.0001, 
            interval = 4.00, 
            noisy_init = False, 
            *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        '''Utilize Euler Integration to update system "discretely"'''
        state = self.state.copy()
        target = self.get_target()
        #state += (self.dx(state, action, target) * self.interval*self.ts)
        state += (self.dx(state, action) * self.interval*self.ts)
        #state[0] = state[0] % np.pi
        self.steps += self.interval * self.ts
        self.state_history.append(state.copy())
        self.state = state
         
    def dx(self, x, u = None, r = None, *args) -> np.ndarray:
    #def dx(self, x, u, *args) -> np.ndarray:
        '''x' = dx * x + du * u'''
        #if u is None:
        #    u = 0.0
        #dx = np.array([x[1], 
        #    -4*np.sin(x[0])])
        #du = np.array([0, 1]) * u
        #print("Dx: %s \n Du: %s" % (dx, du))
        ##if r is not None:
        ##    return (dx - u) + np.array([0, 1])*r
        #return dx + du
        if u is None:
            u = 0.0
        dx = self.d_dx(x, u, r) 
        #dx = np.array([x[1], 4*np.cos(x[0])])
        if r is not None:
            #print("Ref: ", r) 
            #print("B*u: ", np.array([0, 1]) * u)
            #print("B*u - dx: ", np.array([0, 1]) * u - dx)
            #print("B*r: ", np.array([0, 1]) * r)
            #d_dx = dx + np.array([0, 1]) * (u)
            #d_dx = dx + np.array([0, 1]) * (u) - np.array([0, 1])*dx[1]
            d_dx = dx + np.array([0, 1]) * (u)
            return d_dx
        du = np.array([0, 1]) * u
        #print("Dx: %s \n Du: %s" % (dx, du))
        return dx + du

    def d_dx(self, x, u=None, r=None, *args):
        dx = np.array([x[1], -4*np.sin(x[0])])
        dx = np.array([x[1], 4*np.sin(x[0]) - 0.001*x[1]]) #additional friction term
        return dx
    def d_du(self, x, u, r=None, *args):
        return np.array([0, 1])

    def get_initial_state(self, noise = False):
        theta = 0 #OBJECTIVE is 180 degrees / pi / 2
        theta_p = 0 
        s = np.array([theta, theta_p], dtype = np.float32)
        if noise:
            s += np.random.uniform(low=0, high=0.1, size=2)
        return s

    def episode_is_done(self):
        if self.steps >= self.interval:
            return True
        return False
    
    def get_observation_size(self):
        return 2

    def get_action_size(self):
        return 2

    def get_action_constraints(self):
        return [np.ndarray([-2, -2]), np.ndarray([2, 2])]

    def generate_plots(self):
        self.generate_state_history_plot()

    def generate_state_history_plot(self, history = None):
        if history is None:
            if not hasattr(self, 'state_fig'):
                self.state_fig = plt.figure()
            history = self.state_history
            fig = self.state_fig 
        else:
            if not hasattr(self, 'secondary_state_fig'):
                self.secondary_state_fig = plt.figure()
            fig = self.secondary_state_fig
        x = [s[0] for s in history]
        y = [s[1] for s in history]
        plt.plot(x,y, label='parametric curve')
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'g')
        plt.draw()
        plt.pause(0.01)

    def get_reward(self):
        '''Reward = -Cost and vise-versa'''
        return 0

def retrieve_control_environment(env_type = 'rossler', *args, **kwargs):
    env = None
    if env_type == 'rossler':
        env = RosslerEnvironment(*args, **kwargs)
    if env_type == 'inverted':
        env = InvertedPendulumEnvironment(*args, **kwargs)

    if env is None:
        raise Exception("%s was not a recognized control environment type..." % (env_type))
    return env

if __name__ == '__main__':
    #env = retrieve_control_environment('rossler')
    #env.reset()
    #while not env.episode_is_done():
    #    env.step(None)
    #env.generate_plots()
    noisy_init = True
    env = retrieve_control_environment('inverted', 
            noisy_init = noisy_init, 
            interval = 8.0, ts = 0.001,
            mode = 'point', 
            target = np.array([np.pi/4, 0], dtype = np.float32))
    env.reset()
    gamma = 0.7
    wn = 1
    while not env.episode_is_done():
        print("STEP ", env.steps)
        x = env.state
        target = env.get_target()
        x_ = target - x
        print("State: ", x)
        print("Target: %s \n Error: %s"%(target, x_))
        #w = np.array([wn**2 + 4 * np.cos(x[0]), 2*wn*gamma]) #linearizing feedback
        #w = np.array([(4 * np.sin(x[0]) - wn**2 * x[0]) + (-1*2*wn*gamma*x[1])]) #linearizing feedback for OPEN-LOOP stability (reeeeee)
        #dw = np.array([-4*np.cos(x[0]) - wn**2 * x[0] - 2*wn*gamma*x[1]]) 
        dw = np.zeros(1)
        print("W: ", dw)
        #print("U: ", target - w)
        #env.step(target - w)
        env.step(dw)
    env.generate_plots()
    input()



