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
    def __init__(self, friction = 0.1, initial_state = None, ts = 0.0001, 
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

        self.friction = friction
        
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
        state += (self.dx(state, action) * self.ts)
        #state += (self.dx(state, action))
        #state[0] = state[0] % np.pi
        self.steps += self.ts
        self.state_history.append(state.copy())
        self.state = state
         
    def dx(self, x, u = None, r = None, *args) -> np.ndarray:
    #def dx(self, x, u, *args) -> np.ndarray:
        '''x' = dx * x + du * u'''
        if u is None:
            u = 0.0
        #dx = self.d_dx(x, u, r) 
        #dx = np.array([x[1], -4*np.sin(x[0]) - 0.01*x[1]])
        #input("Self.friction: %s" % (self.friction))
        dx = np.array([x[1], 1*np.cos(x[0]) - self.friction*x[1]])
        #dx = np.array([x[1], -4*np.sin(x[0])])
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
        #dx = np.array([[0, x[1]], [4*np.cos(x[0]), -0.1*x[1]]])
        #dx = np.array([[0, x[1]], [4*np.cos(x[0]), -0.1*x[1]]])
        dx = np.array([[0, 1], [-1*np.sin(x[0]), -self.friction]])
        #dx = np.array([[0, 1], [-4*np.sin(x[0]), 0]])
        #dx = np.array([x[1], 4*np.sin(x[0]) - 0.001*x[1]]) #additional friction term
        return dx

    def d_du(self, x, u, r=None, *args):
        #return np.array([0, 1]) * u
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
        return 1

    def get_action_constraints(self):
        return [np.ndarray([-2]), np.ndarray([2])]

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
        plt.ylabel("Velocity")
        plt.xlabel("Position")
        plt.title("Nonlinear Pendulum Phase Plot")
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'go')
        if hasattr(self, 'target_point'):
            target = self.target_point
            plt.plot(target[0], target[0], 'b^')
        plt.draw()
        plt.pause(0.01)

    def get_reward(self):
        '''Reward = -Cost and vise-versa'''
        return 0



class CartpoleEnvironment(ControlEnvironment):
    def __init__(self, mc = 1, mp = 0.5, L = 1, g=9.8,
            simplified_derivatives = False, 
            initial_state = None, ts = 0.0001, 
            interval = 4.00, 
            noisy_init = False, 
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc = mc
        self.mp = mp
        self.L = L
        self.g = g
        self.simplified = simplified_derivatives

        self.noisy_init = noisy_init
        self.ts = ts
        self.interval = interval 
        
        if initial_state is not None:
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
        state += (self.dx(state, action) * self.ts)
        #state += (self.dx(state, action))
        #state[0] = state[0] % np.pi
        self.steps += self.ts
        self.state_history.append(state.copy())
        self.state = state.copy()
         
    def dx(self, x, u = None, r = None, *args) -> np.ndarray:
        '''x' = dx * x + du * u
        This is really ugly. And I feel bad for writing it down. '''
        if u is None:
            u = 0.0
        x_denom = (self.mc + self.mp * (np.sin(x[2]))**2)
        theta_denom = self.L * (self.mc + self.mp * (np.sin(x[2]))**2)
        ddx = self.mp * np.sin(x[2]) * (self.L * (x[3])**2 - self.g*np.cos(x[2]))
        ddtheta = -self.mp * self.L * (x[3])**2 * np.sin(x[2])*np.cos(x[2])
        ddtheta += (self.mc + self.mp) * self.g * np.sin(x[2])
        ux = u 
        utheta = -u*np.cos(x[2]) 
        dx = np.array([x[1], (ddx + ux) / x_denom, 
            x[3], (ddtheta + utheta) / theta_denom])
        return dx

    def d_dx(self, x, u=None, r=None, *args):
        '''This is really ugly. And I feel bad for writing it down. '''
        if self.simplified:
            #raise Exception('REEEEEEEEEEEEEEEE linearize WHERE')
            return np.array([[0, 1, 0, 0],
                [0, 0, -self.mp * self.g / self.mc, 0],
                [0, 0, 0, 1],
                [0, 0, (self.mc + self.mp)*self.g / (self.L * self.mc), 0]])
        sx = np.sin(x[3])
        cx = np.cos(x[3])
        mp_denom = self.mp * sx ** 2
        mp_denom_dx = self.mp * 2 * sx * cx 
        mp_denom_2 = (self.mc + mp_denom) ** 2
        mpL = self.L * self.mp
        #d2x3 = 0
        d2x3 = ((mpL*cx*x[3]**2 * mp_denom) - (mpL*cx*x[3]**2 * mp_denom_dx))
        d2x3 += (-(self.mp*(cx**2 - sx**2) * mp_denom) - (-2*self.mp*cx*sx * mp_denom_dx)) #TODO: verify signs, validity of mp_denom and derivatives
        d2x3 = d2x3 / mp_denom_2
        d2x4 = mpL * sx * 2 * x[3] / (self.mc + mp_denom)
            
        mp_denom = self.mp * sx ** 2
        mp_denom_dx = self.mp * 2 * sx * cx 
        mp_denom_2 = (self.L*(self.mc + mp_denom)) ** 2
        #d4x3 = 0
        #d4x4 = 0
        d4x3 = (((-mpL * x[3]**2 * (cx**2 - sx**2)) * mp_denom_dx) - ((-mpL*x[3]**2*sx*cx) * mp_denom_dx)) 
        d4x3 += (-(self.mc + self.mp)*self.g*cx)*(self.L*(self.mc + mp_denom)) - (((self.mc + self.mp)*self.g*cx) * mp_denom_dx)  #TODO: verify mp_denom is valid
        d4x3 = d4x3 / mp_denom_2
        d4x4 = -mpL * sx * cx/(self.L * (self.mc + mp_denom))
        dx = np.array([[0, 1, 0, 0],
                [0, 0, d2x3, d2x4],
                [0, 0, 0, 1],
                [0, 0, d4x3, d4x4]])
        #return np.zeros([4, 4])
        return dx

    def d_du(self, x, u, r=None, *args):
        '''This is really ugly. And I feel bad for writing it down. '''
        if self.simplified: #simplified derivative ignores higher order 
            #raise Exception('REEEEEEEEEEEEEEEE linearize WHERE')
            return np.array([0, 
                1/(self.mc), 
                0, 
                -1/(self.L*(self.mc))])
        #return np.zeros([4,])
        return np.array([0, 1/(self.mc + self.mp * np.sin(x[2])**2), 0, 
            -np.cos(x[2])/(self.L*(self.mc + self.mp * np.sin(x[2])**2))])

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
        return 4

    def get_action_size(self):
        return 1

    def get_action_constraints(self):
        return [np.ndarray([-2]), np.ndarray([2])]

    def generate_plots(self):
        self.generate_theta_phase_plot()
        self.generate_cart_phase_plot()

    def generate_theta_phase_plot(self, history = None):
        if history is None:
            if not hasattr(self, 'theta_fig'):
                self.theta_fig = plt.figure()
            fig = self.theta_fig 
            history = self.state_history
        else:
            if not hasattr(self, 'secondary_theta_fig'):
                self.secondary_theta_fig = plt.figure()
            fig = self.secondary_theta_fig
        x = [s[2] for s in history]
        y = [s[3] for s in history]
        plt.figure(fig.number)
        plt.plot(x,y, label='parametric curve')
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'g')
        plt.title("Cartpole Theta Phase Plot")
        plt.xlabel("Theta")
        plt.ylabel("dTheta/dt")
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'go')
        if hasattr(self, 'target_point'):
            target = self.target_point
            plt.plot(target[2], target[3], 'b^')
        plt.draw()
        plt.pause(0.01)
    
    def generate_cart_phase_plot(self, history = None):
        if history is None:
            if not hasattr(self, 'cart_fig'):
                self.cart_fig = plt.figure()
            fig = self.cart_fig 
            history = self.state_history
        else:
            if not hasattr(self, 'secondary_cart_fig'):
                self.secondary_cart_fig = plt.figure()
            fig = self.secondary_cart_fig
        plt.figure(fig.number)
        x = [s[0] for s in history]
        y = [s[1] for s in history]
        plt.plot(x,y, label='parametric curve')
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'g')
        plt.title("Cartpole Position Phase Plot")
        plt.xlabel("Position")
        plt.ylabel("dX/dt")
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'go')
        if hasattr(self, 'target_point'):
            target = self.target_point
            plt.plot(target[0], target[1], 'b^')
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
    if env_type == 'cartpole':
        env = CartpoleEnvironment(*args, **kwargs)
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
    target = np.array([0, 0])
    horizon = 30
    friction = 0.0
    env = retrieve_control_environment('inverted', 
            friction = friction,  
            noisy_init = noisy_init, 
            interval = horizon, ts = 0.001,
            mode = 'point', 
            target = target)
    env.reset()
    env.state = np.array([0.0001, 0])
    gamma = 0.7
    wn = 20
    while not env.episode_is_done():
        print("STEP ", env.steps)
        x = env.state
        x_ = target - x
        print("State: ", x)
        print("Target: %s \n Error: %s"%(target, x_))
        w = np.array([4*np.sin(x[0]) - wn**2 * x[0] + friction*x[1] - 2*gamma*wn*x[1]])
        dw = np.array([[0,0],[4*np.cos(x[0]) - wn**2, friction - 2*gamma*wn]])
        #dw = np.array([4*np.cos(x[0]) - wn**2, 0.1 - 2*gamma*wn])
        dw = np.zeros(1)
        #dw = np.ones(1)
        #print("dW: ", dw)
        #print("U: ", target - w)
        #env.step(target + dw)
        #env.step(target + w)
        env.step(dw)
        #env.step(w)
    env.generate_plots()
    input()



