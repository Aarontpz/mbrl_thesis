import abc

import numpy as np
import scipy.integrate as spi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math import copysign

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
    def __init__(self, mode = 'point', target = None, 
            error_func = lambda x, t:np.linalg.norm(x - t), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        if target is None:
            self.set_target(np.zeros())
        self.set_target(target)
        self.state_history = []
        self.control_history = []
        self.error_func = error_func

    def dx(self, x, u, *args) -> np.ndarray:
        self.control_history.append(u)

    @abc.abstractmethod
    def d_dx(self, x, u=None, r=None, *args):
        pass
    @abc.abstractmethod
    def d_du(self, x, u=None, r=None, *args):
        pass
        
    def step(self, action):
        state = self.state.copy()
        target = self.get_target()
        #state += (self.dx(state, action, target) * self.interval*self.ts)
        state += (self.dx(state, action) * self.ts)
        #state += (self.dx(state, action))
        #state[0] = state[0] % np.pi
        self.steps += self.ts
        self.state_history.append(state.copy())
        self.state = state

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
    
    def reset(self):
        self.state = self.get_initial_state(noise = self.noisy_init)
        self.state_history = []
        self.control_history = []
        self.steps = 0
    
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
        plt.ylabel("dTheta/dt")
        plt.xlabel("Theta")
        plt.title("Inverted Pendulum Phase Plot")
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'go')
        if hasattr(self, 'target_point'):
            target = self.target_point
            plt.plot(target[0], target[1], 'b^')
        plt.draw()
        plt.pause(0.01)
    
    
    def generate_control_plot(self, history = None):
        if history is None:
            if not hasattr(self, 'control_fig'):
                self.control_fig = plt.figure()
            fig = self.control_fig 
            history = self.state_history
        else:
            if not hasattr(self, 'secondary_control_fig'):
                self.secondary_control_fig = plt.figure()
            fig = self.secondary_control_fig
        y = self.control_history 
        length = len(self.control_history)
        x = [i * self.ts for i in range(length)] 
        plt.figure(fig.number)
        plt.plot(x,y, label='parametric curve')
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'g')
        plt.title("%s Control History" % (self.get_environment_name()))
        plt.xlabel("Time [s]")
        plt.ylabel("Control [N]")
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'go')
        plt.draw()
        plt.pause(0.01)
    
    def generate_error_plot(self, history = None):
        if history is None:
            if not hasattr(self, 'error_fig'):
                self.error_fig = plt.figure()
            fig = self.error_fig 
            history = self.state_history
        else:
            if not hasattr(self, 'secondary_error_fig'):
                self.secondary_error_fig = plt.figure()
            fig = self.secondary_error_fig
        target = self.target_point
        y = [self.error_func(s, target) for s in self.state_history]
        length = len(self.state_history)
        x = [i * self.ts for i in range(length)] 
        plt.figure(fig.number)
        plt.plot(x,y, label='parametric curve')
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'g')
        plt.title("%s Error History" % (self.get_environment_name()))
        plt.xlabel("Time [s]")
        plt.ylabel("Error")
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'go')
        plt.draw()
        plt.pause(0.01)
    
    def generate_plots(self):
        self.generate_control_plot()
        self.generate_error_plot()
        #self.generate_eigenvalues_history_plot()

    @abc.abstractmethod
    def get_environment_name(self):
        return 'ControlEnvironment'

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
    
    @abc.abstractmethod
    def dx(self, x, u = None, *args) -> np.ndarray:
        '''x' = dx * x + bu * u; u = 0 vector though'''
        dx = np.array([10*(x[1] - x[0]), 
            x[0]*(28-x[2])-x[1], x[0]*x[1]-(8/3)*x[2]])
        super().dx(x, u)
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

    def get_reward(self):
        return 0

    def get_environment_name(self):
        return 'Rossler'

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
        super().dx(x, u)
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

    def get_reward(self):
        '''Reward = -Cost and vise-versa'''
        return 0
    
    def get_environment_name(self):
        return 'Inverted Pendulum'

    def generate_plots(self):
        super().generate_plots()
        self.generate_state_history_plot()

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
        self.control_history = []
        
        self.steps = 0
    
    def reset(self):
        self.state = self.get_initial_state(noise = self.noisy_init)
        self.state_history = []
        self.control_history = []
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
        super().dx(x, u)
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
        super().generate_plots()

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
    
    def get_environment_name(self):
        return 'ControlEnvironment'

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
    TEST_INVERTED_PENDULUM = True
    TEST_CARTPOLE = True

    if TEST_CARTPOLE:
        horizon = 15
        mc = 1
        mp = 0.1
        L = 0.5
        g = 9.8
        dt = 1e-2
        target = np.array([0.0, 0, 0.0, 0])
        x0 = np.array([-1, 0, 0.2, -0.1])
        simplified_derivatives = False
        env = retrieve_control_environment('cartpole', 
                mc, mp, L, g,
                simplified_derivatives,
                x0, 
                interval = horizon, ts = dt,
                mode = 'point', 
                target = target)
        env.reset()
        env.state = x0.copy()
        while not env.episode_is_done():
            print("STEP ", env.steps)
            x = env.state
            x_ = x - np.array([x[0], x[1], target[2], x[3]])
            #x_ = x - target
            print("State: ", x)
            print("Target: %s \n Error: %s"%(target, x_))

            ##Linearizing Feedback Controls

            ## Sliding Mode Controls
            x_denom = (mc + mp * (np.sin(x[2]))**2)
            ddx = mp * np.sin(x[2]) * (L * (x[3])**2 - g*np.cos(x[2]))
            ddtheta = -mp * L * (x[3])**2 * np.sin(x[2])*np.cos(x[2])
            ddtheta += (mc + mp) * g * np.sin(x[2])
            dx = x[1] + (ddx) / x_denom + x[3] + (ddtheta) / (L * x_denom)
            sigma = np.array([1, 1, 1, 1]) #sliding surface definition
            ucoeff = 2
            umin = -10
            umax = 10
            ucomp = (1 - np.cos(x[2]) / L) * (1/x_denom)
            #ucomp = 1
            u = lambda sigma, x: ucoeff * (1/ucomp) * -(np.abs(dx)) * np.sign(np.dot(sigma.T, x))
            print("Sliding Surface: ", np.dot(sigma.T, x_)) 
            control = np.clip(u(sigma, x_), umin, umax)
            #control = u(sigma, x_)
            print("Control: ", control)
            env.step(control)
            #env.step(np.zeros(1))
        env.generate_plots()
        input()

    if TEST_INVERTED_PENDULUM:
        noisy_init = True
        target = np.array([np.pi, 0])
        horizon = 5
        friction = 0.0
        env = retrieve_control_environment('inverted', 
                friction = friction,  
                noisy_init = noisy_init, 
                interval = horizon, ts = 0.001,
                mode = 'point', 
                target = target)
        env.reset()
        env.state = np.array([0.0, 0.0])
        gamma = 0.7
        wn = 20
        while not env.episode_is_done():
            print("STEP ", env.steps)
            x = env.state
            x_ = x - target
            print("State: ", x)
            print("Target: %s \n Error: %s"%(target, x_))

            ##Linearizing Feedback Controls
            w = np.array([4*np.sin(x[0]) - wn**2 * x[0] + friction*x[1] - 2*gamma*wn*x[1]])
            dw = np.array([[0,0],[4*np.cos(x[0]) - wn**2, friction - 2*gamma*wn]])
            #dw = np.array([4*np.cos(x[0]) - wn**2, 0.1 - 2*gamma*wn])
            #dw = np.zeros(1)
            #dw = np.ones(1)
            #print("dW: ", dw)
            #print("U: ", target - w)
            #env.step(target + dw)
            #env.step(target + w)
            #env.step(dw)
            #env.step(w)

            ## Sliding Mode Controls
            sigma = np.array([1, 1]) #sliding surface definition
            umax = 2
            u = lambda sigma, x: umax * -(np.abs(-x[1] - np.cos(x[0]) + friction * x[1])) * np.sign(np.dot(sigma.T,  x))
            print("Sliding Surface: ", np.dot(sigma.T, x)) 
            #control = u(sigma, x)
            control = u(sigma, x_)
            print("Control: ", control)
            #env.step(u(sigma, x_))
            env.step(control)
        env.generate_plots()
        input()



