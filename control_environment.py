#Author: Aaron Parisi
#3/11/19
import abc

import numpy as np
import scipy.integrate as spi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math import copysign

from model import *

import csv

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
            error_func = lambda x, t:np.linalg.norm(x - t, ord = 1), *args, **kwargs):
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
        plt.figure(fig.number)
        plt.clf()
        plt.plot(x,y, label='parametric curve')
        plt.ylabel("dTheta/dt")
        plt.xlabel("Theta")
        plt.title("%s Phase Plot" % (self.get_environment_name()))
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
        if len(y[0].shape) >= 2: #reduce from array
            #if type(y[0]) == float:
            #y = [c for c in y]
            #else:
            y = [c[0] for c in y]
        length = len(y)
        x = [i * self.ts for i in range(length)] 
        #x = np.linspace(0, length * self.ts, self.ts)
        plt.figure(fig.number)
        plt.clf()
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
        self.error_history = y #this is gross...but for csv sake
        length = len(self.state_history)
        x = [i * self.ts for i in range(length)] 
        plt.figure(fig.number)
        plt.clf()
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

    def step(self, action):
        '''Utilize Euler Integration to update system "discretely"'''
        state = self.state.copy()
        target = self.get_target()
        #state += (self.dx(state, action, target) * self.interval*self.ts)
        state += (self.dx(state, action) * self.ts)
        if state[0] > 0:
            state[0] %= 2*np.pi
        elif state[0] < 0:
            state[0] %= -2*np.pi
        #state += (self.dx(state, action))
        #state[0] = state[0] % np.pi
        self.steps += self.ts
        self.state_history.append(state.copy())
        self.state = state.copy()

    def dx(self, x, u = None, r = None, *args) -> np.ndarray:
    #def dx(self, x, u, *args) -> np.ndarray:
        '''x' = dx * x + du * u'''
        if u is None:
            u = np.array([0.0])
        if len(u.shape) > 1:
            u = u[0]
        #dx = self.d_dx(x, u, r) 
        #dx = np.array([x[1], -4*np.sin(x[0]) - 0.01*x[1]])
        #input("Self.friction: %s" % (self.friction))
        dx = np.array([x[1], 1*np.sin(x[0]) - self.friction*x[1]])
        #dx = np.array([x[1], -4*np.sin(x[0])])
        if r is not None:
            #print("Ref: ", r) 
            #print("B*u: ", np.array([0, 1]) * u)
            #print("B*u - dx: ", np.array([0, 1]) * u - dx)
            #print("B*r: ", np.array([0, 1]) * r)
            #d_dx = dx + np.array([0, 1]) * (u)
            #d_dx = dx + np.array([0, 1]) * (u) - np.array([0, 1])*dx[1]
            d_dx = dx + np.array([0, -1]) * (u)
            return d_dx
        du = np.array([0, 1]) * -u
        #print("Dx: %s \n Du: %s" % (dx, du))
        super().dx(x, u)
        return dx + du

    def d_dx(self, x, u=None, r=None, *args):
        #dx = np.array([[0, x[1]], [4*np.cos(x[0]), -0.1*x[1]]])
        #dx = np.array([[0, x[1]], [4*np.cos(x[0]), -0.1*x[1]]])
        print("INP: ", x)
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
            s[0] += np.random.uniform(low=0, high=np.pi, size=1)
            s[1] += np.random.uniform(low=-0.5, high=0.5, size=1)
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
        return [np.array([-2,]), np.array([2,])]

    def get_reward(self):
        '''Reward = -Cost and vise-versa'''
        if hasattr(self, 'error_func'):
            return -self.error_func(self.state, self.get_target())
        return 0
    
    def get_environment_name(self):
        return 'Inverted Pendulum'
    
    def generate_vector_field_plot(self, dx_model = None):
        if dx_model is not None:
            if not hasattr(self, 'secondary_vector_fig'):
                self.secondary_vector_fig = plt.figure()
            fig = self.secondary_vector_fig
        else:
            if not hasattr(self, 'vector_fig'):
                self.vector_fig = plt.figure()
            fig = self.vector_fig 
        x = np.arange(-1.5, 1.0*np.pi, 0.1)
        y = np.arange(-3.0, 3.0, 0.1)
        #xx, yy = np.meshgrid(x, y, sparse = True)
        z = []
        xx = []
        yy = []
        for i in range(x.size): #NOTE: are we...really doing this? YES
            for j in range(y.size): #REALLY. nested for loops instead of vect.
                xx.append(x[i])
                yy.append(y[j])
                if dx_model is not None:
                    xt = np.array([[x[i], y[j]]]).T
                    print("Xt: ", xt)
                    if issubclass(type(dx_model), LinearSystemModel):
                        A, B = dx_model.update(xt)
                        dx = np.dot(A, xt)
                        z.append(dx.copy())
                    elif issubclass(type(dx_model), GeneralSystemModel):
                        f, g = dx_model.update(xt)
                        print("f: ", f)
                        z.append((f.copy())) 
                else:
                    dx = self.dx(np.array([x[i], y[j]]))
                    print("Dx: ", dx)
                    print("Diag: ", np.diag(dx))
                    z.append(dx)
                    #z.append(np.diag(self.dx(np.array([x[i], y[j]]))))
                    #z.append(np.diag(dx))
                    #z.append(self.d_dx(np.array([x[i], y[j]])))
                    #z.append(np.array([[np.sin(x[i]), 0], [0, np.cos(y[j])]]))
        z = np.array(z)
        print("Z: ", z.shape)
        if dx_model is not None: #temporary measure, we have TOO MANY figures
            plt.figure(54)
        else:
            plt.figure(55)
        plt.clf()
        #if dx_model is None:
        #    plt.quiver(xx, yy, eig[0][:,0], eig[0][:,1])
        #else:
        plt.quiver(xx, yy, z[:,0], z[:,1])
        plt.title("%s Quiver Plot" % (self.get_environment_name()))
        plt.xlabel("Radians")
        plt.ylabel("Radians / s")
        if True:
            history = self.state_history
            x = [s[0] for s in history]
            y = [s[1] for s in history]
            est_history = []
            x_ = history[0]
            if True and dx_model is not None: #estimate state history using dx_model
                for i in range(len(x)): #we're hacky because T I M E
                    est_history.append(x_)
                    ut = self.control_history[i]
                    dt = dx_model.dt
                    #print("x_: ", x_)
                    #print("x_ shape: ", x_.shape)
                    #print("ut: ", ut)
                    #print("ut shape: ", ut.shape)
                    if len(x_.shape) > 1:
                        x_ = x_[:,0]
                    #forward = (dx_model.forward_predict(x_, ut, dt)).detach().cpu().numpy()
                    forward = (dx_model.forward(x_, ut, *dx_model.module(x_, ut))).detach().cpu().numpy()
                    #print("forward shape: ", forward.shape)
                    if len(x_.shape) < 2:
                        x_ = x_[..., np.newaxis]
                    x_ = x_ + dt*forward
                    #x_ = forward.copy()
                x_ = [s[0] for s in est_history]
                y_ = [s[1] for s in est_history]
                plt.figure(54)
                plt.plot(x_, y_, c = 'g')
                plt.figure(55)
                plt.plot(x_, y_, c = 'g')
                #input()
            plt.plot(x, y, c = 'b')
            plt.plot(x[0], y[0], 'ro')
            plt.plot(x[-1], y[-1], 'go')
            if hasattr(self, 'target_point'):
                target = self.target_point
                plt.plot(target[0], target[1], 'b^')
        plt.draw()
        plt.pause(0.01)
        

    def generate_plots(self):
        super().generate_plots()
        self.generate_state_history_plot()
        self.generate_vector_field_plot()

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
        if state[2] > 0:
            state[2] %= 2*np.pi
        elif state[2] < 0:
            state[2] %= -2*np.pi
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
        theta = np.pi #OBJECTIVE is 0 degrees
        theta_p = 0 
        s = np.array([0, 0, theta, theta_p], dtype = np.float32)
        if noise:
            s += np.random.uniform(low=-0.5, high=0.5, size=4)
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
        return [np.array([-10]), np.array([10])]

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
        plt.clf()
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
        plt.clf()
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

    def generate_theta_vector_field_plot(self, dx_model = None):
        if dx_model is not None:
            if not hasattr(self, 'secondary_vector_fig'):
                self.secondary_vector_fig = plt.figure()
            fig = self.secondary_vector_fig
        else:
            if not hasattr(self, 'vector_fig'):
                self.vector_fig = plt.figure()
            fig = self.vector_fig 
        t = np.arange(-2*np.pi, 2*np.pi, 0.1)
        t_dx = np.arange(-6, 6, 0.1)
        #xx, yy = np.meshgrid(x, y, sparse = True)
        z = []
        xx = []
        yy = []
        for i in range(t.size): #NOTE: are we...really doing this? YES
            for j in range(t_dx.size): #REALLY. nested for loops instead of vect.
                xx.append(t[i])
                yy.append(t_dx[j])
                if dx_model is not None:
                    dx = dx_model.update(np.array([t[i], t_dx[j]]))[0]
                    if len(dx.shape) > 1:
                        z.append(dx) #linear / Jacobian
                    else:
                        z.append(np.diag(dx)) #vector rep. transition
                else:
                    z.append(np.diag(self.dx(np.array([t[i], t_dx[j]]))))
                    #z.append(self.d_dx(np.array([x[i], y[j]])))
                    #z.append(np.array([[np.sin(x[i]), 0], [0, np.cos(y[j])]]))
                #z.append(np.diag(self.dx(np.array([0, 0, t[i], t_dx[j]]))))
                #z.append(self.d_dx(np.array([x[i], y[j]])))
                #z.append(np.array([[np.sin(x[i]), 0], [0, np.cos(y[j])]]))
        z = np.array(z)
        #print("Z: ", z.shape)
        eig = np.linalg.eig(np.array(z))
        #print("Eig: ", len(eig))
        #print("EIG[0]: ", eig[0][:,:])
        #print("Eig[0][0,0]:", eig[0][0,0])
        #print("Eig[0][0, 1]:", eig[0][0,1])
        if dx_model is not None: #temporary measure, we have TOO MANY figures
            plt.figure(54)
        else:
            plt.figure(55)
        plt.clf()
        #plt.figure(fig.number)
        plt.quiver(xx, yy, eig[0][:,2], eig[0][:,3])
        plt.title("%s Quiver Plot" % (self.get_environment_name()))
        plt.xlabel("Radians")
        plt.ylabel("Radians / s")
        if True:
            history = self.state_history
            x = [s[2] for s in history]
            y = [s[3] for s in history]
            plt.plot(x, y)
        plt.draw()
        plt.pause(0.01)
    
    def generate_plots(self):
        self.generate_theta_phase_plot()
        self.generate_cart_phase_plot()
        super().generate_plots() #order matters because quiver calls dx
        #self.generate_theta_vector_field_plot()

    def generate_vector_field_plot(self, dx_model = None):
        self.generate_theta_vector_field_plot()

    def get_reward(self):
        '''Reward = -Cost and vise-versa'''
        if hasattr(self, 'error_func'):
            return -self.error_func(self.state, self.get_target())
        return 0
    
    def get_environment_name(self):
        return 'Cartpole'

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
    TEST_CARTPOLE = False

    CSV = False
    if TEST_CARTPOLE:
        horizon = 20
        mc = 1
        mp = 0.1
        L = 1
        g = 9.8
        dt = 1e-2
        ARCTAN = False
        TSSMC = False
        ISMC = True
        MAX_GD = 20
        alpha = 1e-1
        #ucoeff = 1.5
        #umax = 1.5e1
        umax = 0.5e1
        umin = -umax
        #ucoeff = 3*umax / 4
        ucoeff = umax
        sigma_base = np.array([[1e0, 1e0, 1e0, 1e0]]).T #sliding surface definition
        sigma = sigma_base.copy() #sliding surface definition
        if ARCTAN:
            switch = lambda s: np.arctan(s) * 2/np.pi
        else:
            switch = lambda s: np.sign(s)
        #target = np.array([[0.0, 0, np.pi/16, 0]]).T
        target = np.array([[0.0, 0, 0, 0]]).T
        x0 = np.array([[-0, -.0, np.pi/1, 0.0]]).T
        #x0 = np.array([[-0, -.0, 0.0001, 0]]).T #NOTE: makes a cool pattern :)
        simplified_derivatives = False
        env = retrieve_control_environment('cartpole', 
                mc, mp, L, g,
                simplified_derivatives,
                x0, 
                interval = horizon, ts = dt,
                mode = 'point', 
                target = target)
        env.error_func = lambda x, t: (((x[2]-t[2])) + (x[3]-t[3]))
        env.reset()
        env.state = x0.copy()
        smc_lyap = []
        lyap = []
        d_du = []
        v = 0 #for TSSMC integrator
        integrated = np.zeros(env.state.shape) #integral-error
        if len(integrated.shape) < 2:
            integrated = integrated[..., np.newaxis]
        #x = env.state.copy()
        #if len(x.shape) > 1:
        #    x = x.reshape(-1, x.size)
        #if len(x.shape) < 2:
        #    x = x[..., np.newaxis]
        #x_ = x - np.array([x[0], x[1], target[2], x[3]])
        #x_ = x - target
        #sx = np.dot(sigma.T, x_)
        while not env.episode_is_done():
            print("STEP ", env.steps)
            x = env.state.copy()
            
            #if len(x.shape) > 1:
            #    x = x.reshape(-1, x.size)[0]
            print("State: ", x)
            x_ = x - np.array([x[0], x[1], target[2], target[3]])
            #x_ = x - np.array([x[0], x[1], target[2], x[3]])
            #x_ = np.array([x[0], x[1], target[2], x[3]]) - x
            #x_ = x - target
            #x_ = target - x
             
            if len(x_.shape) < 2:
                x_ = x_[..., np.newaxis]
            #x = x_ #to see if it makes a difference
            print("Target: %s \n Error: %s"%(target, x_))

            ## Sliding Mode Controls
            x_denom = (mc + mp * (np.sin(x[2]))**2)
            ddx = mp * np.sin(x[2]) * (L * (x[3])**2 - g*np.cos(x[2]))
            ddtheta = -mp * L * (x[3])**2 * np.sin(x[2])*np.cos(x[2])
            ddtheta += (mc + mp) * g * np.sin(x[2])
            dx = x[1] + (ddx) / x_denom + x[3] + (ddtheta) / (L * x_denom)
            #x' = h(x) + g(x)u
            hx = np.array([x[1], ddx/x_denom, x[3],ddtheta/(L*x_denom)])
            gx = np.array([[0, 1/x_denom, 0, -np.cos(x[2])/L * 1/x_denom]]).T
            print("Hx: %s \nGx: %s" % (hx, gx)) 
            ucomp = (1 - np.cos(x[2]) / L) * (1/x_denom)
            #ucomp = 1

            if ISMC == True:
                x = x_
                sigma = sigma_base.copy()
                sign = None
                mag = float('inf')
                i = 0
                while umax < mag and i < MAX_GD:
                    sf = -np.dot(sigma.T, hx)
                    sg = np.dot(sigma.T, gx)
                    if sf.size == 1 and sg.size == 1:
                        sf = sf[0]
                        sg = sg[0]
                    print("Sf: %s \nSg: %s" % (sf, sg))
                    mag = np.abs((sf / sg))
                    print("Mag: %s" % (mag))
                    if mag <= umax:
                        break
                    if sf.size == 1 and sg.size == 1:
                        grad = hx * 1/sg 
                        print("Grad: ", grad)
                        grad += -(1/sg * gx * 1/sg) * sf
                    else:
                        grad = np.dot(hx, 1/sg) 
                        grad += -np.dot(1/sg, np.dot(gx, 1/sg)) * sf
                    print("Grad+: ", grad)
                    #print("sg: %s sf: %s" % (sg, sf))
                    #if len(grad.shape) < len(sigma.shape):
                    #    grad = grad[..., np.newaxis]
                    sigma = sigma - alpha * grad
                    sigma = np.clip(sigma, 1e-2, 1e1)
                    #sigma = np.clip(sigma, -1e1, 1e1)
                    i += 1
                    #input()
                #if umax < mag:
                #    sigma = sigma_base.copy()
                print("Final Sigma: ", sigma)
                print("Final Mag: %s" % (mag))
                s = np.dot(sigma.T, x)
                sign = np.sign(s)
                #control = -umax * mag * sign
                control = ucoeff * sign
                #control += -(1/(np.dot(sigma.T, gx))) * (np.dot(sigma.T, hx))
                control = np.clip(control, -umax, umax)
            else:
                ## FOSMC
                sx = np.dot(sigma.T, x_)
                #u = lambda sigma, x: -ucoeff * (1/(np.dot(sigma.T, gx))) * -(np.abs(np.dot(sigma.T, hx))) * switch(np.dot(sigma.T, x))
                u = lambda sigma, x: ucoeff * switch(np.dot(sigma.T, x_))
                #u = lambda sigma, x: -umax * switch(sx)
                #u = lambda sigma, x: umax * (1/(np.dot(sigma.T, gx))) * (np.abs(np.dot(sigma.T, hx))) * switch(np.dot(sigma.T, x))
                #du_ds = lambda sigma, x : -ucoeff * (gx / hx) * 1/(1+(np.dot(sigma.T, x))**2) * x
                #print("du/ds: ", du_ds(sigma, x_))
                #d_du.append(du_ds(sigma, x_))
                #u = lambda sigma, x: ucoeff * (1/(np.dot(sigma.T, gx))) * -(np.abs(dx)) * np.sign(sx)
                control = u(sigma, x_)
                #control -= (1/(np.dot(sigma.T, gx))) * ((np.dot(sigma.T, hx)))
                control = np.clip(control, umin, umax)
                print("Control: ", control)

                ## TSSMC (twisted-surface)
                ##integrated += x_
            if TSSMC == True:
                sx = np.dot(sigma.T, x_)
                #sx = np.dot(sigma.T,x_ - integrated)
                c2 = 10
                lamb = 2
                #vi = np.clip(v, -1, 1)
                vi = v
                u = control + np.clip(-lamb * np.abs(sx)**(0.5) * switch(sx) + vi, umin, umax)

                if np.abs(v) < 1: #v is a saturation integrator, with 1 as its saturation value
                    v += (-c2 * switch(sx)) * dt
                #else:
                #    v += -v * dt
                #vi = np.clip(v + (-c2 * np.sign(sx)) * dt, -1, 1)
                #print("Integrated: ", integrated)
                print("Integral term: ", v)
                #dsigma = np.array([[0, 0, x[3], lamb*(hx[3]+gx[3]*u)]]).T#dsigma = dsigma/dt + d/dxsigma(f(x,u)), 2nd order surface changes OVER TIME
                #print("dsigma: ", dsigma)
                print("sigma: ", sx)
                #sx += (lamb*x[3] + (hx[3]+gx[3]*u)) * dt
                #print("NEW sigma: ", sigma)
                control = np.clip(u, umin, umax)
                #control = u
                
            print("CONTROL: ", control)
            
            print("Sliding Surface: ", np.dot(sigma.T, x_)) 
            print("(SMC) Lyapunov function: ", np.dot(sigma.T, x_).T * np.dot(sigma.T, x_))
            smc_lyap.append(np.dot(sigma.T, x_).T * np.dot(sigma.T, x_))
            lyap.append(np.linalg.norm(x_))
            print("(System) Lyapunov function: ", lyap[-1])
            #input()
            #env.step(np.zeros(1))
            env.step(control)
        env.generate_plots()
        if CSV: #write to local csv, this is sloppy to get data OUT
            filename = 'ismc' if ISMC else 'fosmc'
            filename += '_%s_%s' % (sigma_base[2], sigma_base[3])
            filename += '_%s' % (str(umax))
            filename += '.csv'
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(env.error_history)
                f.close()

        input()

    if TEST_INVERTED_PENDULUM:
        noisy_init = False
        target = np.array([0, 0])
        #target = np.array([0, 0])
        horizon = 8
        friction = 0.0
        env = retrieve_control_environment('inverted', 
                friction = friction,  
                noisy_init = noisy_init, 
                interval = horizon, ts = 0.01,
                mode = 'point', 
                target = target)
        env.reset()
        #env.state = np.array([0.0, 0.0])
        env.state = np.array([np.pi, 0e0])
        gamma = 0.7
        wn = 20
        umax = 0.80e-0
        ARCTAN = False
        ISMC = True
        sigma_base = np.array([1e0, 5e0], dtype=np.float64) #sliding surface definition
        sigma_history = []
        record_sigma = True
        while not env.episode_is_done():
            print("STEP ", env.steps)
            x = env.state
            #x_ = x - [target[0], x[1]]
            x_ = x - target
            x = x_
            print("State: ", x)
            print("Target: %s \n Error: %s"%(target, x_))
            ## Sliding Mode Controls
            g = np.array([0, -1]) #b vector
            f = np.array([x[1], np.sin(x[0]) - friction * x[1]])
            alpha = 1e0
            if ISMC:
                sigma = sigma_base.copy()
                sign = None
                mag = float('inf')
                i = 0
                while umax < mag and i < 20:
                    s = np.dot(sigma.T, x)
                    sign = np.sign(s)
                    sf = -np.dot(sigma.T, f)
                    sg = np.dot(sigma.T, g)
                    mag = np.abs((sf / sg))
                    if mag <= umax:
                        break
                    grad = np.dot(f, 1/sg) 
                    grad += -np.dot(1/sg, np.dot(g, 1/sg)) * sf
                    #print("sg: %s sf: %s" % (sg, sf))
                    print("Grad: ", grad)
                    sigma = sigma - alpha * grad
                    sigma = np.clip(sigma, 1e-1, float('inf'))
                    i += 1
                    #input()
                #if umax < mag:
                #    sigma = sigma_base.copy()
                print("Final Sigma: ", sigma)
                print("Final Mag: %s" % (mag))
                s = np.dot(sigma.T, x)
                sign = np.sign(s)
                #control = -umax * mag * sign
                control = umax * sign
                #control += -(1/(np.dot(sigma.T, g))) * (np.dot(sigma.T, f))
                control = np.clip(control, -umax, umax)
            else:
                sigma = sigma_base.copy()
                s = np.dot(sigma.T, x)
                sign = np.sign(s)
                if ARCTAN:
                    u = lambda sigma, x: umax * -((-x[1] - np.cos(x[0]) + friction * x[1])) * (2/np.pi * np.arctan(np.dot(sigma.T,  x)))
                else:
                    u = lambda sigma, x: umax * -((-x[1] - np.cos(x[0]) + friction * x[1])) * np.sign(np.dot(sigma.T,  x))
                print("Sliding Surface: ", np.dot(sigma.T, x)) 
                #control = u(sigma, x)
                control = umax * sign
                #control = u(sigma, x_)
            #input()
            #control = np.zeros(1)
            print("Control: ", control)
            #env.step(u(sigma, x_))
            env.step(control)
            if record_sigma:
                sigma_history.append(sigma)
        env.generate_plots()
        if record_sigma:
            plt.figure(15) #eh
            plt.clf()
            indices = [i for i in range(len(env.state_history)) if i % 10 == 0]
            x = [env.state_history[i][0] for i in indices]
            y = [env.state_history[i][1] for i in indices]
            dx = [-sigma_history[i][0]/sigma_history[i][1] for i in indices]
            dy = [1 for i in indices]
            plt.quiver(x, y, dx, dy)
            plt.title("Sliding Curve")
            plt.xlabel("Theta")
            plt.ylabel("dTheta/dt")
            plt.draw()
            plt.pause(0.01)

        if CSV: #write to local csv, this is sloppy to get data OUT
            filename = 'ismc' if ISMC else 'fosmc'
            filename += '_%s_%s' % (sigma_base[2], sigma_base[3])
            filename += '_%s' % (str(umax))
            filename += '.csv'
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(env.error_history)
                f.close()
        input()



