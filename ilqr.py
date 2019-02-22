import abc

from control_environment import *

import numpy as np

if True:
    import torch
    from torch.autograd import grad
    from torchviz import make_dot
    
    ## PyTorch helper functions
    def dx(f, x, create_graph = True):
        return grad(f, x, create_graph = create_graph)
    
    def dxx(f, x, create_graph = True):
        pass
    ##

class Model: 
    #TODO: Use LINEAR MODEL/Env to test iLQG / verify convergence
    #TODO: COMPARE WITH FINITE DIFFERENCING
    '''Generic Model, wrapping some sort of vector (or scalar) function
    with functionality to be applied to iLQG. 
    
    NOTE: This should be able to be vector-valued (for f) OR scalar (for
    l, the differential loss function)'''
    #def __init__(self, state_shape, state_size, 
    #        action_shape, action_size):
    #    self.state_shape = state_shape
    #    self.state_size = state_size
    #    self.action_shape = action_shape
    #    self.action_size = action_size
    @abc.abstractmethod
    def d_dx(self, xt, ut, dt, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        pass
    @abc.abstractmethod
    def d_dxx(self, xt, ut, dt, *args, **kwargs):
        '''Calculate df^2/(dxdx), given timestep
        size dt'''
        pass

    @abc.abstractmethod 
    def d_du(self, xt, ut, dt, *args, **kwargs):
        '''Calculate df/du, given timestep
        size dt'''
        pass
    @abc.abstractmethod 
    def d_duu(self, xt, ut, dt, *args, **kwargs):
        '''Calculate df^2/(dudu), given timestep
        size dt'''
        pass
    @abc.abstractmethod 
    def d_dxu(self, xt, ut, dt, *args, **kwargs):
        '''Calculate df^2/(dxdu), given timestep
        size dt'''
        pass
    def __call__(self, xt, ut, dt, *args, **kwargs):
        '''Returns dx, the change in state xt given control ut
        and timestep size dt.'''
        pass

class LinearQuadraticModel(Model):
    '''Generic linear quadratic model to use with linear quadratic
    functions.'''
    def __init__(self, Q):
        self.Q = Q
        self.shape = Q.shape
        self.size = Q.size
        print("NEW LQM: ", Q)
    def d_dx(self, xt, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        return 2*np.dot(xt, self.Q)
    def d_dxx(self, xt, ut=None, dt=None, *args, **kwargs):
        '''Calculate df^2/(dxdx), given timestep
        size dt'''
        return 2*self.Q #TODO TODO confirm this
    def __call__(self, xt=None, ut=None, dt=None, *args, **kwargs):
        '''Returns dx, the change in state xt given control ut
        and timestep size dt.'''
        #NOTE: xt is a ROW VECTOR   
        #print("QUADRATIC: X: %s Q: %s OUT: %s" % (xt, self.Q, 
        #    np.dot(xt.T, np.dot(self.Q,xt))))
        return np.dot(xt.T, np.dot(self.Q ,xt))

class CostModel(Model):
    '''Calling the CostModel corresponds to getting the cost
    at state xt (considering whether or not it is in terminal mode)
    given timestep size dt.'''
    def __init__(self):
        self.terminal = False
    def terminal_mode(self):
        self.terminal = True
    def normal_mode(self):
        self.terminal = False

class LQC(CostModel):
    def __init__(self, Q : np.ndarray, R : np.ndarray,
            Qf = None, target = None, diff_func =(lambda t,x:t-x)):
        print("I sure hope Q and R are symmetric (and/or EYE)...")
        super().__init__()
        self.Q = LinearQuadraticModel(Q)
        self.R = LinearQuadraticModel(R)

        self.Qf = Qf
        self.tmp_Q = None
        if Qf is not None:
            self.Qf = LinearQuadraticModel(Qf)
        self.set_target(target)

        self.diff_func = diff_func

    def terminal_mode(self):
        if not self.terminal:
            super().terminal_mode()
            self.tmp_Q = self.Q
            self.Q = self.Qf
    def normal_mode(self):
        if self.terminal:
            super().normal_mode()
            self.Q = self.tmp_Q
    def set_target(self, target):
        self.target = target

    def clear_target(self):
        self.target = None


    def d_dx(self, xt=None, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        assert(dt is not None)
        #print("Dx: ",self.Q.d_dx(xt, dt = dt) * dt)
        #print("MODE: ", self.terminal)
        if self.target is not None:
            xt = self.diff_func(self.target, xt)
        return self.Q.d_dx(xt, dt = dt) * dt
    def d_dxx(self, xt=None, ut=None, dt=None, *args, **kwargs):
        '''Calculate df^2/(dxdx), given timestep
        size dt'''
        assert(dt is not None)
        #print("Dxx: ",self.Q.d_dxx(xt, dt = dt) * dt)
        if self.target is not None:
            xt = self.diff_func(self.target, xt)
        return self.Q.d_dxx(xt, dt = dt) * dt
    def d_du(self, xt=None, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/du, given timestep
        size dt'''
        assert(dt is not None)
        #print("Du: ",self.R.d_dx(ut, dt = dt) * dt)
        return self.R.d_dx(ut, dt = dt) * dt
    def d_duu(self, xt=None, ut=None, dt=None, *args, **kwargs):
        '''Calculate df^2/(dudu), given timestep
        size dt'''
        assert(dt is not None)
        #print("Duu: ",self.R.d_dxx(ut, dt = dt) * dt)
        return self.R.d_dxx(ut, dt = dt) * dt
    def d_dxu(self, xt=None, ut=None, dt=None, *args, **kwargs):
        '''Calculate df^2/(dxdu), given timestep
        size dt'''
        assert(dt is not None)
        #print("Dxu: ",np.zeros((self.R.Q[0].size, self.Q.Q[0].size)))
        return np.zeros((self.R.Q[0].size, self.Q.Q[0].size))
    def __call__(self, xt, ut, dt=None, *args, **kwargs):
        if dt is None:
            dt = 1.0
        if self.target is not None:
            xt = self.diff_func(self.target, xt)
        else:
            raise Exception("You need a target atm. Deal with it.")
        if self.terminal:
            #if self.target is not None:
            #    #xt = 0.5*(self.target - xt)**2
            #    xt = self.target - xt
            #    print("ERROR (LQC Xt = xt - target): ", xt)
            return 0.5 * self.Qf(xt) 
            #return self.Qf(xt) * dt
        #return (self.Q(xt) + self.R(ut)) 
        #print("Q(xt): %s \n R(ut): %s" % (self.Q(xt), self.R(ut)))
        return 0.5 * (self.Q(xt) + self.R(ut)) * dt

class LQC_Controlled(LQC):
    def __init__(self, target : np.ndarray, *args, **kwargs):
        pass

class PyTorchModel(Model):
    '''Wraps a Pytorch Module in a Model so it can be painlessly
    integrated with iLQG.'''
    pass

class FiniteDifferencesModel(Model):
    '''Just...for comparison's sake, if it's EVEN possible with
    the environments I'm working with...'''
    pass

class LinearSystemModel(Model):
    '''For testing convergence / functionality of iLQR.'''
    def __init__(self, A : np.ndarray, B):
        self.A = A
        self.B = B
    def d_dx(self, xt, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        assert(dt is not None)
        return self.A * dt
    def d_du(self, xt = None, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        assert(dt is not None)
        assert(ut is not None)
        return self.B * dt
    def __call__(self, xt, ut, dt=None, *args, **kwargs):
        print("A*xt: ",np.dot(self.A, xt))
        print("B*ut: ",self.B * ut)
        return np.dot(self.A, xt) + self.B * ut

class ControlEnvironmentModel(Model):
    def __init__(self, env : ControlEnvironment):
        self.env = env
    def d_dx(self, xt, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        assert(dt is not None)
        #eye = np.eye(self.env.get_observation_size()) 
        #return (eye + self.env.d_dx(xt)) * dt
        #return eye + self.env.d_dx(xt) * dt
        return self.env.d_dx(xt) * dt
    def d_du(self, xt = None, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        assert(dt is not None)
        assert(ut is not None)
        print("d_du: ", self.env.d_du(xt, u = ut) * dt)
        return self.env.d_du(xt, u = ut) * dt
    def __call__(self, xt, ut, dt=None, *args, **kwargs):
        return self.env.dx(xt, ut) 

class LQG:
    def __init__():
        pass

class ILQG: #TODO: technically THIS is just iLQR, no noise terms cause NO
    '''Reference papers: https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf
    http://maeresearch.ucsd.edu/skelton/publications/weiwei_ilqg_CDC43.pdf
    https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

    Reference material: https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
    '''
    def __init__(self, state_shape, state_size, 
            action_shape, action_size, 
            model : Model, cost : Model, 
            noise : Model = None,
            action_constraints = None, 
            lamb_factor = 10, lamb_max = 1000,
            horizon = 10, initialization = 0.0, dt = 0.001,
            max_iterations = 1000, eps = 0.1):
        self.model = model
        self.cost = cost

        self.noise = noise

        self.lamb_factor = lamb_factor #for updating lambda
        self.lamb_max = lamb_max

        self.horizon = horizon
        self.state_shape = state_shape
        self.state_size = state_size
        self.action_shape = action_shape
        self.action_size = action_size
        self.initial_ut = initialization
        self.dt = dt

        self.max_iterations = max_iterations
        self.eps = eps

    def step(self, xt) -> (np.ndarray, np.ndarray):
        '''Perform a step through iLQG, starting with a given
        state xt, initializing a control sequence, and performing
        backwards recursion on the results until convergence is met
        for a control sequence linearized along xt.

        Based on https://github.com/studywolf/control/blob/master/studywolf_control/controllers/ilqr.py and details in https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf.
        '''
        sim_new_trajectory = True #I MISSED THIS IN THE ALGORITHM
        U = self.initialize_constant_control(xt, self.initial_ut)
        #print('States: ', X)
        #input()
        #self.model.env.generate_state_history_plot(X)
        #input()
        #self.cost.terminal_mode()
        #cost = self.cost(X[-1], U[-1], self.dt) #dt = 1 for terminal?
        #self.cost.normal_mode()

        lamb = 1.0 # regularization parameter, for LM Heuristic(seepaper/ref)
        ii = 0
        for ii in range(self.max_iterations): 
            print("STEP: ", ii)
            if sim_new_trajectory:
                X = self.forward(xt, U)
                cost = self.forward_cost(X, U) #dt = 1 for terminal?
                prev_cost = cost.copy()
                ## Forward Rollout
                TRAJ = [t for t in zip(X, U)]
                #print("TRAJ: ", [t for t in TRAJ])
                #calculate local derivatives along trajectories
                fx = [self.model.d_dx(t[0], t[1], self.dt) for t in TRAJ]
                fu = [self.model.d_du(t[0], t[1], self.dt) for t in TRAJ]
                #calculate first and 2nd derivatives for cost along traj
                #TODO TODO: should the LAST element be cost.d_dx in TERMINAL mode?
                #NOTE: ANSWER THIS pls it may be important ^
                lx = [self.cost.d_dx(t[0], t[1], self.dt) for t in TRAJ]
                lu = [self.cost.d_du(t[0], t[1], self.dt) for t in TRAJ]
                lxx = [self.cost.d_dxx(t[0], t[1], self.dt) for t in TRAJ]
                luu = [self.cost.d_duu(t[0], t[1], self.dt) for t in TRAJ]
                lxu = [self.cost.d_dxu(t[0], t[1], self.dt) for t in TRAJ]
                #l = [self.cost(t[0], t[1], self.dt) for t in TRAJ] 
                l = np.zeros((self.horizon, 1)) 
                l[-1] = self.forward_cost(X, U)
                self.cost.terminal_mode()
                print("FINAL: X: %s U: %s" % (X[-1], U[-1]))
                #l[-1] = self.cost(X[-1], U[-1]) #get terminal cost
                lx[-1] = self.cost.d_dx(X[-1], U[-1], self.dt) #terminal dl/dx
                lu[-1] = self.cost.d_du(X[-1], U[-1], self.dt) # EQUALS 0
                lxx[-1] = self.cost.d_dxx(X[-1], U[-1], self.dt) #ya get it
                luu[-1] = self.cost.d_duu(X[-1], U[-1], self.dt) # EQUALS 0
                lxu[-1] = self.cost.d_dxu(X[-1], U[-1], self.dt) # EQUALS 0
                print("Final Cost: ", l[-1])
                print("Final dl/dx: ", lx[-1])
                print("Final dl/du: ", lu[-1])
                #print("Len X: %s Len U: %s Len l: %s" % (len(X), len(U),len(l)))
                #input()
                self.cost.normal_mode() #this is...sorta grossth
                sim_new_trajectory = False

            V = l[-1].copy() #initialize cost-to-go at TERMINAL state
            Vx = lx[-1].copy()
            Vxx = lxx[-1].copy()
            k = np.zeros((self.horizon, self.action_size))
            K = np.zeros((self.horizon, self.action_size, self.state_size)) #

            #NOTE: To enable transition to higher-order methods, define
            #fxx, fuu, fxu and implement in backwards recursion

            ## Backwards Rollout
            for i in reversed(range(self.horizon - 1)):  
                #From Todorov: "The Q-function is the discrete-time 
                #analogue of the Hamiltonian, sometimes known as the 
                #pseudo-Hamiltonian"

                #We start from after terminal state, V(x,u) = cost(Xterm)
                #and perform backwards recursion to get seq. of approx.
                #cost terms.
                print("V: %s Vx: %s Vxx: %s \nFu: %s Fx: %s" % (V, Vx, 
                    Vxx, fu[i], fx[i]))
                print("lxx: %s lxu: %s luu: %s" % (lxx[i], 
                    lxu[i], luu[i]))

                Qx = lx[i] + np.dot(Vx, fx[i]) 
                Qu = lu[i] + np.dot(Vx, fu[i]) #lu[term] = 0]
                print("Qx: %s Qu: %s" % (Qx, Qu))
                #input()
                Qxx = lxx[i] + np.dot(fx[i].T, np.dot(Vxx, fx[i]))
                print("DOT: ",  np.dot(fx[i].T, np.dot(Vxx, fu[i])))
                Qxu = lxu[i] + np.dot(fx[i].T, np.dot(Vxx, fu[i]))
                Quu = luu[i] + np.dot(fu[i].T, np.dot(Vxx, fu[i]))
                print("Qxx: %s Qxu: %s Quu: %s" % (Qxx, Qxu, Quu))
                #input() 
                #apparently it's not recommended to calculate the
                #'raw' inverse of Quu, so instead we perform
                #the Levenberg-Marquardt heuristic, where we
                #instead calculate the inverse of the normalized
                #eigenvector-matrix WITH ALL NEGATIVE EIGENVALUES
                #SET TO ZERO (heuristic serves to ensure descent is
                #performed across all dimensions, or no steps at all)
                
                Quu_eval, Quu_evec = np.linalg.eig(Quu) 
                print("Quu Eigenvals: ", Quu_eval)
                print("Quu Eigenvecs: ", Quu_evec)
                Quu_eval[Quu_eval < 0] = 0.0 #remove negative eigenvals
                Quu_eval += lamb
                print("MODIFIED Quu Eigenvals: ", Quu_eval)
                Quu_inv = np.dot(Quu_evec, np.dot(np.diag(1.0/Quu_eval),
                    Quu_evec.T)) #quadratic function with reciproc-eigvals
                #Quu_inv = np.linalg.pinv(Quu_inv)
                print("Quu Inv: ", Quu_inv) 
                #update gain matrices, to be used to get delta-u
                #as a linear function of delta-x
                k[i] = -np.dot(Quu_inv, Qu)
                K[i] = -np.dot(Quu_inv, Qxu)
                print("ki: %s \n Ki: %s " % (k[i], K[i]))

                #update necessary values to go back one timestep
                #these updates are based on the Bellman equations,
                #but in a "backwards" time direction
                Vx = Qx - np.dot(K[i].T, np.dot(Quu, k[i]))
                Vxx = Qxx - np.dot(K[i].T, np.dot(Quu, K[i]))

            #forward pass to calculate new control sequence Unew
            #based on U' = U + deltaU = ut + kt + Kt * delta-xt
            xnew = xt.copy()
            Unew = np.zeros((self.horizon, self.action_size)) #"new" control sequence
            for i in range(self.horizon - 1):
                deltax = xnew - X[i]
                print("Dot: ", np.dot(K[i], deltax))
                Unew[i] = U[i] + k[i] + np.dot(K[i], deltax)
                dx = self.model(xnew, Unew[i]) * self.dt
                print("Unew: ", Unew[i])
                xnew += self.model(xnew, Unew[i]) * self.dt #get next state
                #xnew += self.model(xnew, Unew[i]) #get next state
                print("Xnew: %s dx: %s" % (xnew, dx))
            
            Xnew = self.forward(xt, Unew) #use updated control sequence
            costnew = self.forward_cost(Xnew, Unew) #dt = 1 for terminal?
            print("NEWCOST: ", costnew)
            print("Original: ", cost)
            #LM Heuristic:
            # Based on change in cost, update lambda to move optimization
            # toward 2nd (Newton's) or 1st (Gradient Descent) order method
            if costnew < cost: #move towards 2nd order method, update X,U
                lamb /= self.lamb_factor

                X = Xnew.copy()
                U = Unew.copy()
                prev_cost = np.copy(cost) #for stopping conditions
                cost = np.copy(costnew)
                sim_new_trajectory = True
                print("Delta-Cost: %s, Threshold: %s" % ((abs(prev_cost - cost)/cost),  self.eps))
                if ii > 0 and ((abs(prev_cost - cost)/cost) < self.eps):
                    print("CONVERGED at Iteration %s, Cost: %s" % (ii, 
                        cost)) 
                    print("Prev Cost: ", prev_cost)
                    break
            else: #move towards gradient descent
                lamb *= self.lamb_factor
                if lamb > self.lamb_max:
                    print("We're not converging (lamb = %s): %s vs %s" % (lamb, costnew, cost))
                    if False:
                        X = Xnew.copy()
                        U = Unew.copy()
                    break
        #input("Converged after %s steps!" % (ii))
        return X, U



    def initialize_constant_control(self, xt, initial=0.0) -> np.ndarray:
        #return np.ones((int((self.horizon)/self.dt), self.action_size))  * self.initial_ut 
        return np.ones((self.horizon - 1, self.action_size))  * self.initial_ut 

    def initialize_random_control(self, xt) -> np.ndarray:
        pass

    def forward(self, xt, control : []) -> []:
        '''Apply control sequence 'control' to system
        in starting state xt (via Euler Integration), receive 
        list containing xt, xt+1, ..., xt+horizon
        '''
        xt_ = xt.copy()
        states = [xt_.copy(),]
        for u in control:
            dx = self.model(xt_, u)
            xt_ += dx * self.dt
            #xt_ += dx
            #xt_[0] = xt_[0] % np.pi
            states.append(xt_.copy())
        return states

    def forward_cost(self, X, U):
        cost = 0.0
        for i in range(len(X)):
            if i == len(X) - 1: #terminal
                self.cost.terminal_mode()
                #cost += self.cost(X[i], None, dt = self.dt) * self.dt
                cost += self.cost(X[i], None, dt = self.dt) 
                self.cost.normal_mode()
            else:
                assert(self.cost.terminal is False)
                if self.cost(X[i], U[i], dt = self.dt) < 0:
                    assert("Negative cost - something's wrong!")
                cost += self.cost(X[i], U[i], dt = self.dt)
        return cost


def create_MPCController(control_base, *args, **kwargs):
    class iLQR_MPC(control_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def step(self, xt) -> (np.ndarray, np.ndarray):
            X, U = super().step(xt)
            assert(True)
            return X, U
        #TODO: allow for reusing of U given "similar" X (multiarmed)
    return iLQR_MPC(*args, **kwargs)        


#class Controller:
#    def __init__():
#        pass


if __name__ == '__main__':
    LINEARIZED_PENDULUM_TEST = False
    NONLINEAR_PENDULUM_TEST = True


    ##Nonlinear (inverted pendulum) controls test
    if NONLINEAR_PENDULUM_TEST:
        lamb_factor = 10
        lamb_max = 1000
        horizon = 1
        initialization = 0.0
        #initialization = 1.0
        dt = 1e-2
        max_iterations = 40
        eps = 0.001

        SECONDARY_STEP = False
        
        MPC_COMPARISON = True
        MPC_HORIZON = 0.1e0
        MPC_DT = dt
        MPC_STEPS = 1
        #MPC_STEPS = int(MPC_HORIZON / MPC_DT) - 1
        MPC_MAX_STEPS = int(horizon / dt / 2)
        MPC_THRESHOLD = 0.25
        

        
        #
        state_shape = [1, 2]
        state_size = 2
        action_shape = [1]
        action_size = 1
        
        #cost_func = lambda h,dt:1e4 * (5 * 1e-2) / (horizon * dt)
        cost_func = lambda h,dt:1e4
        #input("COST WEIGHT: %s" % (cost_func(horizon, dt)))
        #cost_func = lambda h,dt:1e4
        Q = np.eye(state_size) * cost_func(horizon, dt) * 1
        #Qf = Q
        Qf = np.eye(state_size) * cost_func(horizon, dt) * 0.5
        R = np.eye(action_size) * 1e0 * 0

        Q[1][1] = 0 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
        #Qf[1][1] = 0 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
        #Qf[1][1] = Qf[0][0] / 4 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
        #Q[1][1] = Q[0][0]/4 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
        #target = None
        #target = np.array([0, 0], dtype = np.float64)
        #target = np.array([0.5, 0], dtype = np.float64)
        #target = np.array([np.pi, 0], dtype = np.float64)
        target = np.array([np.pi/2, 0], dtype = np.float64)
        #target = np.array([np.pi/4, 0], dtype = np.float64)
        diff_func = lambda t,x : x - t
        #diff_func = lambda t,x : x + t
        #diff_func = lambda t,x : (t - x)**2
        #diff_func = lambda t,x : abs(t - x)
        #diff_func = lambda t,x:x #returns x, not a funciton of t
        cost = LQC(Q, R, Qf = Qf, target = target, 
                diff_func = diff_func)
        noisy_init = True
        friction = 0.000
        env = retrieve_control_environment('inverted', 
                friction = friction,
                noisy_init = noisy_init, 
                interval = horizon, ts = dt, #THESE DON'T MATTER FOR THIS
                mode = 'point', 
                target = (target if target is not None else np.zeros((state_size)))) #unnecess

        model = ControlEnvironmentModel(env)
        action_constraints = None
        #x0 = np.array([0, np.pi/2],dtype=np.float64)
        #x0 = np.array([0.0, np.pi/8],dtype=np.float64)
        ##x0 = np.array([np.pi, 0.0],dtype=np.float64) #NOTE: don't do 
        x0 = np.array([0.05, 0.00],dtype=np.float64)
        
        env.reset()
        env.state = x0.copy()
        #

        ilqg = ILQG(state_shape, state_size, action_shape, action_size,
                model, cost, None, action_constraints, #no noise model, iLQR
                lamb_factor = lamb_factor,
                lamb_max = lamb_max,
                horizon = int(horizon*1/dt),
                initialization = initialization,
                dt = dt,
                max_iterations = max_iterations, eps = eps)
        

        X, U = ilqg.step(x0)
        #U = [np.zeros(action_size) for i in range(int(horizon/dt))]
        print("FINAL U: ", U)
        env.state = x0.copy()
        for u in U:
            env.step(u)
        print("Final State: ", env.state_history[-1])
        print("Target: ", target)
        env.generate_plots()
        input()
        if SECONDARY_STEP:
            print("Next: the return trip to initial position")
            xt = env.state_history[-1].copy()
            target = x0.copy()
            #target = np.array([0, 0])
            #target = np.array([np.pi, 0])
            target = np.array([np.pi/2, 0])
            env.reset()
            env.set_target_point(target)
            cost.set_target(target)
            print("Initial state: %s Target: %s" % (xt, target))
            input()
            X, U = ilqg.step(xt) #MOVE BACK TO INITIAL POSITION
            print("FINAL U: ", U)
            env.state = xt.copy()
            for u in U:
                env.step(u)
            print("Final State: ", env.state_history[-1])
            print("Target: ", target)
            env.generate_plots()
            input()
        
        if MPC_COMPARISON:
            cost_func = lambda h,dt:1e8
            #input("COST WEIGHT: %s" % (cost_func(horizon, dt)))
            #cost_func = lambda h,dt:1e4
            Q = np.eye(state_size) * cost_func(horizon, dt) * 1
            #Qf = Q
            Qf = np.eye(state_size) * cost_func(horizon, dt) * 0
            #R = np.eye(action_size) * cost_func(horizon, dt) * 0 #NO controls applied
            R = np.eye(action_size) * 1e1
            Q[1][1] = 0 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
            cost = LQC(Q, R, Qf = Qf, target = target, 
                    diff_func = diff_func)

            mpc_ilqg = ILQG(state_shape, state_size, action_shape, action_size,
                model, cost, None, action_constraints, #no noise model, iLQR
                lamb_factor = lamb_factor,
                lamb_max = lamb_max,
                horizon = int(MPC_HORIZON*1/MPC_DT),
                initialization = initialization,
                dt = MPC_DT,
                max_iterations = max_iterations, eps = eps)
            env.reset()
            env.state = x0.copy()
            env.state_history.append(env.state.copy())
            xt = x0.copy()
            for i in range(MPC_MAX_STEPS):
                X, U = mpc_ilqg.step(xt)
                for j in range(MPC_STEPS):
                    u = U[j]
                    #U = [np.zeros(action_size) for i in range(int(horizon/dt))]
                    env.step(u)
                xt = env.state.copy()
                if abs(sum(diff_func(xt, target))) < MPC_THRESHOLD:
                    print("Early stopping condition met after %ss"% (i * dt))
                    print("Error: ", abs(sum(diff_func(xt, target))))
                    break
                #input("NEXT STEP!")
            print("Final State: ", env.state_history[-1])
            print("Target: ", target)
            env.generate_plots()
            input()


        
    
    ##Linear System test (linearized inverted pendulum)
    if LINEARIZED_PENDULUM_TEST: #TODO: wrap this shit in Unittest eventually
        lamb_factor = 5
        lamb_max = 1000
        horizon = 3
        initialization = 0.0
        #initialization = 1.0
        dt = 1e-2
        max_iterations = 50
        eps = 0.001
        
        MPC_COMPARISON = True
        MPC_HORIZON = 0.1e0
        MPC_DT = dt
        MPC_MAX_STEPS = int(horizon / dt) * 1
        
        #
        state_shape = [1, 2]
        state_size = 2
        action_shape = [1]
        action_size = 1
        
        cost_func = lambda h,dt:1e4 * (5 * 1e-2) / (horizon * dt)
        cost_func = lambda h,dt:1e5
        #input("COST WEIGHT: %s" % (cost_func(horizon, dt)))
        #cost_func = lambda h,dt:1e4
        Q = np.eye(state_size) * cost_func(horizon, dt) * 1
        #Qf = Q
        Qf = np.eye(state_size) * cost_func(horizon, dt) * 1
        R = np.eye(action_size) * 1e0 * 0
        #Q[1][1] = 0 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
        #Q[1][1] = Q[0][0]/4 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
        #R[0][0] = 0.0 #only concerned with force applied to controllable var
        #target = None
        target = np.array([0, 0], dtype = np.float64)
        #target = np.array([0.05, 0], dtype = np.float64)
        #target = np.array([np.pi, 0], dtype = np.float64)
        #target = np.array([np.pi/2, 0], dtype = np.float64)
        diff_func = lambda t,x : x - t
        #diff_func = lambda t,x : x + t
        #diff_func = lambda t,x : (t - x)**2
        #diff_func = lambda t,x : abs(t - x)
        #diff_func = lambda t,x:x #returns x, not a funciton of t
        cost = LQC(Q, R, Qf = Qf, target = target, 
                diff_func = diff_func)
        noisy_init = True
        friction = 0.00
        A = np.array([[0, 1],[1, -friction]])
        B = np.array([0, 1]) 
        model = LinearSystemModel(A, B)
        action_constraints = None
        #x0 = np.array([0, np.pi/2],dtype=np.float64)
        #x0 = np.array([0.0, np.pi/8],dtype=np.float64)
        x0 = np.array([0.03, 0.00],dtype=np.float64)
        #x0 = np.array([0.00, 0.00],dtype=np.float64)
        #

        ilqg = ILQG(state_shape, state_size, action_shape, action_size,
                model, cost, None, action_constraints, #no noise model, iLQR
                lamb_factor = lamb_factor,
                lamb_max = lamb_max,
                horizon = int(horizon*1/dt),
                initialization = initialization,
                dt = dt,
                max_iterations = max_iterations, eps = eps)


        X, U = ilqg.step(x0)
        #U = [np.zeros(action_size) for i in range(int(horizon/dt))]
        X = ilqg.forward(x0, U)
        print("FINAL U: ", U)
        fig = plt.figure()
        x = [s[0] for s in X]
        y = [s[1] for s in X]
        plt.plot(x,y, label='parametric curve')
        plt.plot(x[0], y[0], 'ro')
        plt.plot(x[-1], y[-1], 'gx')
        plt.plot(target[0], target[0], 'b^')
        plt.title('Phase plot of Linearized Inverted Pendulum')
        plt.ylabel("Velocity")
        plt.xlabel("Position")
        plt.draw()
        plt.pause(0.01) 
        input()
        if MPC_COMPARISON:
            #cost_func = lambda h,dt:1e4 * (5 * 1e-2) / (horizon * dt)
            cost_func = lambda h,dt:1e8
            #input("COST WEIGHT: %s" % (cost_func(horizon, dt)))
            #cost_func = lambda h,dt:1e4
            Q = np.eye(state_size) * cost_func(horizon, dt) * 1
            #Qf = Q
            Qf = np.eye(state_size) * cost_func(horizon, dt) * 0
            R = np.eye(action_size) * 1e0 * 0
            Q[1][1] = 0 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
            cost = LQC(Q, R, Qf = Qf, target = target, 
                    diff_func = diff_func)
            mpc_ilqg = ILQG(state_shape, state_size, action_shape, action_size,
                model, cost, None, action_constraints, #no noise model, iLQR
                lamb_factor = lamb_factor,
                lamb_max = lamb_max,
                horizon = int(MPC_HORIZON*1/MPC_DT),
                initialization = initialization,
                dt = MPC_DT,
                max_iterations = max_iterations, eps = eps)
            xt = x0.copy()
            x_hist = [xt.copy(),]
            for i in range(MPC_MAX_STEPS):
                X, U = mpc_ilqg.step(xt)
                u = U[0]
                print("Next u: ", u)
                dx = model(xt, u)
                xt += dx * MPC_DT
                x_hist.append(xt.copy())
                #input("NEXT STEP!")
            print("Final State: ", xt)
            print("Target: ", target)
            x = [s[0] for s in x_hist]
            y = [s[1] for s in x_hist]
            plt.plot(x,y, label='parametric curve')
            plt.plot(x[0], y[0], 'ro')
            plt.plot(x[-1], y[-1], 'g')
            plt.draw()
            plt.pause(0.01) 
            input()



    ### Simple differentiation test
    #x = torch.arange(4, dtype = torch.float, requires_grad=True).reshape(2, 2)
    #loss = (x ** 4).sum()
    #print("LOSS: ", loss)
    #grads = dx(loss, x, create_graph = True)
    ##make_dot(grads).view()
    ##input()
    #print("Gradients: ", grads)

    ### differentiation w.r.t single LINEAR / AFFINE torch.nn weights
    #lin_size = 5
    #indim = lin_size
    #outdim = 1
    #inp = torch.arange(lin_size, dtype = torch.float, requires_grad = True)
    #W = torch.ones([1, lin_size], dtype = torch.float, requires_grad = True)
    #linear = torch.nn.Linear(indim, outdim, bias = False)
    #linear.weight.data = W
    #print("Linear weights: ", linear.weight)
    #out = linear(inp)
    #jac = dx(out, inp, create_graph = True)
    #print("OUT: ", out)
    #print('"Jacobian" : ', jac)
    #expected = torch.ones([outdim], dtype = torch.float)
    #loss = torch.nn.MSELoss()(out, expected)
    #loss_dx = dx(loss, inp)
    #loss_dW = dx(loss, linear.weight)
    #print('Loss : ', loss)
    #print('Loss d/dx: ', loss_dx)
    #print('Loss d/dW: ', loss_dW)
    ##make_dot(grads).view()
    ##input()

    ### differentiation w.r.t two (sequential) LINEAR torch.nn weights
    #lin_size = 5
    #indim = lin_size
    #hdim = 3
    #outdim = 1
    #inp = torch.arange(lin_size, dtype = torch.float, requires_grad = True)
    #W1 = torch.ones([hdim, lin_size], dtype = torch.float, requires_grad = True)
    #W2 = torch.ones([1, hdim], dtype = torch.float, requires_grad = True) * 2
    #l1 = torch.nn.Linear(indim, hdim, bias = False)
    #l2 = torch.nn.Linear(hdim, outdim, bias = False)
    #l1.weight.data = W1
    #l2.weight.data = W2
    #print("Linear weights: ", l1.weight)
    #h = l1(inp)
    #out = l2(h)
    #print("OUT: ", out)
    #jac = dx(out, inp, create_graph = True)
    #print('"Jacobian" : ', jac)
    #expected = torch.ones([outdim], dtype = torch.float)
    #loss = torch.nn.MSELoss()(out, expected)
    #loss_dx = dx(loss, inp)
    #loss_dW1 = dx(loss, l1.weight)
    #loss_dW2 = dx(loss, l2.weight)
    #print('Loss : ', loss)
    #print('Loss d/dx: ', loss_dx)
    #print('Loss d/dW1: ', loss_dW1)
    #print('Loss d/dW2: ', loss_dW2)
    ##make_dot(grads).view()
    

