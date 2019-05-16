import abc

from control_environment import *

import numpy as np

from model import * 

    
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
        return self.env.d_dx(xt)
    def d_du(self, xt = None, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        assert(dt is not None)
        assert(ut is not None)
        #print("d_du: ", self.env.d_du(xt, u = ut) * dt)
        return self.env.d_du(xt, u = ut)
    def __call__(self, xt, ut, dt=None, *args, **kwargs):
        return self.env.dx(xt, ut) 

#class LQG:
#    def __init__():
#        pass
#NAAAAAAH 

class DDP:
    '''Abstract baseclass for DDP methods (and DDP-like methods, like
    "iterative" sliding mode control.'''
    def __init__(self, state_shape, state_size, 
            action_shape, action_size,
            model : Model, cost : Model = None, 
            noise : Model = None, 
            action_constraints = None, 
            horizon = 10, dt = 0.001,
            max_iterations = 1000, eps = 0.1,
            update_model = True):
        self.model = model
        self.update_model = update_model #determine if call model.forward to update internal state
        self.cost = cost

        self.noise = noise

        self.horizon = horizon
        self.state_shape = state_shape
        self.state_size = state_size
        self.action_shape = action_shape
        self.action_size = action_size
        self.dt = dt

        self.action_constraints = action_constraints

        self.max_iterations = max_iterations
        self.eps = eps


    @abc.abstractmethod
    def step(self, xt) -> (np.ndarray, np.ndarray):
        pass

    def forward(self, xt, control : []) -> []:
        '''Apply control sequence 'control' to system
        in starting state xt (via Euler Integration), receive 
        list containing xt, xt+1, ..., xt+horizon
        '''
        xt_ = xt.copy()
        states = [xt_.copy(),]
        for u in control:
            if self.update_model:
                traj = (xt_, u)
                self.model.update(traj[0])
            dx = self.model(xt_, u)
            #print("Xt: %s \ndx: %s " % (xt_, dx))
            xt_ += dx * self.dt
            #xt_ += dx
            #xt_[0] = xt_[0] % np.pi
            states.append(xt_.copy())
        return states

    def forward_cost(self, X, U):
        assert(self.cost is not None)
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


class ILQG(DDP): #TODO: technically THIS is just iLQR, no noise terms cause NO
    '''Reference papers: https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf
    http://maeresearch.ucsd.edu/skelton/publications/weiwei_ilqg_CDC43.pdf
    https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

    Reference material: https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
    '''
    def __init__(self, full_iterations = False, 
            lamb_factor = 10, lamb_max = 1000, 
            initialization = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("INITIALIZATION: ", initialization)
        self.lamb_factor = lamb_factor #for updating lambda
        self.lamb_max = lamb_max
        self.initial_ut = initialization

        self.full_iterations = full_iterations


    def step(self, xt) -> (np.ndarray, np.ndarray):
        #if len(xt.shape) < 2:
        #    xt = xt[..., np.newaxis]
        if self.full_iterations:
            return self._step_full_iterations(xt)
        else:
            return self._step_closed_form(xt)

    def _step_full_iterations(self, xt, U = None) -> (np.ndarray, np.ndarray):
        '''Perform a step through iLQG, starting with a given
        state xt, initializing a control sequence, and performing
        backwards recursion on the results until convergence is met
        for a control sequence linearized along xt.

        Based on https://github.com/studywolf/control/blob/master/studywolf_control/controllers/ilqr.py and details in https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf.
        '''
        sim_new_trajectory = True #I MISSED THIS IN THE ALGORITHM
        U = self.initialize_control(xt, self.initial_ut)
        #print('States: ', X)
        #input()
        #self.model.env.generate_state_history_plot(X)
        #input()
        print("INITIAL CONTROLS: ", U.shape)
        #self.cost.terminal_mode()
        #cost = self.cost(X[-1], U[-1], self.dt) #dt = 1 for terminal?
        #self.cost.normal_mode()

        lamb = 1.0 # regularization parameter, for LM Heuristic(seepaper/ref)
        ii = 0
        for ii in range(self.max_iterations): 
            #print("STEP: ", ii)
            if sim_new_trajectory:
                X = self.forward(xt, U)
                #if len(X[0].shape) < 2:
                #    X = [x.flatten() for x in X]
                cost = self.forward_cost(X, U) #dt = 1 for terminal?
                prev_cost = cost.copy()
                ## Forward Rollout
                TRAJ = [t for t in zip(X, U)]
                #print("TRAJ: ", [t for t in TRAJ])
                #calculate local derivatives along trajectories
                if self.update_model:
                    fx = []
                    fu = []
                    for t in TRAJ:
                        self.model.update(t[0])
                        fx.append(np.eye(X[0].shape[0]) + self.model.d_dx(t[0], t[1], self.dt) * self.dt)
                        #fx.append(self.model.d_dx(t[0], t[1], self.dt) * self.dt)
                        fu.append(self.model.d_du(t[0], t[1], self.dt) * self.dt)
                else:
                    fx = [np.eye(X[0].shape[0]) + self.model.d_dx(t[0], t[1], self.dt) * self.dt for t in TRAJ]
                    #fx = [self.model.d_dx(t[0], t[1], self.dt) * self.dt for t in TRAJ]
                    fu = [self.model.d_du(t[0], t[1], self.dt) * self.dt for t in TRAJ]
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
                #print("FINAL: X: %s U: %s" % (X[-1], U[-1]))
                #l[-1] = self.cost(X[-1], U[-1]) #get terminal cost
                lx[-1] = self.cost.d_dx(X[-1], U[-1], self.dt) #terminal dl/dx
                lu[-1] = self.cost.d_du(X[-1], U[-1], self.dt) # EQUALS 0
                lxx[-1] = self.cost.d_dxx(X[-1], U[-1], self.dt) #ya get it
                luu[-1] = self.cost.d_duu(X[-1], U[-1], self.dt) # EQUALS 0
                lxu[-1] = self.cost.d_dxu(X[-1], U[-1], self.dt) # EQUALS 0
                #print("Final Cost: ", l[-1])
                #print("Final dl/dx: ", lx[-1])
                #print("Final dl/du: ", lu[-1])
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
                Qx = self.dq_dx(lx[i],lu[i],lxx[i],luu[i],lxu[i],fx[i],fu[i],
                        None,None,Vx,Vxx) 
                Qu = self.dq_du(lx[i],lu[i],lxx[i],luu[i],lxu[i],fx[i],fu[i],
                        None,None,Vx,Vxx) 
                Qxx = self.dq_dxx(lx[i],lu[i],lxx[i],luu[i],lxu[i],fx[i],fu[i],
                        None,None,Vx,Vxx) 
                Qxu = self.dq_dxu(lx[i],lu[i],lxx[i],luu[i],lxu[i],fx[i],fu[i],
                        None,None,Vx,Vxx) 
                Quu = self.dq_duu(lx[i],lu[i],lxx[i],luu[i],lxu[i],fx[i],fu[i],
                        None,None,Vx,Vxx) 
                #np.clip(Quu, 1e4, 1e4, out = Quu)
                #apparently it's not recommended to calculate the
                #'raw' inverse of Quu, so instead we perform
                #the Levenberg-Marquardt heuristic, where we
                #instead calculate the inverse of the normalized
                #eigenvector-matrix WITH ALL NEGATIVE EIGENVALUES
                #SET TO ZERO (heuristic serves to ensure descent is
                #performed across all dimensions, or no steps at all)
                Quu_inv = self.dq_duu_inv(Quu, LM_METHOD = True, lamb = lamb)
                
                #update gain matrices, to be used to get delta-u
                #as a linear function of delta-x
                print("Quu: %s Qu: %s " % (Quu_inv.shape, Qu.shape))
                #input()
                k[i] = -np.dot(Quu_inv, Qu)
                K[i] = -np.dot(Quu_inv, Qxu)
                #print("ki: %s \n Ki: %s " % (k[i], K[i]))

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
                #print("Dot: ", np.dot(K[i], deltax))
                Unew[i] = U[i] + k[i] + np.dot(K[i], deltax)
                if self.update_model:
                    traj = (xnew, Unew[i])
                    self.model.update(traj[0])
                dx = self.model(xnew, Unew[i]) * self.dt
                #print("Unew: ", Unew[i])
                if len(dx) > 1:
                    dx = dx.flatten()
                print("Xnew: %s dx: %s" % (xnew.shape, dx.shape))
                xnew += dx #get next state
                #xnew += self.model(xnew, Unew[i]) #get next state
            
            Xnew = self.forward(xt, Unew) #use updated control sequence
            costnew = self.forward_cost(Xnew, Unew) #dt = 1 for terminal?
            #print("NEWCOST: ", costnew)
            #print("Original: ", cost)
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
                #print("Delta-Cost: %s, Threshold: %s" % ((abs(prev_cost - cost)/cost),  self.eps))
                if ii > 0 and ((abs(prev_cost - cost)/cost) < self.eps):
                    #print("CONVERGED at Iteration %s, Cost: %s" % (ii, 
                        #cost)) 
                    #print("Prev Cost: ", prev_cost)
                    break
            else: #move towards gradient descent
                lamb *= self.lamb_factor
                if lamb > self.lamb_max:
                    #print("We're not converging (lamb = %s): %s vs %s" % (lamb, costnew, cost))
                    if False:
                        X = Xnew.copy()
                        U = Unew.copy()
                    break
        #input("Converged after %s steps!" % (ii))
        return X, U
    
    def _step_closed_form(self, xt) -> (np.ndarray, np.ndarray):
        sim_new_trajectory = True #I MISSED THIS IN THE ALGORITHM
        U = self.initialize_control(xt, self.initial_ut)
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
                if self.update_model:
                    fx = []
                    fu = []
                    for t in TRAJ:
                        self.model.update(t[0])
                        fx.append(self.model.d_dx(t[0], t[1], self.dt))
                        fu.append(self.model.d_du(t[0], t[1], self.dt))
                else:
                    fx = [self.model.d_dx(t[0], t[1], self.dt) for t in TRAJ]
                    fu = [self.model.d_du(t[0], t[1], self.dt) for t in TRAJ]
                #calculate first and 2nd derivatives for cost along traj
                #TODO TODO: should the LAST element be cost.d_dx in TERMINAL mode?
                sim_new_trajectory = False


            v = np.zeros((self.horizon, self.state_size, 1))
            K = np.zeros((self.horizon, self.action_size, self.state_size))
            Kv = np.zeros((self.horizon, self.action_size, self.state_size)) #
            Ku = np.zeros((self.horizon, self.action_size, self.action_size)) #
            S = self.cost.Qf.Q #initialize cost-to-go at TERMINAL state
            
            #self.cost.terminal_mode()
            #v[-1] = self.cost(X[-1], None) 
            #self.cost.normal_mode()
            diff = self.cost.diff_func(self.cost.target, X[-1])
            if len(diff.shape) < 2:
                diff = diff[..., np.newaxis]
            print("Diff: ", diff)
            v[-1] = np.dot(self.cost.Qf.Q, diff)
            #v[-1] = cost.copy()
            print("VN: ", v[-1])
            ## (CLOSED-FORM) Backwards Rollout
            for i in reversed(range(self.horizon - 1)):  
                #From Todorov: "The Q-function is the discrete-time 
                #analogue of the Hamiltonian, sometimes known as the 
                #pseudo-Hamiltonian"
                #https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf

                if self.update_model:
                    self.model.update(X[i])
                dx = self.model(X[i], U[i])
                if len(dx) > 1:
                    dx = dx.flatten()
                dx = dx * self.dt
                #Ak = np.dot(fx[i], dx)
                #Bk = np.dot(fu[i], dx)
                Ak = fx[i] * self.dt
                Ak += np.eye(Ak.shape[0])
                Bk = fu[i] * self.dt
                if len(Bk.shape) < 2:
                    Bk = Bk[..., np.newaxis]
                if len(Ak.shape) < 2:
                    Ak = Ak[..., np.newaxis]
                
                print("Ak: %s \n Bk: %s" % (Ak, Bk))
                print("fx: %s \n fu: %s" % (fx[i], fu[i]))
                Q = self.cost.Q.Q
                #Q += np.eye(Q.shape[0])
                R = self.cost.R.Q
                ricatti = np.linalg.inv(np.dot(Bk.T, np.dot(S, Bk)) + R)
                print("Ricatti: ", ricatti.shape)
                print("R: ", R.shape)
                K[i] = np.dot(ricatti, (np.dot(Bk.T, np.dot(S, Ak))))
                Kv[i] = np.dot(ricatti, (Bk.T))
                Ku[i] = np.dot(ricatti, R)
                print("K: %s \n Kv: %s \n Ku: %s" % (K[i], Kv[i], Ku[i]))
                S = np.dot(np.dot(Ak.T, S), (Ak - np.dot(Bk, K[i]))) + Q
                v[i] = np.dot((Ak - np.dot(Bk, K[i])).T , v[i+1])
                print("V: ", v[i])
                u_term = np.dot(K[i].T, np.dot(R, U[i]))
                if len(u_term.shape) < 2:
                    u_term = u_term[..., np.newaxis]
                print("u_term: ", u_term)
                x_term = np.dot(Q, X[i])
                if len(x_term.shape) < 2:
                    x_term = x_term[..., np.newaxis]
                print("X_term: ", x_term)
                v[i] -= u_term
                v[i] += u_term
                print("V: ", v[i])
                #input()

            #input()
            #forward pass to calculate new control sequence Unew
            #based on U' = U + deltaU = ut + kt + Kt * delta-xt
            xnew = xt.copy()
            Unew = np.zeros((self.horizon, self.action_size, 1)) #"new" control sequence
            deltax = np.zeros(xnew.shape)
            if len(deltax.shape) < 2:
                deltax = deltax[..., np.newaxis]
            for i in range(self.horizon - 1):
                print("Deltax: ", deltax)
                deltau = -np.dot(K[i],deltax) 
                if len(deltau.shape) < 2:
                    deltau = deltau[..., np.newaxis]
                deltau -= np.dot(Kv[i],v[i+1]) 
                u_term = np.dot(Ku[i], U[i])
                if len(u_term.shape) < 2:
                    u_term = u_term[..., np.newaxis]
                deltau -= u_term
                Unew[i] = U[i] + deltau
                #if self.update_model:
                #    traj = (xnew, Unew[i])
                #    self.model.update(traj[0])
                print("Unew: ", Unew[i])
                
                #dx = self.model(xnew, Unew[i]) * self.dt
                #xnew += dx #get next state
                #deltax = xnew - X[i + 1]
                
                
                #Ak = fx[i]
                #Bk = fu[i]
                #Ak = fx[i] * self.dt
                #Ak += np.eye(Ak.shape[0])
                #Bk = fu[i] * self.dt
                Ak = self.model.d_dx(xnew, Unew[i], self.dt) * self.dt
                Ak += np.eye(Ak.shape[0])
                Bk = self.model.d_du(xnew, Unew[i], self.dt) * self.dt
                if len(Bk.shape) < 2:
                    Bk = Bk[..., np.newaxis]
                print("Ak: ", Ak)
                print("Bk: ", Bk)
                print("deltau: ", deltau)
                print("Ax*dx: ", np.dot(Ak, deltax))
                print("Bu*du: ", np.dot(Bk, deltau))
                deltax = (np.dot(Ak, deltax) + np.dot(Bk, deltau)) 
                #input()
            
            Xnew = self.forward(xt, Unew) #use updated control sequence
            costnew = self.forward_cost(Xnew, Unew) #dt = 1 for terminal?
            print("NEWCOST: ", costnew)
            print("Original: ", cost)
            #X = Xnew.copy()
            #U = Unew.copy()
            #sim_new_trajectory = True
            #input()
            if costnew < cost: #move towards 2nd order method, update X,U
                X = Xnew.copy()
                U = Unew.copy()
                prev_cost = np.copy(cost) #for stopping conditions
                cost = np.copy(costnew)
                sim_new_trajectory = True
                #print("Delta-Cost: %s, Threshold: %s" % ((abs(prev_cost - cost)/cost),  self.eps))
                if ii > 0 and ((abs(prev_cost - cost)/cost) < self.eps):
                    print("CONVERGED at Iteration %s, Cost: %s" % (ii, 
                        cost)) 
                    print("Prev Cost: ", prev_cost)
                    break
                #input()
            else: #move towards gradient descent
                pass
            #input()
        #input("Converged after %s steps!" % (ii))
        return X, U

    def initialize_control(self, xt, initial = 0.0):
        if hasattr(self, 'smc'):
            return self.initialize_smc_control(xt)
        else:
            return self.initialize_constant_control(xt, initial)

    def initialize_constant_control(self, xt, initial=0.0) -> np.ndarray:
        #return np.ones((int((self.horizon)/self.dt), self.action_size))  * self.initial_ut 
        return np.ones((self.horizon - 1, self.action_size))  * self.initial_ut 
    def set_smc(self, smc):
        self.smc = smc

    def initialize_smc_control(self, xt) -> np.ndarray: 
        #TODO TODO TODO: current method performs euler integration TWICE needlessly
        assert(hasattr(self, 'smc'))
        #assert(type(self.model) == LinearSystemModel or 
        #        type(self.model) == PyTorchDecoupledSystemModel)
        xt_ = xt.copy()
        if len(xt_.shape) < 2:
            xt_ = xt_[..., np.newaxis]
        U = []
        for i in range(self.horizon): 
            _, u = self.smc.step(xt_)
            U.append(u)
            print("U: ", u)
            dx = self.model(xt_, u)
            print("Xt: %s \ndx: %s " % (xt_.shape, dx.shape))
            xt_ += dx * self.dt
        if len(u[0].shape) < 2:
            U = [u.flatten() for u in U]
        return np.array(U)

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
            if self.update_model:
                traj = (xt_, u)
                self.model.update(traj[0])
            dx = self.model(xt_, u)
            if len(xt_) > 1:
                xt_ = xt_.flatten()
            if len(dx) > 1:
                dx = dx.flatten()
            print("Xt_ : ", xt_.shape)
            print("dx : ", dx.shape)
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
                print("Cost: ", self.cost(X[i], U[i], dt = self.dt))
                if self.cost(X[i], U[i], dt = self.dt) < 0:
                    assert("Negative cost - something's wrong!")
                cost += self.cost(X[i], U[i], dt = self.dt)
        return cost

    def dq_dx(self, lx, lu, lxx, luu, lxu, fx, fu, fuu, fxx, Vx, Vxx, *args, **kwargs):
        return lx + np.dot(Vx, fx) 
    def dq_du(self, lx, lu, lxx, luu, lxu, fx, fu, fuu, fxx, Vx, Vxx, *args, **kwargs):
        return lu + np.dot(Vx, fu) #lu[term] = 0]
    def dq_dxx(self, lx, lu, lxx, luu, lxu, fx, fu, fuu, fxx, Vx, Vxx, *args, **kwargs):
        return lxx + np.dot(fx.T, np.dot(Vxx, fx))
    def dq_dxu(self, lx, lu, lxx, luu, lxu, fx, fu, fuu, fxx, Vx, Vxx, *args, **kwargs):
        return lxu + np.dot(fu.T, np.dot(Vxx, fx))
    def dq_duu(self, lx, lu, lxx, luu, lxu, fx, fu, fuu, fxx, Vx, Vxx, *args, **kwargs):
        return luu + np.dot(fu.T, np.dot(Vxx, fu))
    def dq_duu_inv(self, Quu, lamb = 1.0, LM_METHOD = True, *args, **kwargs):
        if not LM_METHOD:
            Quu_inv = np.linalg.pinv(Quu)
        else:
            try:
                Quu_eval, Quu_evec = np.linalg.eig(Quu) 
            except:
                print("Quu: ", Quu)
                input()
            Quu_eval[Quu_eval < 0] = 0.0 #remove negative eigenvals
            Quu_eval += lamb 
            #print("MODIFIED Quu Eigenvals: ", Quu_eval)
            Quu_inv = np.dot(Quu_evec, np.dot(np.diag(1.0/Quu_eval),
                Quu_evec.T)) #quadratic function with reciproc-eigvals
            return Quu_inv       

class SMC(DDP):
    '''(Abstract, default LINEAR, STATIC) Implementation of iterative sliding mode controller.'''
    def __init__(self, surface_base : np.ndarray, 
            target : np.ndarray, diff_func = lambda t,x : x-t,
            switching_function = 'sign', 
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surface_base = surface_base
        self.surface = self.surface_base

        self.switching_function = switching_function

        self.target = target
        self.diff_func = diff_func

    def step(self, xt):
        #print("MODEL: ", self.model)
        if len(xt.shape) < 2:
            xt = xt[..., np.newaxis]
        #print('Target: %s Xt: %s' % (self.target.shape, xt.shape))
        #x = xt.copy()
        xt = self.diff_func(self.target, xt)
        if self.update_model:
            #if len(xt.shape) > 1:
            #    xt = xt.flatten()
            self.model.update(xt)
        #self.update_surface(x)
        self.update_surface(xt)
        #raise Exception("Figure out the-above")
        return None, self.compute_control(xt) #TODO: make this NOT a subclass of DDP so we don't need this janky None returnval?

    def forward_cost(self, X, U):
        return 0.0

    def compute_control(self, xt) -> np.ndarray:
        '''Compute FIRST-ORDER sliding mode control given either
        a linear system model (x' = Ax + Bu) or general system model
        (x' = f + gu)'''
        if (issubclass(type(self.model), LinearSystemModel)):
            sign = self.compute_switch(xt)
            ds_dx = self.get_surface_d_dx(xt)
            dB = np.dot(ds_dx.T, self.model.B)
            dA = np.dot(ds_dx.T, np.dot(self.model.A, xt))
            #print("ds_dx: %s \nSign: %s" % (ds_dx, sign))
            #print("ds_dx.T: ", (np.dot(self.surface.T, xt)))
            #print("Sign: %s" % (sign))
            #print("B: ", self.model.B)
            #print("A: ", self.model.A)
            #print("xt: ", xt)
            #print("DOT: ", (dB))
            #print("|DOT|: ", np.linalg.det(dB))
            magnitude = -(np.linalg.inv(dB)) #TODO: Verify this step
            #print("dB inv: ", magnitude)
            #print("dA: ", dA)
            #magnitude *= dA
            magnitude = np.dot(magnitude, dA)
            if len(sign.shape) < 2:
                sign = sign[..., np.newaxis]
            #print("mag: ", magnitude.shape)
            #print("sign: ", sign.shape)
            if False:
                out = -(magnitude / np.abs(magnitude)) * sign
                #out = np.dot(magnitude, sign)
                out = self.action_constraints[1] * np.sign(out).T
            else:
                #print("Action Constraints: ", self.action_constraints[1])
                #print("Sign: ", sign)
                #print("Yeah", self.action_constraints[1] * sign.T)
                out = self.action_constraints[1] * sign.T
            #if len(out.shape) < 2:
            #    out = out[..., np.newaxis]
            #print("OUT SHAPE: ", out.shape)
        elif (issubclass(type(self.model), GeneralSystemModel)):
            sign = self.compute_switch(xt)
            ds_dx = self.get_surface_d_dx(xt)
            #print("g: ", self.model.g)
            #print("DOT: ", np.linalg.det(np.dot(ds_dx.T, self.model.g)))
            #print("DOT: ", (np.dot(ds_dx.T, self.model.g)))
            magnitude = -(2 * np.linalg.inv(np.dot(ds_dx.T, self.model.g))) #TODO: Verify this step
            magnitude *= np.dot(ds_dx.T, self.model.f)
            if len(sign.shape) < 2:
                sign = sign[..., np.newaxis]
            #print("mag: ", magnitude)
            #print("sign: ", sign)
            #out = magnitude * sign
            if False:
                out = -(magnitude / np.abs(magnitude)) * sign
                #out = np.dot(magnitude, sign)
                out = self.action_constraints[1] * np.sign(out).T
                np.clip(out, -1e1, 1e1, out = out)
            else:
                #print("Action Constraints: ", self.action_constraints[1])
                #print("Sign: ", sign)
                #print("Yeah", self.action_constraints[1] * sign.T)
                out = self.action_constraints[1] * sign.T
            #if len(out.shape) < 2:
            #    out = out[..., np.newaxis]
            #print("OUT SHAPE: ", out)
        else:
            raise Exception("Unsupported Model for SMC")
        return out 

    def compute_switch(self, xt) -> int: 
        #print("X: %s Surface: %s " % (xt, self.surface))
        if self.switching_function == 'sign':
            return np.sign(np.dot(self.surface.T, xt))
        elif self.switching_function == 'arctan':
            return np.arctan(np.dot(self.surface.T, xt)) * 2 / np.pi

    def update_surface(self, xt):
        pass

    def get_surface_d_dx(self, xt):
        return self.surface #NOTE: assumes linear


class GD_SMC(SMC):
    '''Implementation of gradient-descent SMC, which modifies the sliding
    surface to ensure the system remains within the domain of attraction for some
    positive-eigenvalued surface, if such a surface is reachable.'''
    def __init__(self, alpha = 1e-2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        
    def update_surface(self, xt):
        umax = self.action_constraints[1]
        mag = np.ones(umax.shape) * float('inf')
        sigma = self.surface_base.copy()
        
        if (issubclass(type(self.model), LinearSystemModel)):
            hx = np.dot(self.model.A, xt)
            gx = self.model.B
        elif (issubclass(type(self.model), GeneralSystemModel)):
            hx = self.model.f.copy()
            if len(hx.shape) < 1:
                hx = hx[..., np.newaxis]
            gx = self.model.g.copy()
        #print("hx: ", hx.shape)
        #print("gx: ", gx.shape)
        i = 0
        while (umax < mag).any() and i < 20: #or .any()?!
            for c in range(sigma.shape[1]):
                sf = -np.dot(sigma[:,c].T, hx)
                sg = np.dot(sigma[:,c].T, gx[:,c])
                #sg_inv = np.linalg.inv(sg)
                sg_inv = 1/sg
                if len(sg.shape) < 1:
                    sg = sg[..., np.newaxis]
                if len(sf.shape) < 1:
                    sf = sf[..., np.newaxis]
                #if sf.size == 1 and sg.size == 1:
                #    sf = sf[0]
                #    sg = sg[0]
                    #mag = np.abs((sf / sg))
                else:
                    mag = np.abs(np.dot(sg_inv, sf))
                if not (umax < mag).any(): #or .any()?!
                    break

                if sf.size == 1 and sg.size == 1:
                    grad = hx * 1/sg 
                    gterm = -(1/sg * gx[:,c] * 1/sg) * sf
                    if len(gterm.shape) < 2 and len(grad.shape) > 1:
                        gterm = gterm[..., np.newaxis]
                    #grad += -(1/sg * gx[:,c] * 1/sg) * sf
                    #if gx[:,c].shape[0] == sf.shape[0]:
                    #    grad += -np.dot(1/sg * gx[:,c] * 1/sg,sf)
                    #else:
                    grad += (gterm)
                else:
                    gs = np.dot(gx[:,c], sg_inv)
                    grad = -np.dot(np.dot(sg_inv, gs), sf)
                    grad += np.dot(hx, sg_inv) 
                if len(grad.shape) > 1:
                    grad = grad.flatten()
                sigma[:,c] = sigma[:,c] - self.alpha * grad
                sigma[:,c] = np.clip(sigma[:,c], 1e-1, float('inf'))
            i += 1
        sigma += np.eye(sigma.shape[0], M = sigma.shape[1]) * 1e-1
        #print("GD SURFACE: ", sigma)
        self.surface = sigma


#class ISMC(ILQG, SMC):
#    '''Implementation of iterative sliding-mode control, modifying
#    the sliding surface a system is to be entrained upon to minimize
#    a differentiable cost function along a locally-linearized model
#    of the system-in-question.'''
#    def __init__(self, surface_base, switching_function,
#            *args, **kwargs):
#        
#        super().__init__(*args, **kwargs)
#        self.surface_base = surface_base
#        self.surface = self.surface_base
#        
#        self.surface_trajectory = []
#
#        self.switching_function = switching_function
#        
#        self.surface_ind = 0
#
#    
#    def update_surface(self, xt):
#        if self.surface_ind < len(self.surface_trajectory):
#            self.surface = self.surface_trajectory[self.surface_ind]
#            self.surface_ind += 1
#        else:
#            self.surface = self.surface_base.copy()
#
#    def initialize_surface_trajectory(self, xt) -> np.ndarray: 
#        self.surface_trajectory = np.array([self.surface_base.copy() for i in range(self.horizon)])
#
#    def initialize_control(self, xt, initial = 0.0):
#        self.surface_ind = 0
#        self.initialize_surface_trajectory(xt)
#        return self.surface_trajectory #we are TREATING trajectory as control iLQG is optimizing
#
#    def dq_dx(self, lx, lu, lxx, luu, lxu, fx, fu, fuu, fxx, Vx, Vxx, *args, **kwargs):
#        return lx + np.dot(Vx, fx) 
#    def dq_du(self, lx, lu, lxx, luu, lxu, fx, fu, fuu, fxx, Vx, Vxx, *args, **kwargs):
#        return lu + np.dot(Vx, fu) #lu[term] = 0]
#    def dq_dxx(self, lx, lu, lxx, luu, lxu, fx, fu, fuu, fxx, Vx, Vxx, *args, **kwargs):
#        return lxx + np.dot(fx.T, np.dot(Vxx, fx))
#    def dq_dxu(self, lx, lu, lxx, luu, lxu, fx, fu, fuu, fxx, Vx, Vxx, *args, **kwargs):
#        return lxu + np.dot(fu.T, np.dot(Vxx, fx))
#    def dq_duu(self, lx, lu, lxx, luu, lxu, fx, fu, fuu, fxx, Vx, Vxx, *args, **kwargs):
#        return luu + np.dot(fu.T, np.dot(Vxx, fu))
#    def dq_duu_inv(self, Quu, lamb = 1.0, LM_METHOD = True, *args, **kwargs):
#        if not LM_METHOD:
#            Quu_inv = np.linalg.pinv(Quu)
#        else:
#            try:
#                Quu_eval, Quu_evec = np.linalg.eig(Quu) 
#            except:
#                print("Quu: ", Quu)
#                input()
#            Quu_eval[Quu_eval < 0] = 0.0 #remove negative eigenvals
#            Quu_eval += lamb
#            #print("MODIFIED Quu Eigenvals: ", Quu_eval)
#            Quu_inv = np.dot(Quu_evec, np.dot(np.diag(1.0/Quu_eval),
#                Quu_evec.T)) #quadratic function with reciproc-eigvals
#            return Quu_inv       



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
    NONLINEAR_PENDULUM_TEST = False
    
    NONLINEAR_CARTPOLE_TEST = True

    ##NONLINEAR CARTPOLE TEST
    if NONLINEAR_CARTPOLE_TEST:
        lamb_factor = 10
        lamb_max = 1000
        horizon = 4
        initialization = 0.0
        #initialization = 1.0
        dt = 1e-2
        max_iterations = 10
        eps = 0.001
        
        state_shape = [1, 4]
        state_size = 4
        action_shape = [1]
        action_size = 1

        SECONDARY_STEP = False
        
        MPC_COMPARISON = True
        MPC_HORIZON = 0.15e0
        MPC_DT = dt
        MPC_STEPS = 1
        #MPC_STEPS = int(MPC_HORIZON / MPC_DT) - 1
        MPC_MAX_STEPS = int(horizon / dt)
        MPC_THRESHOLD = 0e-2
        #MPC_THRESHOLD = 0e-2
        MPC_MAX_ITER = 1
        
        #DDP_CLASS = 'ISMC'
        DDP_CLASS = 'ILQG'
        SMC_SURFACE_BASE = np.eye(state_size, M = action_size)
        #SMC_SWITCHING_FUNCTION = 'arctan'

        update_model = False
        FULL_ITERATIONS = True
        
        #
        
        #target = None
        target = np.array([0, 0, 0.0, 0], dtype = np.float64)
        #target = np.array([-1.0, 0, 0.7, 0], dtype = np.float64)
        #target = np.array([-1.0, 0, 0.0, 0], dtype = np.float64)
        x0 = np.array([0, .0, 1*np.pi, 0.1],dtype=np.float64)
        #x0 = np.array([0, 1, 0.0, 1.00],dtype=np.float64)
        #x0 = np.array([0, 1, 0, 5.00],dtype=np.float64)
        diff_func = lambda t,x : x - np.array([x[0], x[1], t[2], t[3]])
        #diff_func = lambda t,x : x - t
        #diff_func = lambda t,x : t - x
                   
        #def diff_func(t, x, null_ind = []):
        #    '''Null_ind sets the target at (i in null_ind)
        #    to equal x at (i in null_ind), effectively 
        #    "freeing" the system from controlling at that
        #    position.'''
        #    t[null_ind] = x[null_ind]
        #    return x-t
        #cost_func = lambda h,dt:1e4 * (5 * 1e-2) / (horizon * dt)
        null_ind = [0, 1, 3] #TODO: control INPUT, not ERROR?!
        cost_func = lambda h,dt:1e4
        #input("COST WEIGHT: %s" % (cost_func(horizon, dt)))
        #cost_func = lambda h,dt:1e4
        Q = np.eye(state_size) * cost_func(horizon, dt) * 1
        Q[2] *= 1
        #Qf = Q
        Qf = np.eye(state_size) * cost_func(horizon, dt) * 1e0
        R = np.eye(action_size) * 1e0

        priority_cost = True
        if priority_cost:
            for i in null_ind:
                Q[i][i] = np.sqrt(Q[i][i])
        else:
            for i in null_ind:
                Q[i][i] = 0
        #Q[0][0] /= 1e1 #set position Q term to 0 REEEEEE HAHAHAHAHHAA
        #Q[0][0] = 0 #set position Q term to 0 REEEEEE HAHAHAHAHHAA
        #Q[1][1] = 0 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
        #Q[3][3] = 0 #set dtheta/dt Q term to 0 REEEEEE HAHAHAHAHHAA
        #Q[2][2] = 0
        #Qf[1][1] = 0 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
        #Qf[1][1] = Qf[0][0] / 4 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
        cost = LQC(Q, R, Qf = Qf, target = target, 
                diff_func = diff_func)
        noisy_init = False
        friction = 0.000
        mc = 1
        mp = .1
        L = 1.0
        g = 9.8
        simplified_derivatives = False
        env = retrieve_control_environment('cartpole', 
                mc, mp, L, g,
                simplified_derivatives, 
                x0, 
                noisy_init = noisy_init, 
                interval = horizon, ts = dt, #THESE DON'T MATTER FOR THIS
                mode = 'point', 
                target = (target if target is not None else np.zeros((state_size)))) #unnecess
        env.error_func = lambda t,x : sum(x - np.array([x[0], x[1], t[2], t[3]]))

        model = ControlEnvironmentModel(env)
        action_constraints = None
        
        env.reset()
        env.state = x0.copy()
        #

        if DDP_CLASS == 'ILQG':
            ddp = ILQG(FULL_ITERATIONS,
                    lamb_factor,
                    lamb_max,
                    initialization,
                    state_shape, state_size, action_shape, action_size,
                    model, cost, None, action_constraints, #no noise model, iLQR
                horizon = int(horizon*1/dt),
                dt = dt,
                max_iterations = max_iterations, eps = eps,
                update_model = update_model)
        elif DDP_CLASS == 'ISMC':
            ddp = ISMC(SMC_SURFACE_BASE,
                    SMC_SWITCHING_FUNCTION,
                    True,
                    lamb_factor,
                    lamb_max,
                    initialization,
                    state_shape, state_size, action_shape, action_size,
                    model, cost, None, action_constraints, #no noise model, iLQR
                    horizon = int(horizon*1/dt),
                    dt = dt,
                    max_iterations = max_iterations, eps = eps,
                    update_model = update_model)
        #can't decouple Cartpole dynamics currently....... 
        #smc = ISMC(np.eye(state_size, M=action_size),
        #        state_shape, state_size, action_shape, action_size,
        #        model, cost, None, action_constraints, #no noise model, iLQR
        #        horizon = int(horizon*1/dt),
        #        dt = dt,
        #        max_iterations = max_iterations, eps = eps,
        #        update_model = update_model)
        #ilqg.set_smc(smc)

        X, U = ddp.step(x0)
        #U = [np.zeros(action_size) for i in range(int(horizon/dt))]
        print("FINAL U: ", U)
        env.state = x0.copy()
        env.set_target_point(target)
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
            #target = np.array([np.pi/2, 0])
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
            Qf = np.eye(state_size) * cost_func(horizon, dt) * 1e-1
            #R = np.eye(action_size) * cost_func(horizon, dt) * 0 #NO controls applied
            R = np.eye(action_size) * 1e4
            if priority_cost:
                for i in null_ind:
                    Q[i][i] = np.sqrt(Q[i][i])
            else:
                for i in null_ind:
                    Q[i][i] = 0
            cost = LQC(Q, R, Qf = Qf, target = target, 
                    diff_func = diff_func)

            FULL_ITERATIONS = FULL_ITERATIONS
            #FULL_ITERATIONS = True
            mpc_ilqg = ILQG(FULL_ITERATIONS,
                    lamb_factor,
                    lamb_max, 
                    initialization,
                    state_shape, state_size, action_shape, action_size,
                    model, cost, None, action_constraints, #no noise model, iLQR
                    horizon = int(MPC_HORIZON*1/MPC_DT),
                    dt = MPC_DT,
                    max_iterations = MPC_MAX_ITER, eps = eps,
                    update_model = update_model)
            env.reset()
            env.state = x0.copy()
            env.state_history.append(env.state.copy())
            xt = x0.copy()
            for i in range(MPC_MAX_STEPS):
                X, U = mpc_ilqg.step(xt)
                for j in range(MPC_STEPS):
                    u = U[j]
                    #U = [np.zeros(action_size) for i in range(int(horizon/dt))]
                    #u = np.clip(u, -1e1, 1e1)
                    env.step(u)
                xt = env.state.copy()
                #if abs(sum(diff_func(xt, target))) < MPC_THRESHOLD:
                diff = diff_func(xt, target)
                if  np.sqrt(cost.Q(diff)/cost_func(horizon, dt))  < MPC_THRESHOLD:
                    print("Early stopping condition met after %ss"% (i * dt))
                    print("Error: ", abs(sum(diff_func(xt, target))))
                    break
                #input("NEXT STEP!")
            print("Final State: ", env.state_history[-1])
            print("Target: ", target)
            env.generate_plots()
            input()
    ##Nonlinear (inverted pendulum) controls test
    if NONLINEAR_PENDULUM_TEST:
        lamb_factor = 10
        lamb_max = 1000
        horizon = 3
        initialization = 0.0
        #initialization = 1.0
        dt = 1e-2
        max_iterations = 10
        eps = 0.001

        SECONDARY_STEP = False
        
        update_model = False
        FULL_ITERATIONS = True
        
        MPC_COMPARISON = True
        MPC_HORIZON = 0.15e0
        MPC_DT = dt
        MPC_STEPS = 1
        #MPC_STEPS = int(MPC_HORIZON / MPC_DT) - 1
        MPC_MAX_STEPS = int(horizon / dt)
        MPC_THRESHOLD = 0e-2
        
        MPC_NOISY_OBS = False
        MPC_NOISY_MU = np.array([0, 0])
        MPC_NOISY_SIG = np.eye(2) * 5

        
        #
        state_shape = [1, 2]
        state_size = 2
        action_shape = [1]
        action_size = 1
        
        cost_func = lambda h,dt:1e3
        #input("COST WEIGHT: %s" % (cost_func(horizon, dt)))
        #cost_func = lambda h,dt:1e4
        null_ind = [1,]
        Q = np.eye(state_size) * cost_func(horizon, dt) * 1
        #Qf = Q
        Qf = np.eye(state_size) * cost_func(horizon, dt) * 1e3
        R = np.eye(action_size) * 1e1* 1
        
        priority_cost = True
        if priority_cost:
            for i in null_ind:
                Q[i][i] = np.sqrt(Q[i][i])
        else:
            for i in null_ind:
                Q[i][i] = 0
        #target = None
        target = np.array([0, 0], dtype = np.float64)
        #target = np.array([0.5, 0], dtype = np.float64)
        #target = np.array([np.pi, 0], dtype = np.float64)
        #target = np.array([np.pi/2, 0], dtype = np.float64)
        #target = np.array([np.pi/4, 0], dtype = np.float64)
        
        #x0 = np.array([0, np.pi/2],dtype=np.float64)
        x0 = np.array([0.0, np.pi/4],dtype=np.float64)
        #x0 = np.array([2*np.pi/4, -1.00],dtype=np.float64) #NOTE: don't do 
        #x0 = np.array([0.1, 0.10],dtype=np.float64)
        
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
        
        env.reset()
        env.state = x0.copy()
        #
        ilqg = ILQG(FULL_ITERATIONS,
                lamb_factor,
                lamb_max,
                initialization,
                state_shape, state_size, action_shape, action_size,
                model, cost, None, action_constraints, #no noise model, iLQR
                horizon = int(horizon*1/dt),
                dt = dt,
                max_iterations = max_iterations, eps = eps,
                update_model = update_model)

        X, U = ilqg.step(x0)
        #U = [np.zeros(action_size) for i in range(int(horizon/dt))]
        print("FINAL U: ", U)
        env.state = x0.copy()
        for u in U:
            env.step(u)
        print("ILQR Final State: ", env.state_history[-1])
        print("Target: ", target)
        env.generate_plots()
        input()
        if SECONDARY_STEP:
            print("Next: the return trip to initial position")
            xt = env.state_history[-1].copy()
            target = x0.copy()
            #target = np.array([0, 0])
            #target = np.array([np.pi, 0])
            #target = np.array([np.pi/2, 0])
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
            print("ILQR Final State: ", env.state_history[-1])
            print("Target: ", target)
            env.generate_plots()
            input()
        
        if MPC_COMPARISON:
            cost_func = lambda h,dt:1e5
            Q = np.eye(state_size) * cost_func(horizon, dt) * 1
            #Qf = Q
            Qf = np.eye(state_size) * cost_func(horizon, dt) * 0
            #R = np.eye(action_size) * cost_func(horizon, dt) * 0 #NO controls applied
            R = np.eye(action_size) * 1e2
            if priority_cost:
                for i in null_ind:
                    Q[i][i] = np.sqrt(Q[i][i])
            else:
                for i in null_ind:
                    Q[i][i] = 0
            cost = LQC(Q, R, Qf = Qf, target = target, 
                    diff_func = diff_func)
            mpc_ilqg = ILQG(FULL_ITERATIONS,
                    lamb_factor,
                    lamb_max, 
                    initialization,
                    state_shape, state_size, action_shape, action_size,
                    model, cost, None, action_constraints, #no noise model, iLQR
                    horizon = int(MPC_HORIZON*1/MPC_DT),
                    dt = MPC_DT,
                    max_iterations = max_iterations, eps = eps,
                    update_model = update_model)

            env.reset()
            env.state = x0.copy()
            env.state_history.append(env.state.copy())
            xt = x0.copy()
            for i in range(MPC_MAX_STEPS):
                if MPC_NOISY_OBS:
                    x_ob = xt + np.random.multivariate_normal(MPC_NOISY_MU, MPC_NOISY_SIG) * MPC_DT
                else:
                    x_ob = xt
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
            print("MPC Final State: ", env.state_history[-1])
            print("Target: ", target)
            env.generate_plots()
            input()


        
    
    ##Linear System test (linearized inverted pendulum)
    if LINEARIZED_PENDULUM_TEST: #TODO: wrap this shit in Unittest eventually
        lamb_factor = 5
        lamb_max = 1000
        horizon = 4
        initialization = 0.0
        #initialization = 1.0
        dt = 1e-2
        max_iterations = 50
        eps = 0.001
        
        MPC_COMPARISON = True
        MPC_HORIZON = 0.1e0
        MPC_DT = dt
        MPC_MAX_STEPS = int(horizon / dt) * 1
        MPC_THRESHOLD = 0e-3

        MPC_NOISY_OBS = True
        MPC_NOISY_MU = np.array([0, 0])
        MPC_NOISY_SIG = np.eye(2) / 8
        
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
        x0 = np.array([2*np.pi, 0.00],dtype=np.float64)
        #x0 = np.array([0.00, 0.00],dtype=np.float64)
        #
        FULL_ITERATIONS = True
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
        plt.plot(x[-1], y[-1], 'go')
        plt.plot(target[0], target[1], 'b^')
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
            R = np.eye(action_size) * 1e1 
            Q[1][1] = np.sqrt(Q[1][1]) #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
            #Q[1][1] = 0 #set velocity Q term to 0 REEEEEE HAHAHAHAHHAA
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
                if MPC_NOISY_OBS:
                    x_ob = xt + np.random.multivariate_normal(MPC_NOISY_MU, MPC_NOISY_SIG) * MPC_DT
                else:
                    x_ob = xt
                X, U = mpc_ilqg.step(x_ob)
                u = U[0]
                print("Next u: ", u)
                dx = model(xt, u)
                xt += dx * MPC_DT
                x_hist.append(xt.copy())
                #input("NEXT STEP!")
                if abs(sum(diff_func(xt, target))) < MPC_THRESHOLD:
                    print("Early stopping condition met after %ss"% (i * dt))
                    print("Error: ", abs(sum(diff_func(xt, target))))
                    break
            print("Final State: ", xt)
            print("Target: ", target)
            x = [s[0] for s in x_hist]
            y = [s[1] for s in x_hist]
            plt.plot(x,y, label='parametric curve')
            plt.plot(x[0], y[0], 'ro')
            plt.plot(x[-1], y[-1], 'go')
            plt.draw()
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
    
    ### differentiation w.r.t single LINEAR / AFFINE torch.nn weights
    #NOTE: With CONCATTED INPUTS!!! (forward dynamics - relevant)
    #linA_size = 5
    #linB_size = 3
    #indim = linA_size + linB_size
    #outdim = 1
    #inpA = torch.arange(linA_size, dtype = torch.float, requires_grad = True)
    #inpB = torch.arange(linB_size, dtype = torch.float, requires_grad = True)
    #W = torch.ones([1, indim], dtype = torch.float, requires_grad = True)
    #inp = torch.cat([inpA, inpB])
    #linear = torch.nn.Linear(indim, outdim, bias = False)
    #linear.weight.data = W
    #print("Linear weights: ", linear.weight)
    #out = linear(inp)
    #jac = dx(out, inp, create_graph = True)
    #print("OUT: ", out)
    #print('"Jacobian" : ', jac)
    #expected = torch.ones([outdim], dtype = torch.float)
    #loss = torch.nn.MSELoss()(out, expected)
    #lossA_dx = dx(loss, inpA)
    #lossB_dx = dx(loss, inpB)
    #loss_dx = dx(loss, inp)
    #loss_dW = dx(loss, linear.weight)
    #print('Loss : ', loss)
    #print('Loss d/dx (inpA): ', lossA_dx)
    #print('Loss d/dx (inpB): ', lossB_dx)
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
    #print("Inp: ", inp)
    #print("Hidden: ", h)
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
    

