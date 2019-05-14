import abc
import numpy as np
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
    def update(self, xt, ut):
        pass

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



if True:
    import torch
    from torch.autograd import grad
    from torchviz import make_dot
    
    ## PyTorch helper functions
    def dx(f, x, create_graph = True):
        return grad(f, x, create_graph = create_graph)
    
    def dxx(f, x, create_graph = True):
        pass


    class PyTorchDecoupledSystemModel(Model):
        pass

    ##

class LinearQuadraticModel(Model):
    '''Generic linear quadratic model to use with linear quadratic
    functions.'''
    def __init__(self, Q):
        self.Q = Q
        self.shape = Q.shape
        self.size = Q.size
    def d_dx(self, xt, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        #return 2*np.dot(xt, self.Q)
        return np.dot(xt, (self.Q.T + self.Q))
    def d_dxx(self, xt, ut=None, dt=None, *args, **kwargs):
        '''Calculate df^2/(dxdx), given timestep
        size dt'''
        return 2*self.Q #TODO TODO confirm this
    def __call__(self, xt=None, ut=None, dt=None, *args, **kwargs):
        '''Returns dx, the change in state xt given control ut
        and timestep size dt.'''
        #NOTE: xt is a ROW VECTOR   
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
            Qf = None, target = None, diff_func =(lambda t,x:x - t)):
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
        #return np.zeros((self.Q.Q[0].size, self.R.Q[0].size))
        return np.zeros((self.R.Q[0].size, self.Q.Q[0].size))
    def __call__(self, xt, ut, dt=None, *args, **kwargs):
        if dt is None:
            dt = 1.0
        if self.target is not None:
            target = self.target
            if len(target.shape) > 1:
                target = target.flatten()
            xt = self.diff_func(target, xt)
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
        #print("xt shape: ", xt.shape)
        #print("ut shape: ", ut.shape)
        #input()
        #print("Q(xt): %s \n R(ut): %s" % (self.Q(xt), self.R(ut)))
        #print(self.Q(xt))
        #print(self.R(ut))
        return 0.5 * (self.Q(xt) + self.R(ut)) * dt



class LinearSystemModel(Model):
    '''Model a system as a linear relationship in the state-transition
    equation'''
    def __init__(self, A : np.ndarray = None, B : np.ndarray = None):
        self.A = A
        self.B = B
    def d_dx(self, xt, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        assert(dt is not None)
        return self.A * dt
        #return self.A * dt
    def d_du(self, xt = None, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        assert(dt is not None)
        assert(ut is not None)
        return self.B * dt
    def __call__(self, xt, ut, dt=None, A = None, B = None, *args, **kwargs):
        A = A if A is not None else self.A
        B = B if B is not None else self.B
        print("B: %s ut: %s" % (B.shape, ut.shape))
        ax = np.dot(A, xt)
        if len(ut.shape) == 1 and len(B.shape) == 1:
            bu = B * ut
        else:
            bu = np.dot(B, ut)
        if len(ax.shape) < 2:
            ax = ax[..., np.newaxis]
        if len(bu.shape) < 2:
            bu = bu[..., np.newaxis]
        print("A*xt: ",ax.shape)
        print("B*ut: ",bu.shape)
        print("A*xt + B*ut: ",(ax+bu).shape)
        return ax + bu


class FiniteDifferencesModel(LinearSystemModel):
    '''Just...for comparison's sake, if it's EVEN possible with
    the environments I'm working with...'''
    pass

class GeneralSystemModel(Model):
    '''Model a system as a "general" relationship which assumes only
    a linear control relationship: x' = f(x,t) + g(x,t)*u, where
    f has no guarentees on linearity.'''
    def __init__(self, f : np.ndarray = None, g : np.ndarray = None):
        self.f = f
        self.g = g
    def d_dx(self, xt, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        raise Exception("This needs to be considered carefully.")
        assert(dt is not None)
        return self.f * dt
    def d_du(self, xt = None, ut=None, dt=None, *args, **kwargs):
        '''Calculate df/dx, given timestep
        size dt'''
        assert(dt is not None)
        assert(ut is not None)
        return self.g * dt
    def __call__(self, xt, ut, dt=None, f = None, g = None, *args, **kwargs):
        f = f if f is not None else self.f
        g = g if g is not None else self.g
        print("g: %s ut: %s" % (g.shape, ut.shape))
        if len(ut.shape) == 1 and len(g.shape) == 1:
            gu = g * ut
        else:
            gu = np.dot(g, ut)
        if len(f.shape) < 2:
            f = f[..., np.newaxis]
        if len(gu.shape) < 2:
            gu = gu[..., np.newaxis]
        print("f shape: ",f.shape)
        print("g shape: ",gu.shape)
        print("f + g*u: ",(f+gu).shape)
        return f + gu
