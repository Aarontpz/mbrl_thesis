from control_environment import *

import numpy as np

import torch
from torch.autograd import grad
from torchviz import make_dot

def dx(f, x, create_graph = True):
    return grad(f, x, create_graph = create_graph)


class LQG:
    def __init__():
        pass

class ILQG:
    def __init__(env : ControlEnvironment, initial_ut = 0.0):
        self.env = env
        self.initial_ut = initial_ut #for scalining generated trajectories

    def step(self, xt) -> np.ndarray:
        pass

    def initialize_trajectory(self, xt) -> np.ndarray:
        return np.ones()  * self.initial_ut

#class Controller:
#    def __init__():
#        pass

def MPCController(control_base, *args, **kwargs):
    pass



if __name__ == '__main__':
     

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
    

