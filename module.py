import torch
from torch.autograd import Variable
import numpy as np

import random as random
from functools import reduce

class PyTorchMLP(torch.nn.Sequential):
    def __init__(self, device, indim, outdim, hdims : [] = [], 
            activations : [] = [], initializer = None, batchnorm = False,
            bias = True, rec_size = 0, rec_type = 'gru', rec_batch = 1):
        super(PyTorchMLP, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hdims = hdims
        self.activations = activations
        #self.batchnorm = batchnorm
        self.device = device
        self.rec_size = rec_size
        self.rec_type = rec_type
        self.rec_batch = rec_batch
        assert(len(activations) == len(hdims) + 1)
        
        layers = []
        prev_size = indim
        for i in range(len(hdims)):
            linear = torch.nn.Linear(prev_size, hdims[i], bias = bias)
            layers.append(linear)
            if activations[i] is not None:
                active = self.create_activation(activations[i])
                if active is not None:
                    layers.append(active)
            prev_size = hdims[i]
        linear = torch.nn.Linear(prev_size, outdim, bias = bias)
        layers.append(linear)
        final_ind = len(hdims)
        if activations[final_ind] is not None:
            active = self.create_activation(activations[final_ind])
            if active is not None:
                layers.append(active)
        
        if rec_size > 0:
            prev_size = outdim #for...clarity's sake? 
            if rec_type == 'gru':
                self.rec = torch.nn.GRUCell(prev_size, rec_size)
            else:
                self.rec = torch.nn.LSTMCell(prev_size, rec_size)
            #self.reset_states()
         

        self.layers = torch.nn.ModuleList(layers)

    def reset_states(self):
        #self.hx = Variable(torch.zeros(self.rec_batch, self.rec_size), requires_grad = True).to(self.device)
        #self.cx = Variable(torch.zeros(self.rec_batch, self.rec_size), requires_grad = True).to(self.device)
        self.hx = (torch.zeros(self.rec_batch, self.rec_size, requires_grad = True)).to(self.device)
        self.cx = (torch.zeros(self.rec_batch, self.rec_size, requires_grad = True)).to(self.device)
    
    def create_activation(self, active):
        if active == 'relu':
            return torch.nn.LeakyReLU()
        elif active == 'sig':
            return torch.nn.Sigmoid()

    def forward(self, x, value_input = None):
        if len(x.shape) > 1:
            x = x.flatten()
        for l in range(len(self.layers)):
            x = self.layers[l](x)
        if self.rec_size > 0:
            if not hasattr(self, 'hx') or self.hx is None:
                self.reset_states()
            if self.rec_type == 'gru':
                #print("Hx: ", self.hx)
                #print("X: ", x.shape)
                x = self.rec(x.unsqueeze(0), self.hx)
                self.hx = x 
            else:
                x, cx = self.rec(x.unsqueeze(0), (self.hx, self.cx))
                self.hx = x
                self.cx = cx
        return x





#TODO: Consolidate these Modules and the modules used in clustered modelling
class PyTorchForwardDynamicsLinearModule(PyTorchMLP):
    '''Approximates the dynamics of a system . Composed of a shared
    feature extractor (PyTorchMLP) connected to two modules outputting f 
    and g values for the equation: x' = x + dt(f + g*u)'''
    def __init__(self, A_shape, B_shape, seperate_modules = False, linear_g = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.linear_g = linear_g
        self.seperate_modules = seperate_modules

        self.a_shape = A_shape
        self.a_size = reduce(lambda x,y:x*y, A_shape)
        self.b_shape = B_shape
        self.b_size = reduce(lambda x,y:x*y, B_shape)
        
        outdim = self.outdim if not self.rec_size > 0 else self.rec_size

        self.f_layer = torch.nn.Linear(outdim, A_shape[1], bias = True)
        if self.seperate_modules:
            self.g_module = PyTorchMLP(*args, **kwargs)
        if self.linear_g: #cannot have both, since PyTorchMLP is not guarenteed to be linear / non-affine
            self.g_layer = PyTorchMLP(self.device, B_shape[1],
                    B_shape[0], hdims = [100], 
                    activations = [None, None], bias = False, rec_size = 0)
                    #B_shape[0], hdims = [], 
                    #activations = [None], bias = False)
        else:
            self.g_layer = torch.nn.Linear(outdim, self.b_size, bias = True)

             

    def forward(self, x, *args, **kwargs):
        if type(x) == np.ndarray:
            x = torch.tensor(x, requires_grad = True, device = self.device).float()
        inp = x
        mlp_out = super().forward(inp)
        f = self.f_layer(mlp_out)
        if not self.seperate_modules and not self.linear_g:
            g = self.g_layer(mlp_out)
        elif self.seperate_modules:
            out = self.g_module(inp)
            g = self.g_layer(out)
        elif self.linear_g:
            g = self.du(x, u = np.zeros((1, self.b_shape[1]))) 
        if f.shape[0] != self.a_shape[0]:
            f = f.t()
        return f, g
    
    def du(self, xt, u = None, create_graph = True):
        #dm_du = grad(self.g_layer, u, create_graph = False)
        #dm_du = dm_du.detach().numpy()
        dm_du = np.eye(self.b_shape[1])
        for l in (self.g_layer.layers):
            #print("LAYER: ", l)
            #print("Weight Shape", l.weight.shape) 
            if self.device == torch.device('cuda'):
                weight = l.weight.cpu().detach().numpy()
            else:
                weight = l.weight.detach().numpy()
            dm_du = np.dot(weight, dm_du) 
        return dm_du

        
class PyTorchLinearSystemDynamicsLinearModule(PyTorchMLP):
    '''.'''
    def __init__(self, A_shape, B_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_shape = A_shape
        self.a_size = reduce(lambda x,y:x*y, A_shape)
        self.b_shape = B_shape
        self.b_size = reduce(lambda x,y:x*y, B_shape)
        outdim = self.outdim if not self.rec_size > 0 else self.rec_size
        self.a_module = torch.nn.Linear(outdim, self.a_size, bias = True)
        self.b_module = torch.nn.Linear(outdim, self.b_size, bias = True)

    def forward(self, x, *args, **kwargs):
        if type(x) == np.ndarray:
            x = torch.tensor(x, requires_grad = True, device = self.device).float()
            #x = x.unsqueeze(0)
        if type(x) == torch.Tensor and len(x.size()) > 1:
            x = x.squeeze(0)
        #print("X: ", x)    
        #print("U: ", u)    
        #inp = torch.cat((x, u), 0).float()
        out = super().forward(x)
        a_ = self.a_module(out).clamp(min=-1e3, max=1e3)
        b_ = self.b_module(out).clamp(min=-1e3, max=1e3)
        #print("A: ", a_)
        #print("B: ", b_)
        return a_, b_ #NOTE: flattened version
    

class PyTorchLinearAutoencoder(torch.nn.Module):
    def __init__(self, indim, device, depth, activations : [] = [], encoded_activations = [], 
            reduction_factor = 0.75, uniform_layers = False):
        '''Arguments:
            @activations: Determines the activations at each layer (?except the first and last layer?).
            @Depth: Determines the number of layers in the encoder / decoder layer.
            @Reduction_factor: Determines how much smaller / larger each layer is from the last. This
            also informs the size of the encoded space, which equals indim * floor((factor)^depth)'''
        super(PyTorchLinearAutoencoder, self).__init__()
        self.activations = activations
        self.encoded_activations = encoded_activations
        self.reduction = reduction_factor
        self.encoded_space = math.floor(indim * self.reduction ** depth)
        self.depth = depth
        print("DEPTH: ", self.depth)
        print("Indim: ", indim)
        self.device = device
        self.indim = indim
        #assert(len(activations) == len(encoder_layers + decoder_layers) + 1)
        
        layers = []
        prev_size = indim
        ## create encoder
        for i in range(depth):
            if uniform_layers:
                size = prev_size
            elif i < depth - 1:
                size = math.floor(prev_size * self.reduction) 
            if i >= depth - 1:
                size = math.floor(indim * self.reduction ** depth)
            linear = torch.nn.Linear(math.floor(prev_size), size, bias = True)
            layers.append(linear)
            functions = activations if i < depth - 1 else encoded_activations
            for a in functions:
                if a == 'relu':
                    layers.append(torch.nn.LeakyReLU())
                elif a == 'sig':
                    layers.append(torch.nn.Sigmoid())
            if not uniform_layers:
                prev_size = prev_size * self.reduction #necessary to carry floating point
        layers = [l.to(device) for l in layers]
        self.encoder = torch.nn.ModuleList(layers)
        ##create decoder
        layers = []
        for i in range(depth):
            if uniform_layers:
                if i == 0:
                    size = prev_size
                    prev_size = math.floor(indim * self.reduction ** depth)
                else:
                    size = indim
                    prev_size = size
            elif i < depth - 1:
                size = math.floor(prev_size / self.reduction) 
            else:
                size = indim
            linear = torch.nn.Linear(math.floor(prev_size), size, bias = True)
            layers.append(linear)
            functions = activations if i < depth - 1 else encoded_activations
            for a in activations:
                if i >= depth - 1: #output should HAVE NO NONLINEARITY?
                    break
                if a == 'relu':
                    layers.append(torch.nn.LeakyReLU())
                elif a == 'sig':
                    layers.append(torch.nn.Sigmoid())
            if not uniform_layers:
                prev_size = prev_size / self.reduction #necessary to carry floating point
        layers = [l.to(device) for l in layers]
        self.decoder = torch.nn.ModuleList(layers)
    
    def encode(self, x):
        if len(self.encoder) > 0:
            for l in range(len(self.encoder)):
                x = self.encoder[l](x)
            #print("Len: %s Encoded Space: %s" % (len(x), self.encoded_space))
            assert(len(x) == self.encoded_space)
        return x

    def decode(self, x):
        #print("Inp: ", len(x))
        #print("Encoded: ", self.encoded_space)
        if len(self.decoder) > 0:
            assert(len(x) == self.encoded_space)
            for l in range(len(self.decoder)):
                x = self.decoder[l](x)
        return x

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
            


#class PyTorchLinearUnetEncoder(torch.nn.Module):
#    def __init__(self, indim, device, depth, activations : [] = [], encoded_activations = [], 
#            reduction_factor = 0.75):
#        '''Arguments:
#            @activations: Determines the activations at each layer (?except the first and last layer?).
#            @Depth: Determines the number of layers in the encoder / decoder layer.
#            @Reduction_factor: Determines how much smaller / larger each layer is from the last. This
#            also informs the size of the encoded space, which equals indim * floor((factor)^depth)'''
#        self.activations = activations
#        self.encoded_activations = encoded_activations
#        self.encoded_space = indim * math.floor(reduction ** depth)
#        self.depth = depth
#        self.reduction = reduction_factor
#        self.device = device
#        self.indim = indim
#        #assert(len(activations) == len(encoder_layers + decoder_layers) + 1)

def LinearSAAutoencoder(encoder_base, state_size, action_size, forward_mlp,
        coupled_sa = False,
        forward_dynamics = False, *args, **kwargs):
    '''Construct an Autoencoder, encoding State/Action pairs into some space, 
    and decoding the same state/action pair from that space. 
    
    Arguments: 
        @coupled_sa: Indicates whether or not the State/Action encoding should be
        coupled or decoupled. 
        @forward_state: Determines whether or not the autoencoder also outputs
        x_t+1, computed from the (xt, at).
        @forward_dynamics: Determines whether or not the autoencoder encodes
        x_t+1 = xt + f(encoded(xt, at)), the next state is represented as 
        some additive term.
        @linear_forward: Imposes a limitation in which the forward_dynamics are
        computed linearly: x_t+1 = xt + A*encoded_state(xt, at) + B*encoded_action(xt, at)
        This limitation should be investigated for the potential of imposing linearity on the
        inner space.'''
    class PyTorchLinearSAAutoencoder(encoder_base):
        def __init__(self, state_size, action_size, forward_mlp, 
                coupled_sa = False, forward_dynamics = False, 
                *args, **kwargs):
            self.forward_dynamics = forward_dynamics
            self.coupled_sa = coupled_sa

            self.action_size = action_size
            self.state_size = state_size
            
            indim = state_size if not coupled_sa else state_size + action_size
            if coupled_sa:
                super().__init__(indim, *args, **kwargs)
            else:
                super().__init__(indim, *args, **kwargs) #treat encode, decode as SPECIFICALLY for state
                if action_size == 1: #identity module
                    a = [args[0], 0]
                    a.extend(args[2:])
                    print('ARGS: ', args)
                    print("A: ", a)
                    args = a
                #else:
                self.action_ae = encoder_base(action_size, *args, **kwargs) #create identical second encoder for actions
            
            self.forward_mlp = forward_mlp #LINEAR MLP structure if the desire is to constrain encoded space to linear function
             

        def forward(self, s, a):
            if self.coupled_sa:
                inp = torch.cat((s, a), 0) 
                decoded = super().forward(inp)
                #seperate s / a
                decoded_state, decoded_action = self.separate_decoded(decoded)
            else:
                decoded_state = super().forward(s)
                decoded_action = self.action_ae(a)
            return decoded_state, decoded_action    

        def separate_encoded(self, e):
            assert(self.coupled_sa)
            encoded_state_size = math.floor(self.state_size * self.reduction**self.depth)
            return e[:encoded_state_size], e[encoded_state_size:]
            #raise Exception('self.state_size * REDUCTION**DEPTH, self.action-size * REDUCTION**DEPTH')

        def separate_decoded(self, d):
            assert(self.coupled_sa)
            return d[:self.state_size], d[self.state_size:]

        def forward_encode(self, s, a, detach = False):
            '''Maps ENCODED s / a to an ENCODED s_t+1.'''
            if self.coupled_sa:
                inp = torch.cat((s, a), 0) 
                encoded = super().encode(inp)
                s_, a_ = self.separate_encoded(encoded) 
            else:
                s_ = super().encode(s)
                a_ = self.action_ae.encode(a)
            if detach:
                s_ = s_.detach()
                a_ = a_.detach()
            inp = torch.cat((s_, a_), 0) 
            f = self.forward_mlp(inp) 
            if self.forward_dynamics:
                return s_ + f
            else:
                return f

        def forward_predict(self, s, a, detach = True):
            '''Maps full s/a to s_t+1 (not encoded s_t+1)'''
            f = self.forward_encode(s, a, detach)      
            if self.coupled_sa:
                inp = torch.cat((f, a), 0) 
                #print("Len: %s Encoded Space: %s" % (len(inp), self.encoded_space))
                decoded = self.decode(inp)
                s_d, a_d = self.separate_decoded(decoded)
                return s_d
            else:
                decoded_state = super().decode(f)
                return decoded_state

    return PyTorchLinearSAAutoencoder(state_size, action_size, 
            forward_mlp, coupled_sa, forward_dynamics,
            *args, **kwargs)



class PyTorchDiscreteACMLP(PyTorchMLP):
    '''Adds action / value heads to the end of an MLP constructed
    via PyTorchMLP '''
    def __init__(self, action_space, seperate_value_module = None,
            seperate_value_module_input = False, 
            value_head = True, 
            action_bias = True, value_bias = True,
            *args, **kwargs):
        self.action_space = action_space
        super(PyTorchDiscreteACMLP, self).__init__(*args, **kwargs)

        outdim = self.outdim if not self.rec_size > 0 else self.rec_size

        self.seperate_value_module_input = seperate_value_module_input
        self.sigma_head = sigma_head
        if seperate_value_module is not False: #currently, this creates an IDENTICAL value function
            print("seperate_value_module")
            if seperate_value_module is None:
                print("self.value_mlp = PyTorchMLP")
                self.value_mlp = PyTorchMLP(*args, **kwargs)
            else:
                self.value_mlp = seperate_value_module
        if seperate_value_module is False or value_head is True: #we don't need this as an attribute, do we?
            self.value_module = torch.nn.Linear(outdim, 1, 
                    bias = value_bias)
        self.action_module = torch.nn.Linear(outdim, 
                action_space, bias = action_bias)
    
    def forward(self, x, value_input = None):
        mlp_out = super(PyTorchDiscreteACMLP, self).forward(x)
        #print("MLP OUT: ", mlp_out)
        actions = self.action_module(mlp_out) 
        action_scores = torch.nn.functional.softmax(actions, dim=-1)
        if self.value_mlp is not False:
            if self.seperate_value_module_input == True:
                mlp_out = self.value_mlp.forward(value_input)
            else:
                mlp_out = self.value_mlp.forward(x)
        if hasattr(self, 'value_module'):
            value = self.value_module(mlp_out)
        #print("ACTION: %s VALUE: %s" % (actions, value))
        return action_scores, value

class PyTorchContinuousGaussACMLP(PyTorchMLP):
    '''Adds action (mean, variance) / value heads 
    to the end of an MLP constructed via PyTorchMLP '''
    def __init__(self, action_space, seperate_value_module = None,
            seperate_value_module_input = False, value_head = True, 
            action_bias = True, value_bias = True, sigma_head = False,
            *args, **kwargs):
        self.action_space = action_space
        super(PyTorchContinuousGaussACMLP, self).__init__(*args, **kwargs)
        outdim = self.outdim if not self.rec_size > 0 else self.rec_size

        self.seperate_value_module_input = seperate_value_module_input
        self.sigma_head = sigma_head
        if seperate_value_module is not False: #currently, this creates an IDENTICAL value function
            print("seperate_value_module")
            if seperate_value_module is None:
                print("self.value_mlp = PyTorchMLP")
                self.value_mlp = PyTorchMLP(*args, **kwargs)
            else:
                self.value_mlp = seperate_value_module
        if seperate_value_module is False or value_head is True: #we don't need this as an attribute, do we?
            self.value_module = torch.nn.Linear(outdim, 1, 
                    bias = value_bias)
        self.action_mu_module = torch.nn.Linear(outdim, 
                action_space, bias = action_bias)
        self.action_sigma_module = torch.nn.Linear(outdim,
                1, bias = action_bias)
        self.action_mu_tanh = torch.nn.Tanh()
        self.action_sigma_softplus = torch.nn.Softplus()
    
    def forward(self, x, value_input = None):
        mlp_out = super(PyTorchContinuousGaussACMLP, self).forward(x)
        #print("MLP OUT: ", mlp_out)
        action_mu = self.action_mu_module(mlp_out) 
        action_mu = self.action_mu_tanh(action_mu)
        if self.sigma_head:
            action_sigma = self.action_sigma_module(mlp_out)
            action_sigma = self.action_sigma_softplus(action_sigma) + 0.01 #from 1602.01783 appendix 9
        else: #assume sigma is currently zeros
            action_sigma = torch.ones([1, self.action_space], dtype=torch.float32) * 0.5
        if hasattr(self, 'value_mlp'):
            if self.seperate_value_module_input == True:
                mlp_out = self.value_mlp.forward(value_input)
            else:
                mlp_out = self.value_mlp.forward(x)
        if hasattr(self, 'value_module'):
            value = self.value_module(mlp_out)
        else:
            assert(hasattr(self, 'value_mlp'))
            value = mlp_out #assuming value_mlp creates value
        #print("ACTION: %s VALUE: %s" % (actions, value))
        return action_mu, action_sigma, value

#class PyTorchDynamicsMLP(PyTorchMLP):
#    def __init__(self, state_dim, *args, **kwargs):
#        super(PyTorchDynamicsMLP, self).__init__(*args, **kwargs)
#        self.state_dim = state_dim



def EpsGreedyMLP(mlp_base, eps, eps_decay = 1e-3, eps_min = 0.0, action_constraints = None, 
        *args, **kwargs):
    class PyTorchEpsGreedyModule(mlp_base):
        def __init__(self, mlp_base, eps, eps_decay, eps_min, action_constraints = None, 
                *args, **kwargs):
            self.eps = eps
            self.decay = eps_decay
            self.eps_min = eps_min
            self.base = mlp_base
            self.action_constraints = action_constraints
            super(PyTorchEpsGreedyModule, self).__init__(*args, **kwargs)
    
        def update_eps(self):
            self.eps = max(self.eps_min, self.eps - self.decay)
            #print("EPS: ", self.eps)

        def forward(self, x, value_input = None):
            if self.base == PyTorchDiscreteACMLP: #TODO: can't assume value function exists...??
                action_score, values = super(PyTorchEpsGreedyModule,
                        self).forward(x, value_input)
                if random.random() < self.eps:
                    self.update_eps()
                    with torch.no_grad(): #no gradient for this
                        action = random.choice([0, self.action_space - 1])
                        action_score = torch.tensor( \
                                np.eye(self.action_space)[action],
                                    device = self.device).float()
                        #print("A: %s Score: %s" % (action, action_score))
                return action_score, values
            elif self.base == PyTorchContinuousGaussACMLP:
                action_mu, action_sigma, values = super(PyTorchEpsGreedyModule,
                        self).forward(x, value_input)
                if random.random() < self.eps: #randomize sigma values
                    self.update_eps()
                    mins = self.action_constraints[0]
                    maxs = self.action_constraints[1]
                    action_mu =  np.random.uniform(mins, maxs, (self.action_space))
                    #action_sigma = np.random.random_integers(1, high = 2, size = (1, self.action_space))
                    #eps_sigma = np.random.uniform(low = 0.01, high = 3.0, size = (1, 1))
                    action_mu = torch.as_tensor(action_mu).float()
                    #noise = action_sigma.clone().uniform_(0, 3)
                    noise = action_sigma.clone().uniform_(0, 3)
                    #print("Current sigma: ", action_sigma)
                    action_sigma += noise
                    #print("New Sigma: %s Eps: %s"%(action_sigma, self.eps))
                return action_mu, action_sigma, values 
                raise Exception("Figure this out?")

    return PyTorchEpsGreedyModule(mlp_base, eps, eps_decay, eps_min, action_constraints, *args, **kwargs)


