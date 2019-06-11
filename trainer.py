# Author: Aaron Parisi
# 11/20/18
from agent import Agent, PyTorchModel
import abc

import torch
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from torchviz import make_dot

import random

class Trainer:
    __metaclass__ = abc.ABCMeta
    '''Baseclass for RL Trainers; these assume the agents are well-suited
    for their respective tasks and handle the implementation of the 
    partiuclar learning algorithm.
    
    Args: *replay: not None if random trajectory sampling is enabled
    (to reduce variance / biases from starting conditions)'''
    def __init__(self, agent : Agent, env, 
            replay = 10, max_traj_len = 30, 
            gamma = 0.98, num_episodes = 2, *args, **kwargs):
        self.agent = agent
        self.env = env
        self.replay = replay
        self.num_episodes = num_episodes

        self.max_traj_len = max_traj_len
        self.gamma = gamma
        self.action_losses = []
        self.value_losses = [] #NOTE: may remain empty
        self.action_entropies = [] 

    @abc.abstractmethod 
    def step(self):
        pass

    @abc.abstractmethod
    def get_discounted_reward(self, start, end = None) -> np.ndarray:
        pass

    def get_episode_ends(self):
        terminal_history = self.agent.terminal_history
        ends = [i for i, x in enumerate(terminal_history) if x]
        return ends

    def get_trajectory_end(self, start, end = None):
        terminal_history = self.agent.terminal_history
        max_ind = len(terminal_history)
        if end is None:
            end = min(max_ind, start + self.max_traj_len)
        if True in terminal_history[start:end]:
            end = terminal_history[start:].index(True) + start
            #TODO: consider minimal traj length too?
        return end

    @abc.abstractmethod
    def compute_action_penalty(self, start):
        pass
    @abc.abstractmethod
    def report(self):
        ''' '''
        pass 

class Dataset:
    '''By default Dataset acts as if it is storing samples from an MDP'''
    def __init__(self, aggregate_examples = False, correlated_samples = True):
        self.samples = []
        self.aggregate_examples = aggregate_examples
        self.correlated_samples = correlated_samples

        self.ind = 0
    
    def sample(self, batch_size = None):
        batch = None
        if batch_size == None:
            batch_size = 0
        if not self.correlated_samples:
            batch = random.sample(self.samples, batch_size + 1)  
        else:
            ind = random.choice(range(len(self.samples)))
            end = min(len(self.samples) - 1, ind + batch_size)
            print("Start: %s End: %s" % (ind, end))
            batch = self.samples[ind : end]
        return batch
        

    def trainer_step(self, trainer):
        samples = []
        sample = []
        agent = trainer.agent
        for i in range(len(agent.reward_history) - 1): #-1 for terminal state
            sample.append(agent.state_history[i])
            action = agent.action_history[i]
            if type(action) == torch.Tensor:
                action = action.detach()
            sample.append(action)
            sample.append(agent.reward_history[i])
            sample.append(agent.state_history[i+1])
            samples.append(sample)
            sample = []
        if self.aggregate_examples:
            self.samples.extend(samples)
        else:
            self.samples = samples

    def get_sample_structure(self) -> str:
        return 'sk, ak, rk, sk+1'

#TODO TODO TODO: accomodate multiple-episodes? Is this necessary for the purpose that
#dataset serves (supervised vs reinforcement learning)

class DAgger(Dataset):
    '''When sampling, samples from PREVIOUS episode / trainer step
    with probability recent_prob, else samples from AGGREGATE.'''
    def __init__(self, max_samples = float('inf'), recent_prob = 0.5, *args, **kwargs):
        self.aggregate = []
        self.recent_prob = recent_prob
        super().__init__(*args, **kwargs)
        self.aggregate_examples = False #because we AUTOMATICALLY aggregate samples lol

        self.max_samples = max_samples

    def sample(self, batch_size = None):
        sample_recent = random.random() < self.recent_prob
        if len(self.aggregate) >= batch_size:
            samples = self.samples if sample_recent else self.aggregate
        else:
            samples = self.samples
        #print("Sample Recent: ", sample_recent)
        batch = None
        if batch_size == None:
            batch_size = 0
        if self.correlated_samples:
            if batch_size > len(samples): #to do full runs
                return self.samples
            ind = random.choice(range(len(samples)))
            end = min(len(samples) - 1, ind + batch_size)
            print("Start: %s End: %s" % (ind, end))
            batch = samples[ind : end]
        else:
            batch = random.sample(samples, batch_size + 1)  
        return batch

    def trainer_step(self, trainer):
        self.aggregate.extend(self.samples) #add PREVIOUS self.samples to aggregate
        if len(self.aggregate) > self.max_samples:
            self.aggregate = self.aggregate[self.max_samples:]
            print("Aggregate Size: ", len(self.aggregate))
        super().trainer_step(trainer)


class PyTorchTrainer(Trainer):
    def __init__(self, device, optimizer, scheduler = None, *args, **kwargs):
        self.device = device
        self.opt = optimizer
        self.scheduler = scheduler
        super(PyTorchTrainer, self).__init__(*args, **kwargs)
        
    def get_discounted_rewards(self, rewards, scale = True) -> torch.tensor:
        '''Generate list of discounted rewards. This is an improvement on the previous,
        INCREDIBLY inefficient method of generating discounted rewards previously used.
        Source: https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf'''
        #print("REWARDS BEFORE: ", rewards)
        R = 0
        disc_rewards = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            disc_rewards.insert(0, R)
        rewards = torch.FloatTensor(disc_rewards)
        if scale: #normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        #print("REWARDS AFTER: ", rewards)
        return rewards
    
    def get_sample_s(self, s):
        return s[0]
    def get_sample_a(self, s):
        return s[1]
    def get_sample_r(self, s):
        return s[2]
    def get_sample_s_(self, s):
        return s[3]

    
class PyTorchPolicyGradientTrainer(PyTorchTrainer):
    def __init__(self, value_coeff = 0.1, entropy_coeff = 0.0, 
            entropy_bonus = False, energy_coeff = 0.0, *args, **kwargs):
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.entropy_bonus = entropy_bonus
        self.energy_coeff = energy_coeff
        super(PyTorchPolicyGradientTrainer, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def compute_action_loss(self, advantage, ind, *args, **kwargs):
        pass

    def compute_value_loss(self, rewards, ind, bootstrap = True, *args, **kwargs):
        value_criterion = torch.nn.MSELoss()
        state = self.agent.state_history[ind]
        action, value, normal = self.agent.evaluate(state) 
        #value = self.agent.value_history[ind] 
        if bootstrap:
            state_ = self.agent.state_history[ind+1]
            r = self.agent.reward_history[ind]
            _, value_, __ = self.agent.evaluate(state_) 
            value_loss = self.value_coeff * value_criterion(value, r + self.gamma*value_) 
        else:
            value_loss = self.value_coeff * value_criterion(value, rewards[ind]) 
        return value_loss

    @abc.abstractmethod
    def compute_entropy_loss(self, ind, *args, **kwargs):
        pass

    @abc.abstractmethod
    def compute_advantages(self, rewards, values, gamma = 0.99, normalize = True):
        advantages = torch.tensor([rewards[i] - values[i].detach() for i in range(len(values))], device = self.device) 
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        return advantages

    def step(self):
        super(PyTorchPolicyGradientTrainer, self).step()
        requires_grad = True
        reward = None
        ## Porting this function into the ACTrainer class
        module = self.agent.module
        device = self.device
        optimizer = self.opt
        action_scores = self.agent.action_score_history
        value_scores = self.agent.value_history
        assert(len(value_scores) == len(action_scores)) #necessary
        
        if hasattr(module, 'rec_size') and module.rec_size > 0: 
            module.reset_states() 
        #for score in action_score_history:
        #    make_dot(score).view()
        #    input()
        if self.max_traj_len > len(action_scores)/self.num_episodes: #if avg episode len < max traj len, run through each episode
            print("Max Len: %s Avg Len: %s" % (self.max_traj_len, len(action_scores)/self.num_episodes))
            ends = self.get_episode_ends()
            replay_starts = [0]
            for e in ends[:len(ends) - 1]:
                replay_starts.append(e + 1)
        else:
            replay_starts = [random.choice(range(len(action_scores))) for i in range(self.replay)] #sample replay buffer "replay" times 
        net_action_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        net_value_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        net_entropy_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        net_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        rewards = self.get_discounted_rewards(self.agent.reward_history[:], scale = False).to(device)
        #advantages = self.compute_advantages(rewards, value_scores, self.gamma, False)
        advantages = self.compute_advantages(self.agent.reward_history[:], 
                value_scores, self.gamma, normalize = False, compute_values = False) #generalized advantage
        
        run = 0
        for start in replay_starts:
            if hasattr(module, 'rec_size') and module.rec_size > 0: 
                module.reset_states() 
            end = self.get_trajectory_end(start, end = None)
            #print("END: ", end)
            loss = torch.tensor([0.0], requires_grad = requires_grad).to(device) #reset loss for each trajectory
            #rewards = self.get_discounted_rewards(self.agent.reward_history[start:end], scale = True).to(device)
            print("START: %s END: %s" % (start, end))
            for ind in range(start, end - 1): #-1 because terminal state doesn't matter?
                #print("Rewards: ", len(rewards))
                #print("Ind: ", ind)
                #print("End: ", end)
                #if module.lstm_size > 0:
                #    obs = get_observation_tensor(state_history[i])
                #    _ = module.forward(obs)
                R = rewards[ind - start]
                #if run == 0: #run A2C update instead of PPO, even if PPO trainer
                #    action_loss = PyTorchACTrainer.compute_action_loss(self, R, ind)
#class PyTorchAC#Trainer(PyTorchPolicyGradientTrainer):
                #else:
                #action_loss = self.compute_action_loss(R, ind)
                advantage = advantages[ind]
                action_loss = self.compute_action_loss(advantage, ind)
                value_loss = self.compute_value_loss(rewards, ind, bootstrap = True)
                #if self.agent.module.value_module is not False:
                #    pass
                if hasattr(self.agent, 'energy_history'):
                    action_penalty = self.agent.energy_history[ind]
                    #print("Action penalty: ", action_penalty)
                    action_loss += self.energy_coeff * action_penalty
                #make_dot(action_loss).view()
                #input()
                #make_dot(value_loss).view()
                #input()
                loss += action_loss + value_loss #TODO TODO: verify this is valid, seperate for different modules?
                if self.entropy_coeff > 0.0 and not self.entropy_bonus and self.agent.module.sigma_head:
                    #entropy_loss = self.entropy_coeff * self.agent.policy_entropy_history[ind]
                    entropy_loss = self.compute_entropy_loss(ind)
                    #print("Entropy Loss:  ", entropy_loss)
                    loss += entropy_loss
                    net_entropy_loss += entropy_loss
                net_action_loss += action_loss
                #net_value_loss += value_loss / (end - 1 - start)
                net_value_loss += value_loss
            #if True:
            #    loss /= (end-1 - start) #get mean by dividing by # of samples per minibatch
            optimizer.zero_grad() #HAHAHAHA
            if self.scheduler is not None:
                self.scheduler.step()
            loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.5)
            print("Loss: %s" % (loss))
            optimizer.step()
            optimizer.zero_grad() #HAHAHAHA
            net_loss += loss
            run += 1
        print("Entropy Loss: ", net_entropy_loss)
        self.agent.net_loss_history.append(net_loss.cpu().detach())
        self.agent.action_loss_history.append(net_action_loss.detach().cpu())
        self.agent.value_loss_history.append(net_value_loss.detach().cpu())


class PyTorchACTrainer(PyTorchPolicyGradientTrainer):
    def compute_action_loss(self, advantage, ind, *args, **kwargs):
        action_score = self.agent.action_score_history[ind]
        #value_score = self.agent.value_history[ind] 
        #advantage = self.compute_advantage() #TODO:estimate advantage w/ average disc_reward vs value_scores
        #if self.entropy_bonus:
        #    advantage += self.entropy_coeff * self.agent.policy_entropy_history[ind]
        if self.agent.discrete:
            log_prob = action_score.log().max().to(device)
            action_loss = (-log_prob * advantage).squeeze(0)
        else:
            action_loss = (-action_score * advantage).squeeze(0)
        return action_loss

    def compute_entropy_loss(self, ind, *args, **kwargs):
        entropy_loss = self.entropy_coeff * self.agent.policy_entropy_history[ind]
        return entropy_loss


class PyTorchPPOTrainer(PyTorchPolicyGradientTrainer):
    def compute_action_loss(self, advantage, ind, *args, **kwargs):
        #value_score = self.agent.value_history[ind] 
        action_score = self.agent.action_score_history[ind].detach()
        state = self.agent.state_history[ind]
        action, value, normal = self.agent.evaluate(state) 
        #advantage = R - value_score.detach() #TODO:estimate advantage w/ average disc_reward vs value_scores
        #advantage = R - value.detach() #TODO:estimate advantage w/ average disc_reward vs value_scores
        #if self.entropy_bonus:
        #    print("entropy bonus")
        #    entropy_loss = self.entropy_coeff * self.agent.policy_entropy_history[ind]
        if self.agent.discrete:
            raise Exception("Hasn't been implemented for PPO")
            log_prob = action_score.log().max().to(device)
            action_loss = (-log_prob * advantage).squeeze(0)
        else:
            new_score = normal.log_prob(action)
            rt = torch.exp(new_score - action_score) #action_score is recorded prob of action AT SAMPLE TIME
            eps = 0.2 #recommended in PPO paper
            #loss_clip = torch.sum(torch.min(rt * advantage, torch.clamp(rt, 1-eps, 1+eps) * advantage))
            loss_clip = -1 * torch.min(rt * advantage, torch.clamp(rt, min=1-eps, max=1+eps) * advantage)
            #print("Loss_clip: ", loss_clip)
            action_loss = loss_clip.squeeze(0)
            #action_loss += entropy_loss
        return action_loss
    
    
    def compute_entropy_loss(self, ind, *args, **kwargs):
        state = self.agent.state_history[ind]
        action, value, normal = self.agent.evaluate(state) 
        entropy = -normal.entropy().mean()
        return self.entropy_coeff * entropy

    def compute_advantages(self, rewards, values, gamma = 0.99, normalize = True, compute_values = True):
        advantages = []
        for i in range(len(rewards) - 1):
            if compute_values:
                _, v, __ = self.agent.evaluate(self.agent.state_history[i])
                ___, v_, ____ = self.agent.evaluate(self.agent.state_history[i+1])
            else:
                v = values[i]
                v_ = values[i+1]
            advantages.append(rewards[i] + gamma*v_.detach() - v.detach())
        advantages.append(rewards[i+1] + gamma*v_.detach())
        advantages = torch.tensor(advantages, device = self.device)
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        return advantages

    #def compute_advantages(self, rewards, values, gamma = 0.99, *args, 
    #        **kwargs):
    #    '''Generalized advantage estimation, as described in PPO paper'''
    #    advantages = []
    #    for i in reversed(range(len(rewards): 
    #        delta = rewards[i] - values[i]
    #        advantage = delta + gamma * 
    #        advantages.append()


class PyTorchDynamicsTrainer(PyTorchTrainer):
    '''Generic Dynamics Trainer abstract class. Kinda jank, considering
    its initialization from a class that should really be called
    "RLTrainer" but just keep those fields None for now and 
    figure it out later. 
    
    Available args ('*' denotes
    use in default step() and thus required): [<agent>, <*env>, <*replay>,
    <max_traj_len>, <gamma>, <num_episodes>] why am I doing this
    
    "model" is the PyTorchMLP, "relation" is the means by which
    the inputs (assumed to be some function of ()) get interpreted
    as the next state, to compare to the actual one and/or finite
    differences

    Intended use case: 
    **LinearSystenDynamicsModel
    model = PyTorchMLP subclass with outdim = 2, output denoted f()
    f() : xt, ut -> A, B for xt+1
    relation: xt+1 = xt + dt(np.dot(A*xt) + np.dot(B*ut))
    
    **ForwardDynamicsModel
    model = PyTorchMLP subclass with outdim = 1, output denoted f()
    relation: xt+1 = xt + dt * f(xt, ut)
    
    "Have a nice day ;) "'''
    #TODO TODO TODO TODO TODO finite-differences for additional info.
    #Merge/fuse results <3 <3

    #TODO (OOP): Make "Estimator" class encompass / abstract this functionality
    #to more generic task of model-fitting in this framework, 
    #use inheritance to incrementally update functionality(OOP) 
    #^DEFINITELY overengineering for the task-at-hand
    def __init__(self, model : PyTorchModel, 
            dataset,
            criterion,
            batch_size = 64, collect_forward_loss = False, 
            *args, **kwargs):
        super(PyTorchDynamicsTrainer, self).__init__(*args, **kwargs)
        self.criterion = criterion
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        
        self.collect_forward_loss = collect_forward_loss

        self.net_loss_history = [] #net loss history
        self.forward_loss_history = [] #loss for MOST RECENT iteration 
        #independent of dataset
    
    def compute_model_loss(self, s, a, r, s_, *args, **kwargs):
        #print("S: %s \n A: %s" % (s, a))
        estimate = self.model.forward_predict(s, a, self.model.dt, *args, **kwargs)
        if type(s_) == np.ndarray:
            s_ = torch.tensor(s_, requires_grad = False, device = self.device).float()
        return self.criterion(s_, estimate)

    def step(self):
        requires_grad = True
        #YAY REFERENCES
        device = self.device
        optimizer = self.opt
        dataset = self.dataset
        #print("Dataset: ", dataset)
        dataset.trainer_step(self) 
        model = self.model

        loss = torch.tensor([0.0], requires_grad = requires_grad).to(device) #reset loss for each trajectory
        net_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device) #reset loss for each trajectory
        net_forward_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        for r in range(self.replay):
            print("Replay %s with batch size %s"% (r, self.batch_size))
            if hasattr(model.module, 'rec_size') and model.module.rec_size > 0: 
                model.module.reset_states() 
            loss = torch.tensor([0.0], requires_grad = requires_grad).to(device) #reset loss for each trajectory
            sample = dataset.sample(self.batch_size) 
            for i in range(len(sample)):
                s = self.get_sample_s(sample[i])
                a = self.get_sample_a(sample[i])
                r = self.get_sample_r(sample[i])
                s_ = self.get_sample_s_(sample[i])
                loss += self.compute_model_loss(s, a, r, s_)
            optimizer.zero_grad() #HAHAHAHA
            loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad() 
            net_loss += loss #accumulate loss before resetting it
        self.agent.net_loss_history.append(net_loss.cpu().detach())
        #self.net_loss_history.append(net_loss)
        #if self.collect_forward_loss:
        #    net_forward_loss.append(sum([compute_model_loss(*sample)]))
        #input()
    
    def plot_loss_histories(self):
        if self.collect_forward_loss:
            raise Exception("Do it now, I'm busy ;)")
        else:
            plt.figure(5)
            plt.title("Dynamics Model Loss History")
            plt.xlim(0, len(self.net_loss_history))
            plt.ylim(0, max(self.net_loss_history)+1)
            plt.ylabel("Net \n Dynamic Model Loss")
            plt.xlabel("Timestep")
            plt.scatter(range(len(self.net_loss_history)), [r.numpy()[0] for r in self.net_loss_history], s=1.0)


class PyTorchSAAutoencoderTrainer(PyTorchTrainer):
    def __init__(self, autoencoder, dataset, batch_size = 64, train_forward = True, 
            train_action = True,
            *args, **kwargs):
        super(PyTorchSAAutoencoderTrainer, self).__init__(*args, **kwargs)
        self._dataset = None
        self.dataset = dataset
        self.autoencoder = autoencoder
        self.batch_size = batch_size

        self.train_forward = train_forward
        self.train_action = train_action
        
        self.net_loss_history = []
        self.encoder_loss_history = []
        self.action_loss_history = []
        self.forward_loss_history = []

    @property
    @abc.abstractmethod
    def dataset(self):
        return self._dataset

    @dataset.setter
    @abc.abstractmethod
    def dataset(self, d):
        self._dataset = d

    def step(self):
        super().step()
        requires_grad = True
        reward = None
        ## Porting this function into the ACTrainer class
        module = self.autoencoder
        autoencoder = self.autoencoder
        #YAY REFERENCES
        device = self.device
        optimizer = self.opt
        dataset = self.dataset
        
        #TODO: Make this run with mini-batches?!
        if hasattr(module, 'rec_size') and module.rec_size > 0: 
            #TODO: don't do full FF, to save computations? 
            #compartmentalize .forward()
            module.reset_states(module.batch_size, False) 
        #for score in action_score_history:
        #    make_dot(score).view()
        #    input()
        dataset.trainer_step(self) 
        loss = torch.tensor([0.0], requires_grad = requires_grad).to(device) #reset loss for each trajectory
        net_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device) # accumulate loss after each training step
        net_autoencoder_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        net_forward_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        net_action_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        autoencoder_criterion = torch.nn.MSELoss()
        forward_criterion = torch.nn.MSELoss()
        for r in range(self.replay):
            sample = dataset.sample(self.batch_size) 
            #raise Exception("The following line was missing; test this again you DINGUS <3")
            loss = torch.tensor([0.0], requires_grad = requires_grad).to(device) #reset loss for each trajectory #idiot
            for i in range(len(sample)):
                s = self.get_sample_s(sample[i])
                a = self.get_sample_a(sample[i])
                r = self.get_sample_r(sample[i])
                s_ = self.get_sample_s_(sample[i])
                #print('s: %s \n a: %s \n r: %s \n s_: %s' % (s, a, r, s_))
                decoded_state, decoded_action = autoencoder.forward(s, a)
                autoencoder_state_loss = autoencoder_criterion(s, decoded_state)
                autoencoder_action_loss = autoencoder_criterion(a, decoded_action)
                loss += autoencoder_state_loss
                if self.train_action:
                    net_action_loss += autoencoder_action_loss
                    loss += autoencoder_action_loss
                if self.train_forward:
                    forward = autoencoder.forward_predict(s, a)
                    forward_loss = forward_criterion(s_, forward)
                    loss += forward_loss
                    net_forward_loss += forward_loss
                net_autoencoder_loss += autoencoder_state_loss #now it's just state autoencoding loss??
                #if self.train_action:
                #    net_autoencoder_loss += autoencoder_action_loss
            #torch.nn.utils.clip_grad_norm_(module.parameters(), 100)
            optimizer.zero_grad() #HAHAHAHA
            loss.backward(retain_graph = True)
            optimizer.step()
            optimizer.zero_grad() 
            
            net_loss += loss #accumulate loss before resetting it
        if self.scheduler is not None:
            self.scheduler.step()
        self.net_loss_history.append(net_loss.cpu().detach()) 
        self.encoder_loss_history.append((net_autoencoder_loss / (self.replay * len(sample))).cpu().detach())
        if self.train_forward:
            self.forward_loss_history.append((net_forward_loss / (self.replay * len(sample))).cpu().detach())
        if self.train_action:
            self.action_loss_history.append((net_action_loss / (self.replay * len(sample))).cpu().detach())

    def plot_loss_histories(self):
        plt.figure(5)
        num_plots = 1
        curr_plot = 1
        if self.train_forward:
            num_plots += 1
        if self.train_action:
            num_plots += 1
        plt.subplot(num_plots, 1, curr_plot)
        plt.title("Autoencoder Loss and Autoencoded-Forward-Loss History")
        plt.xlim(0, len(self.encoder_loss_history))
        plt.ylim(0, max(self.encoder_loss_history)+1)
        plt.ylabel("Net \n State AE Loss")
        plt.xlabel("Timestep")
        plt.scatter(range(len(self.encoder_loss_history)), [r.numpy()[0] for r in self.encoder_loss_history], s=1.0)
        curr_plot += 1
        if self.train_action:
            plt.subplot(num_plots, 1, curr_plot)
            plt.xlim(0, len(self.encoder_loss_history))
            plt.ylim(0, max(self.encoder_loss_history)+1)
            plt.ylabel("Net \n Action AE Loss")
            plt.xlabel("Timestep")
            plt.scatter(range(len(self.encoder_loss_history)), [r.numpy()[0] for r in self.action_loss_history], s=1.0)
            curr_plot += 1
        if self.train_forward:
            plt.subplot(3, 1, 3)
            plt.ylabel("Net \n Forward Loss")
            plt.xlabel("Timestep")
            plt.xlim(0, len(self.forward_loss_history))
            plt.ylim(0, max(self.forward_loss_history)+1)
            plt.scatter(range(len(self.forward_loss_history)), [r.numpy()[0] for r in self.forward_loss_history], s=1.0)
        plt.pause(0.01)



class LocalDynamicsTrainer(Trainer):
    '''Relying on SKLearn dynamics models to construct a model of a system. 

    Since SKLearn abstracts away a lot of the computation, the main loop (step) 
    of this algorithm should just call model.fit on some subsection of the dataset.

    That...should be sufficient.
    "Have a nice day ;) "'''
    def __init__(self, model,
            dataset,
            *args, **kwargs):
        #TODO: While we're using SKLearn tools, implement K-fold CV?
        super().__init__(*args, **kwargs)
        self.model = model
        self.dataset = dataset

        self.net_loss_history = [] #net loss history
        self.forward_loss_history = [] #loss for MOST RECENT iteration 
        #independent of dataset
    
    def step(self):
        self.dataset.trainer_step(self) 
        #sample = dataset.sample(self.batch_size)  #ONLY works with non-aggregative methods?
        samples = self.dataset.samples #we're using ALL the samples
        #print("SAMPLES: ", samples)
        states = [self.get_sample_s(s) for s in samples]
        actions = [self.get_sample_a(s) for s in samples]
        states_ = [self.get_sample_s_(s) for s in samples]
        states = np.array(states)
        states_ = np.array(states_)
        actions = np.array(actions)
        self.model.fit(states, states_, actions)
        self.agent.net_loss_history.append(0.0) #TODO: implement SKLearn score(X, y) here
    
    def plot_loss_histories(self):
        pass
        #if self.collect_forward_loss:
        #    raise Exception("Do it now, I'm busy ;)")
        #else:
        #    plt.figure(5)
        #    plt.title("Dynamics Model Loss History")
        #    plt.xlim(0, len(self.net_loss_history))
        #    plt.ylim(0, max(self.net_loss_history)+1)
        #    plt.ylabel("Net \n Dynamic Model Loss")
        #    plt.xlabel("Timestep")
        #    plt.scatter(range(len(self.net_loss_history)), [r.numpy()[0] for r in self.net_loss_history], s=1.0)
    
    #TODO: create superclass to PyTorchTrainer, have this class inherit from that 
    def get_sample_s(self, s):
        return s[0]
    def get_sample_a(self, s):
        return s[1]
    def get_sample_r(self, s):
        return s[2]
    def get_sample_s_(self, s):
        return s[3]

#class PyTorchSystemDynamicsModelTrainer(PyTorchDynamicsTrainer):
#    '''Updates parameterized (PyTorch) module to approximate
#    system dynamics (via approximation f(s,a) = A^(t) and B^(t)
#    and xt+1 = xt + dt(). That last part could be inherited from
#    a more generic "Esitmator" class)'''
#    #NOTE the doctest I'm sorry
#    def compute_model_loss(s, a, r, s_, *args, **kwargs):
#        pass
#
#
#class PyTorchForwardDynamicsModelTrainer(PyTorchDynamicsModelTrainer):
#    '''Updates parameterized (PyTorch) module to approximate
#    system dynamics (via xt+1 = xt + dt*f(). This could be abstracted
#    from a more generic "Estimator" class, or a child of "Estimator" 
#    specifically for system dynamics.)'''
#    #NOTE the doctest
#    def compute_model_loss(s, a, r, s_, *args, **kwargs):
#        pass



class PyTorchNeuralDynamicsMPCTrainer(PyTorchTrainer):
    #based on pg 4 (4D) of 1708.02596

    def __init__(self, mpc_agent, random_agent, 
            batch_size = 512,  
            #horizon = 50, 
            rand_sample_rate = 1.0, rand_sample_decay = 0.05, min_rand_sample_rate = 0.5,
            max_steps = 10000, max_iterations = 1000, *args, **kwargs):
        super(PyTorchNeuralDynamicsMPCTrainer, self).__init__(*args, **kwargs)
        self.mpc_agent = mpc_agent
        self.random_agent = random_agent
        self.batch_size = batch_size

        self.horizon = mpc_agent.horizon #right?!

        self.rand_sample_rate = rand_sample_rate
        self.rand_sample_decay = rand_sample_decay
        self.min_rand_sample_rate = min_rand_sample_rate

        self.max_steps = max_steps
        self.max_iterations = max_iterations

    def dynamic_step_loss(self, agent, state_history, action_history) -> torch.tensor:
        '''Takes, as argument, agent generating trajectory, and (st, at) pairs.'''
        device = self.mpc_agent.device
        loss = torch.tensor([0.0], requires_grad = True).to(device)
        criterion = torch.nn.MSELoss()
        if hasattr(agent, 'shoots'): #agent is MPC
            #print("Agent was MPC")
            forward = [shoot[0] for shoot in agent.shoots]
            predictions = [sum(dynamics) for dynamics in zip(state_history, forward)]
        else: #RANDOM AGENT?
            #print([(st, at) for (st, at) in zip(state_history, action_history)])
            #TODO TODO: PyTorchRandomAgent -> tensors + device
            action_history = [torch.tensor(a, device = device).float().squeeze(0) for a in action_history] #convert random np array -> tensor
            state_history = [s.to(device) for s in state_history]
            predictions = [self.mpc_agent.predict_state(*pair) for pair in zip(state_history, action_history)]
            #print("Predictions: ", predictions)
        for i in range(len(state_history)):
            loss += criterion(state_history[i], predictions[i])    
        return loss
        

    def dynamic_horizon_loss(self, agent, states, actions) -> torch.tensor:
        '''Evaluates the error of the neural dynamics from some starting
        state to a state in the horizon. USED FOR VALIDATION, NOT TRAINING'''
        criterion = torch.nn.MSELoss()
    
    def step(self):
        #raise Exception("Train value function DURING THIS STEP TOO for proper behavioral cloning\
        #?? Investigate! Also implement behavioral cloning phase (do we NEED to implement DAGGER?)?!")
        optimizer = self.opt
        horizon_loss = float('inf')
        avg_step_loss = float('inf')

        loss_history = []
        
        agent = None 
        env = self.env
        step = 0

        DISPLAY_HISTORY = True
        #while (horizon_loss > horizon_validation_thresh and 
        #        avg_step_loss > avg_step_loss_validation_thresh):
        while True: #1708.02596 pg 4: "until a predefined maximum iteration is reached
            print('TRAINING Step: ', step)
            if random.random() < self.rand_sample_rate: #RANDOM TRAJ
                self.rand_sample_rate = max(self.rand_sample_rate - self.rand_sample_decay,
                        self.min_rand_sample_rate)
                agent = self.random_agent
            else: #MPC TRAJ
                agent = self.mpc_agent
            print("Agent: ", agent) 
            timestep = env.reset()
            i = 0
            while not timestep.last() and i < self.max_iterations:
                #print("Iteration: ", i)
                reward = timestep.reward
                if reward is None:
                    reward = 0.0
                #print("TIMESTEP %s: %s" % (step, timestep))
                observation = timestep.observation
                action = agent(timestep)
                agent.store_reward(reward)
                #print("Action: ", type(action))
                timestep = env.step(action)
                #print("Reward: %s" % (timestep.reward))
                i += 1
            agent.terminate_episode() #oops forgot this lmao
            
            ## Training Step
            print("TRAINING STEP")
            loss = self.dynamic_step_loss(agent, agent.state_history, agent.action_history)
            loss.backward(retain_graph = True)
            optimizer.step()
            loss = loss.cpu().detach()
            loss_history.append(loss)
            #self.agent.net_loss_history.append(loss)
            
            ## Reset agent histories
            agent.reset_histories()
            if DISPLAY_HISTORY is True and len(self.mpc_agent.net_reward_history):
                plt.figure(1)
                plt.subplot(2, 1, 1)
                plt.title("MPC Reward and Loss History")
                plt.xlim(0, len(self.mpc_agent.net_reward_history))
                plt.ylim(0, max(self.mpc_agent.net_reward_history)+1)
                plt.ylabel("Net \n Reward")
                plt.scatter(range(len(self.mpc_agent.net_reward_history)), [r for r in self.mpc_agent.net_reward_history], s=1.0)
                plt.subplot(2, 1, 2)
                plt.ylabel("Net \n Loss")
                plt.scatter(range(len(loss_history)), [r.cpu().numpy()[0] for r in loss_history], s=1.0)
                plt.pause(0.01)

            step += 1
            if step > self.max_steps:
                break
        print("Final Avg Step Loss: ")
        print("Final Avg Horizon-Loss: ")


class TFTrainer(Trainer):
    pass
