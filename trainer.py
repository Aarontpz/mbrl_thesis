# Author: Aaron Parisi
# 11/20/18
from agent import Agent
import abc

import torch
import tensorflow as tf
import numpy as np

class Trainer:
    __metaclass__ = abc.ABCMeta
    '''Baseclass for RL Trainers; these assume the agents are well-suited
    for their respective tasks and handle the implementation of the 
    partiuclar learning algorithm.
    
    Args: *replay: not None if random trajectory sampling is enabled
    (to reduce variance / biases from starting conditions)'''
    def __init__(self, agent : Agent, env, optimizer, 
            replay = 10, max_traj_len = 30, 
            gamma = 0.98, *args, **kwargs):
        self.agent = agent
        self.env = env
        self.opt = optimizer
        self.replay = replay

        self.max_trajectory_len = max_traj_len
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

    @abc.abstractmethod
    def get_policy_entropy(self, start, end = None) -> np.ndarray:
        pass

    def get_trajectory_end(self, start, end = None):
        if end is None:
            end = self.max_traj_len
            if True in self.terminal_history[start:end]:
                end = self.terminal_history.index(True)
                #TODO: consider minimal traj length too?
        return end

    def report(self):
        ''' '''
        pass 

#TODO TODO: create Agent + Learner for state-dynamic NN coupling


    
class TFTrainer(Trainer):
    pass


class PyTorchTrainer(Trainer):
    def __init__(self, device, replay_iterations = 20, entropy_coeff = 1.0, *args, **kwargs):
        self.device = device
        self.replay_iterations = replay_iterations
        super(PyTorchTrainer, self).__init__(*args, **kwargs)

    def get_discounted_reward(self, start, end = None) -> torch.tensor:
        # TODO: Do this with solely PyTorch operations?? 
        # BUILD THAT COMP GRAPH (ALSO: TODO requires_grad?)
        net_reward = torch.tensor([0.0], requires_grad = False)
        rewards = self.agent.reward_history[start:]
        end = self.get_trajectory_end(end)
            #print("Net reward: ", net_reward)
        #if i < len(state_value_history): #or i+1?
            #net_reward += gamma**(i) * state_value_history[i].cpu().squeeze(0) #get last state value approximation
            #print("Bootstrap Value: ", gamma**(i) * state_value_history[i].cpu().squeeze(0))
        return net_reward

    def get_policy_entropy(self, start, end = 30):
        net_entropy = torch.tensor([0.0], requires_grad = False).to(self.device)
        end = self.get_trajectory_end(end)
        for s in self.agent.action_history[start : end]:
            s = s.squeeze(0)
            log_prob = s.log()
            entropy = (log_prob * s).sum().to(self.device)
            #print("Entropy: ", entropy)
            if not torch.isnan(entropy):
                net_entropy -= entropy
            else:
                print("ENTROPY ISNAN")
        #print("Net Entropy: ", net_entropy)
        return net_entropy    

class PyTorchACTrainer(PyTorchTrainer):
    def step(self):
        #TODO: Modify this for seperate policy/value networks
        requires_grad = True
        reward = None
        action_scores = None 
        value_scores = None 
        ## Porting this function into the ACTrainer class
        module = self.agent.module
        device = self.device
        optimizer = self.opt
        action_scores = self.agent.action_history
        value_scores = self.agent.value_history
        assert(len(value_scores) == len(action_scores)) #necessary

        ##

        ## FOR STORING LOSS HISTORIES

        #TODO: Make this run with mini-batches?!
        loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        #cum_reward = torch.sum(torch.tensor(reward_history, requires_grad = requires_grad), 0)
        criterion = nn.L1Loss()
        #criterion = nn.MSELoss()
        #print("Cumulative reward: ", cum_reward)
        #disc_reward = get_discounted_reward(reward_history, 0, gamma=gamma)
        #net_action_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        #net_value_loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        if hasattr(module, 'rec_size') and module.rec_size > 0: 
            #TODO: don't do full FF, to save computations? 
            #compartmentalize .forward()
            module.reset_states(module.batch_size, False) 
        #returns = get_return_tensor(reward_history, gamma=gamma)
        #norm_returns = normalize_return_tensor(returns)
        i = 0
        #for score in action_score_history:
        #    make_dot(score).view()
        #    input()
        optimizer.zero_grad()
        replay_starts = [random.choice(range(len(action_scores))) for i in range(self.replay)] #sample replay buffer "replay" times 
        for ind in random_starts: #-1 because terminal state doesn't matter?
            loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
            #i = ind
            #optimizer.zero_grad()
            #update hidden state for LSTM
            #if module.lstm_size > 0:
            #    obs = get_observation_tensor(state_history[i])
            #    _ = module.forward(obs)
            #R = norm_returns[i]
            #R = get_discounted_reward(reward_history, i+1, gamma=gamma).unsqueeze(0).to(device)
            R = self.get_discounted_reward(self, ind, end = None).unsqueeze(0).to(device)
            #print("i: %s len(action_score): %s value_scores: %s reward_history: %s" \
            #        % (i, len(action_score_history), len(state_score_history), len(reward_history)))
            action_score = action_scores[ind]
            value_score = value_scores[ind] #NOTE: i vs i+1?!
            #print("Value Score: ", value_scores)
            #print("Return: ", R)
            #print("Action Scores: %s \n Value Scores: %s\n" % (action_scores, value_scores))
            log_prob = action_score.log().max().to(device)
            #print(torch.isnan(s) for s in log_prob)
            #print("Log Prob: ", log_prob)
            #print("FIXING SCORE")
            ##max_score, max_index = torch.max(action_scores, 1)
            ##m = torch.distributions.Categorical(action_scores)
            ##log_prob = m.log_prob(m.sample()).to(device)
            ##print("Log Prob: ", log_prob)
            #log_prob = m.log_prob(max_score)
            #print("M: %s (sample) vs %s (indexed)"%(-m.log_prob(m.sample()), -m.log_prob(max_index))) #NOTE: This is important to understand wtf is happ
            #reward = torch.clamp(torch.tensor(reward_history[i+1]), -1, 1)
            #print("Discounted Reward: ", disc_reward)
            #reward = reward_history[i+1]
            #if disc_reward > 0.5:
                #print("Discounted reward: ", disc_reward)
            advantage = R - value_score #TODO:estimate advantage w/ average disc_reward vs value_scores
            action_loss = (-log_prob * advantage).squeeze(0)
            #make_dot(action_loss).view()
            #input()
            value_loss = criterion(value_score, R) 
            entropy_loss = self.get_policy_entropy(ind, end=None)
            
            #make_dot(value_loss).view()
            #input()
            #print("Action Loss: %s Value Loss: %s" % (action_loss, value_loss))
            #net_action_loss += action_loss
            #net_value_loss += value_loss
            #print("Action Loss: %s \n Value Loss: %s" % (action_loss, value_loss))
            #if update_both:
            #    loss = value_loss + action_loss
            #else: 
            #    if update_values:
            #        print("CURRENT VALUE LOSS (to min): ", value_loss)
            #        loss = value_loss
            #    else:
            #        loss = action_loss #TODO: occasionally (NOT simultaneously) update value estimator
            #loss.backward(retain_graph = i < len(reward_history) - 2)
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5)
            loss.backward(retain_graph = True)
            optimizer.step()
                #print("Step %s Loss: %s" % (i, loss))
        #print("FINAL i: ", i)
        #if average_value_loss == True and i > 0:
        #    net_value_loss /= i #correct for the inevitability of this loss being MASSIVE
        #if update_both:
        #    loss = net_action_loss + net_value_loss 
            #print("CURRENT VALUE LOSS (to min): ", net_value_loss)
            #print("CURRENT ACTION LOSS (to min): ", net_action_loss)
        #else: 
        #    if update_values:
                #print("CURRENT VALUE LOSS (to min): ", net_value_loss)
        #        loss = net_value_loss
        #    else:
                #print("CURRENT ACTION LOSS (to min): ", net_action_loss)
                #loss = net_action_loss #TODO: occasionally (NOT simultaneously) update value estimator
        #if not update_both and not update_values: #avoid entropy loss if just updating values
        #    loss -= ent_coeff * self.get_policy_entropy(ind, end=None)
        #print("Loss: ", loss)        
        #loss.backward(retain_graph = i < len(reward_history) - 2)
        #torch.nn.utils.clip_grad_norm_(module.parameters(), 2)
        #loss.backward(retain_graph = True)
        #optimizer.step()
        #if module.lstm_size > 0: #TODO: don't do full FF, to save computations? compartmentalize .forward()
        #    module.reset_states() 
        #return net_action_loss, net_value_loss







