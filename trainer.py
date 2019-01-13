# Author: Aaron Parisi
# 11/20/18
from agent import Agent
import abc

import torch
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import random

class Trainer:
    __metaclass__ = abc.ABCMeta
    '''Baseclass for RL Trainers; these assume the agents are well-suited
    for their respective tasks and handle the implementation of the 
    partiuclar learning algorithm.
    
    Args: *replay: not None if random trajectory sampling is enabled
    (to reduce variance / biases from starting conditions)'''
    def __init__(self, agent : Agent, env, optimizer, 
            replay = 10, max_traj_len = 30, 
            gamma = 0.98, num_episodes = 2, *args, **kwargs):
        self.agent = agent
        self.env = env
        self.opt = optimizer
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

    @abc.abstractmethod
    def get_policy_entropy(self, start, end = None) -> np.ndarray:
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

    def report(self):
        ''' '''
        pass 

#TODO TODO: create Agent + Learner for state-dynamic NN coupling


    





class PyTorchTrainer(Trainer):
    def __init__(self, device, value_coeff = 0.1, entropy_coeff = 0.0, *args, **kwargs):
        self.device = device
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        super(PyTorchTrainer, self).__init__(*args, **kwargs)

    def get_discounted_reward(self, start, end = None) -> torch.tensor:
        # TODO: Do this with solely PyTorch operations?? 
        # BUILD THAT COMP GRAPH (ALSO: TODO requires_grad?)
        if end is None: 
            end = self.get_trajectory_end(end)
        net_reward = torch.tensor([0.0], requires_grad = False)
        rewards = self.agent.reward_history[start:]
            #print("Net reward: ", net_reward)
        #if i < len(state_value_history): #or i+1?
        for i in range(start, end):
            net_reward += self.gamma ** (i - start) * self.agent.reward_history[i]
            #net_reward += gamma**(i) * state_value_history[i].cpu().squeeze(0) #get last state value approximation
            #print("Bootstrap Value: ", gamma**(i) * state_value_history[i].cpu().squeeze(0))
        return net_reward.squeeze(0)

    def get_advantage(self, start, end = None) -> torch.tensor: 
        assert(len(self.agent.value_history) >= start) #make sure value estimates exist
        if end is None: 
            end = self.get_trajectory_end(end)
        R = self.get_discounted_reward(start, end)
        V_st = self.agent.value_history[start] 
        V_st_k = self.gamma**(end - start) * self.agent.value_history[end]
        return R + V_st_k - V_st

    def get_policy_entropy(self, start, end = 30):
        raise Exception("Make this accomodate continuous space (use distribution.entropy)")
        if end is None: 
            end = self.get_trajectory_end(start, end)
        net_entropy = torch.tensor([0.0], requires_grad = False).to(self.device)
        for s in self.agent.action_score_history[start : end]:
            s = s.squeeze(0)
            log_prob = s.log()
            entropy = (log_prob * s).sum().to(self.device)
            if not torch.isnan(entropy):
                #print("Entropy: ", entropy)
                net_entropy -= entropy
            else:
                #print("ENTROPY ISNAN")
                pass
        #print("Net Entropy: ", net_entropy)
        return net_entropy    

class PyTorchPreTrainer(PyTorchTrainer):
    def __init__(self, trainer, *args, **kwargs):
        super(PyTorchPreTrainer).__init__(*args, **kwargs)
        self.trainer = trainer       

    @abc.abstractmethod
    def training_criteria(self) -> bool:
        return False


#class PyTorchForwardDynamicsTrainer:
#    def __init__(self, device, mpc_agent, random_agent, optimizer, 
#            batch_size = 512, 
#            *args, **kwargs):
#        self.device = device
#        self.agent = agent
#        self.optimizer = optimizer
#        self.batch_size = batch_size
#
#    def step(self):
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
        #?? Investigate!")
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
                print("Iteration: ", i)
                reward = timestep.reward
                if reward is None:
                    reward = 0.0
                #print("TIMESTEP %s: %s" % (step, timestep))
                observation = timestep.observation
                action = agent(timestep)
                agent.store_reward(reward)
                timestep = env.step(action)
                #print("Reward: %s" % (timestep.reward))
                i += 1
            agent.terminate_episode() #oops forgot this lmao
            
            ## Training Step
            print("TRAINING STEP")
            loss = self.dynamic_step_loss(agent, agent.state_history, agent.action_history)
            loss.backward(retain_graph = True)
            loss_history.append(loss.detach())
            self.agent.net_loss_history.append(loss.cpu().detach())
            optimizer.step()
            
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
        
class PyTorchMB_MFPreTrainer(PyTorchPreTrainer):
    def __init__(self, dynamics_mlp, mpc_agent, horizon = 50, *args, **kwargs):
        super(PyTorchMB_MFPreTrainer, self).__init__(*args, **kwargs)
        self.horizon = horizon
        self.dynamics_mlp = dynamics_mlp
        self.mpc_agent = mpc_agent
    
    def behavioral_cloning_loss(self) -> torch.tensor:
        #TODO: Currently this doesn't clone any VALUE FUNCTION?!
        #TODO: implement DAGGER as well?!
        criterion = torch.nn.MSELoss()
        mpc_actions = self.mpc_agent.action_history
        agent_actions = [(action, normal) for action, _, normal in [self.agent.evaluate(obs) for obs in 
            self.mpc_agent.state_history]] 
        agent_scores = [normal.log_prob(action) for normal, action in agent_actions]
        return 0.5 * torch.sum(criterion(mpc_actions, agent_scores))
   
    def step(self): #TODO: Implement DAGGER as well??
        pass





class PyTorchACTrainer(PyTorchTrainer):
    def step(self):
        #TODO: Modify this for seperate policy/value networks
        #raise Exception("This doesn't seem to work for continuous-action spaces?!")
        super(PyTorchACTrainer, self).step()
        requires_grad = True
        reward = None
        action_scores = None 
        value_scores = None 
        ## Porting this function into the ACTrainer class
        module = self.agent.module
        device = self.device
        optimizer = self.opt
        action_scores = self.agent.action_score_history
        value_scores = self.agent.value_history
        #print("A_H len: %s V_H len: %s Reward Len: %s" % (len(action_scores), len(value_scores), len(self.agent.reward_history)))
        assert(len(value_scores) == len(action_scores)) #necessary
        ## FOR STORING LOSS HISTORIES


        ##
        #TODO: Make this run with mini-batches?!
        loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        #cum_reward = torch.sum(torch.tensor(reward_history, requires_grad = requires_grad), 0)
        #criterion = torch.nn.L1Loss()
        criterion = torch.nn.MSELoss()
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
        if self.max_traj_len > len(action_scores)/self.num_episodes: #if avg episode len < max traj len, run through each episode
            ends = self.get_episode_ends()
            replay_starts = [0]
            for e in ends[:len(ends) - 1]:
                replay_starts.append(e + 1)
        else:
            replay_starts = [random.choice(range(len(action_scores))) for i in range(self.replay)] #sample replay buffer "replay" times 
        for start in replay_starts:
            end = self.get_trajectory_end(start, end = None)
            loss = torch.tensor([0.0], requires_grad = requires_grad).to(device) #reset loss for each trajectory
            print("START: %s END: %s" % (start, end))
            for ind in range(start, end): #-1 because terminal state doesn't matter?
                #i = ind
                #optimizer.zero_grad()
                #update hidden state for LSTM
                #if module.lstm_size > 0:
                #    obs = get_observation_tensor(state_history[i])
                #    _ = module.forward(obs)
                #R = norm_returns[i]
                #R = get_discounted_reward(reward_history, i+1, gamma=gamma).unsqueeze(0).to(device)
                #R = self.get_discounted_reward(ind, end = end).unsqueeze(0).to(device)
                R = self.get_discounted_reward(ind, end = end).unsqueeze(0).unsqueeze(0).to(device)
                #print("i: %s len(action_score): %s value_scores: %s reward_history: %s" \
                #        % (i, len(action_score_history), len(state_score_history), len(reward_history)))
                action_score = action_scores[ind]
                value_score = value_scores[ind] #NOTE: i vs i+1?!
                #print("Value Score: ", value_scores)
                #print("Return: ", R)
                #print("Action Scores: %s \n Value Scores: %s\n" % (action_scores, value_scores))
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
                if self.agent.discrete:
                    log_prob = action_score.log().max().to(device)
                    action_loss = (-log_prob * advantage).squeeze(0)
                else:
                    action_loss = (-action_score * advantage).squeeze(0)
                #action_loss -= self.entropy_coeff * self.get_policy_entropy(ind, end=None) #TODO: fix entropy

                #make_dot(action_loss).view()
                #input()
                #print("Value score: %s Discounted: %s" % (value_score, R))
                value_loss = self.value_coeff * criterion(value_score, R) 
                if self.entropy_coeff > 0.0:
                    entropy_loss = self.entropy_coeff * self.get_policy_entropy(ind, end=None)
                    loss += action_loss + value_loss - entropy_loss #WEIGHTED
                else:
                    loss += action_loss + value_loss #WEIGHTED
                
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
                #raise Exception("Entropy + Value Loss Coefficients!")


            torch.nn.utils.clip_grad_norm_(module.parameters(), 5)
            loss.backward(retain_graph = True)
            self.agent.net_loss_history.append(loss.cpu().detach())
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
        #loss -= ent_coeff * self.get_policy_entropy(ind, end=None)
        #print("Loss: ", loss)        
        #loss.backward(retain_graph = i < len(reward_history) - 2)
        #torch.nn.utils.clip_grad_norm_(module.parameters(), 2)
        #loss.backward(retain_graph = True)
        #optimizer.step()
        #if module.lstm_size > 0: #TODO: don't do full FF, to save computations? compartmentalize .forward()
        #    module.reset_states() 
        #return net_action_loss, net_value_loss


class PyTorchPPOTrainer(PyTorchTrainer):
    def step(self):
        #TODO: Modify this for seperate policy/value networks
        #raise Exception("This doesn't seem to work for continuous-action spaces?!")
        super(PyTorchPPOTrainer, self).step()
        requires_grad = True
        reward = None
        action_scores = None 
        value_scores = None 
        ## Porting this function into the ACTrainer class
        module = self.agent.module
        device = self.device
        optimizer = self.opt
        action_scores = self.agent.action_score_history
        value_scores = self.agent.value_history
        #print("A_H len: %s V_H len: %s Reward Len: %s" % (len(action_scores), len(value_scores), len(self.agent.reward_history)))
        assert(len(value_scores) == len(action_scores)) #necessary
        ## FOR STORING LOSS HISTORIES


        ##
        #TODO: Make this run with mini-batches?!
        loss = torch.tensor([0.0], requires_grad = requires_grad).to(device)
        #cum_reward = torch.sum(torch.tensor(reward_history, requires_grad = requires_grad), 0)
        #criterion = torch.nn.L1Loss()
        criterion = torch.nn.MSELoss()
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
        if self.max_traj_len > len(action_scores)/self.num_episodes: #if avg episode len < max traj len, run through each episode
            ends = self.get_episode_ends()
            replay_starts = [0]
            for e in ends[:len(ends) - 1]:
                replay_starts.append(e + 1)
        else:
            replay_starts = [random.choice(range(len(action_scores))) for i in range(self.replay)] #sample replay buffer "replay" times 
        print("ENDS: %s \n LEN: %s" % (self.get_episode_ends(), len(action_scores)))
        #THIS IMPLIES that the way we are ending the episodes is...improper...
        for start in replay_starts:
            end = self.get_trajectory_end(start, end = None)
            loss = torch.tensor([0.0], requires_grad = requires_grad).to(device) #reset loss for each trajectory
            #print("START: %s END: %s" % (start, end))
            self.agent.steps = 0
            for ind in range(start, end): #-1 because terminal state doesn't matter?
                self.agent.steps += 1
                #i = ind
                #optimizer.zero_grad()
                #update hidden state for LSTM
                #if module.lstm_size > 0:
                #    obs = get_observation_tensor(state_history[i])
                #    _ = module.forward(obs)
                #R = norm_returns[i]
                #R = get_discounted_reward(reward_history, i+1, gamma=gamma).unsqueeze(0).to(device)
                R = self.get_discounted_reward(ind, end = end).unsqueeze(0).unsqueeze(0).to(device)
                #print("i: %s len(action_score): %s value_scores: %s reward_history: %s" \
                #        % (i, len(action_score_history), len(state_score_history), len(reward_history)))
                action_score = action_scores[ind]
                value_score = value_scores[ind] #NOTE: i vs i+1?!
                #print("Value Score: ", value_scores)
                #print("Return: ", R)
                #print("Action Scores: %s \n Value Scores: %s\n" % (action_scores, value_scores))
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
                if self.agent.discrete:
                    raise Exception("Hasn't been implemented for PPO")
                    log_prob = action_score.log().max().to(device)
                    action_loss = (-log_prob * advantage).squeeze(0)
                else:
                    state = self.agent.state_history[ind]
                    action, value, normal = self.agent.evaluate(state) 
                    new_score = normal.log_prob(action)
                    rt = torch.exp(new_score - action_score) #action_score is recorded prob of action AT SAMPLE TIME
                    eps = 0.2
                    #loss_clip = torch.sum(torch.min(rt * advantage, torch.clamp(rt, 1-eps, 1+eps) * advantage))
                    loss_clip = -1 * torch.min(rt * advantage, torch.clamp(rt, min=1-eps, max=1+eps) * advantage)
                    #print("PPO BIATCH. Rt: ", rt)
                    #action_loss = (-action_score * advantage).squeeze(0)
                    action_loss = loss_clip.squeeze(0)
                    #print("Clamped Loss: ", action_loss)
                #action_loss -= self.entropy_coeff * self.get_policy_entropy(ind, end=None) #TODO: fix entropy

                #make_dot(action_loss).view()
                #input()
                #print("Value score: %s Discounted: %s" % (value_score, R))
                value_loss = self.value_coeff * criterion(value_score, R) 
                if self.entropy_coeff > 0.0:
                    entropy_loss = self.entropy_coeff * self.get_policy_entropy(ind, end=None)
                    loss += action_loss + value_loss - entropy_loss #WEIGHTED
                else:
                    loss += action_loss + value_loss #WEIGHTED
                
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
                #raise Exception("Entropy + Value Loss Coefficients!")

            print("LOSS: ", loss)
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5)
            loss.backward(retain_graph = True)
            self.agent.net_loss_history.append(loss.cpu().detach())
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
        #loss -= ent_coeff * self.get_policy_entropy(ind, end=None)
        #print("Loss: ", loss)        
        #loss.backward(retain_graph = i < len(reward_history) - 2)
        #torch.nn.utils.clip_grad_norm_(module.parameters(), 2)
        #loss.backward(retain_graph = True)
        #optimizer.step()
        #if module.lstm_size > 0: #TODO: don't do full FF, to save computations? compartmentalize .forward()
        #    module.reset_states() 
        #return net_action_loss, net_value_loss




class TFTrainer(Trainer):
    pass


