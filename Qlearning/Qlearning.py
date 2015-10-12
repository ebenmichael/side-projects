# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:28:36 2015

@author: Eli
"""
import random
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from collections import deque
import itertools
from matplotlib import animation
import copy
import pandas as pd
from scipy import stats, signal
import time
import pickle

import sys
sys.path.append("../memm tagging")
import getData

class qLearn:
    
    def __init__(self,states,actions,gamma,alpha,epsilon):
        self.gamma = gamma
        self.Q = self.initQ(states,actions)
        self.states = states
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha

    
    def qUpdate(self,s,a,r,sPrime):
        """Updates a dict Q which contains the utilities for (state,action) pairs"""
        #get max_a' Q(s',a')
        """
        maxA = 0
        maxQ = float("-inf")
        for aCurr in actions:
            qCurr = Q[(sPrime,aCurr)]
            if qCurr > maxQ:
                maxA = aCurr
                maxQ = qCurr
        """
        maxQ = self.maxQ(sPrime)[0]
        #update Q and return it
        self.Q[(s,a)] = (1 - self.alpha) * self.Q[(s,a)] + self.alpha * (r + self.gamma * maxQ)
        
    def initQ(self,states,actions):
        """Randomly initializes a Q dict with all state action pairs"""
        
        Q = {}
        
        for a in actions:
            for s in states:
                Q[(s,a)] = random.randrange(100)
                
        return(Q)
    
    def chooseAction(self,state):
        """Chooses the action which maximizes Q(s,a) with prob 1-epsilon,
            chooses randomly with prob epsilon"""
        #generate float btwn 0-1
        choice = random.random()
        
        #choose according to that number
        if choice > self.epsilon:
            return(self.maxQ(state)[1])
        else:
            #choose randomly
            return(self.actions[random.randrange(0,len(self.actions))])
            
            
    def maxQ(self,state):
        """Returns the action which maximizes Q given the state, and the max of Q"""
        maxA = 0
        maxQ = float("-inf")
        for aCurr in self.actions:
            qCurr = self.Q[(state,aCurr)]
            if qCurr > maxQ:
                maxA = aCurr
                maxQ = qCurr  
        return(maxQ,maxA)


class QLearnCont():
    """Reinforcment learner with continuous state space, linear 
    http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html"""
    
    
    def __init__(self,feat_funct,size,actions,gamma,alpha,epsilon,
                 kernel = 'linear',sig = 1):
        """Constructor
        Input:
            feat_funct: function to go from states to features
            size: number of elements in parameter vector
            actions: possible actions as list
            gamma: discounting factor
            alpha: training rate
            epsilon: epsilon greedy value
            kernel: the kernel to use
            sig: kernel param
        """
        
        self.feat_funct = feat_funct
        self.actions = actions
        #get param vector
        self.w = np.random.randn(size)

        #get discounting factor
        self.gamma = gamma
        #get trainign rate
        self.alpha = alpha
        
        #get kernel
        self.kernel = kernel
        self.sig = sig
        self.epsilon = epsilon
        
    def Q(self,state,action):
        """Q function"""
        
        #get feature vector
        f = self.feat_funct(state,action)
        #use apporpriate kernel
        if self.kernel == 'linear':
            out = np.dot(self.w,f)
        elif self.kernel == 'rbf':
            out = np.exp(-(1/self.sig * 2)*np.linalg.norm(f - self.w)**2)
        #return linear function
        return(out)
    
    def grad(self,state,action):
        """Gradient"""
        #gradient is just the feature vector
        #use apporpriate gradient
        if self.kernel == 'linear':
            out = self.feat_funct(state,action)
        elif self.kernel == 'rbf':
            f = self.feat_funct(state,action)
            out = np.exp(-(1/self.sig * 2)*np.linalg.norm(f - self.w)**2) \
                        * 1/self.sig * np.linalg.norm(f - self.w) * self.w
        return(out)
    
        
    def qUpdate(self,state,action,reward,next_state):
        """Performs one update of w using gradient descent"""
        #get delta
        
        #delta = reward + self.gamma * self.Q(next_state,next_action) \
        #                - self.Q(state,action)
        
        #get e update
        #self.e = self.gamma  *self.lam * self.e - self.grad(state,action)
        
        
        #do update to w
        
        #self.w = self.alpha * delta * self.e
        #get difference between current q and new q
        
        delta = reward + self.gamma * self.maxQ(next_state)[0] -  \
                    self.Q(state,action)         
        #update w
        self.w = self.w + self.alpha * delta * self.grad(state,action)            

    def chooseAction(self,state):
        """Choose action espilon greedy"""
        #generate float btwn 0-1
        choice = random.random()
        
        #choose according to that number
        if choice > self.epsilon:
            return(self.maxQ(state)[1])
        else:
            #choose randomly
            return(self.actions[random.randrange(0,len(self.actions))])
                    
    def maxQ(self,state):
        """Find action which maximizes q. Return q and a"""
        
        maxQ = float('-inf')
        maxA = 0
        
        for a in self.actions:
            q = self.Q(state,a)
            #print(q,a)
            if q > maxQ:
                maxQ = q
                maxA = a
        return(maxQ,maxA)
        
class Linear():
    """Linear regression"""

    def __init__(self,dim,kernel = "linear",sig = 1):
        """Constructor
        Input:
            dim: the parameter vector dimension
        """
        
        self.param = np.random.randn(dim)
        self.kernel = kernel
        self.sig = sig
        
    def activate(self,x):
        """Activation function"""
        
        if self.kernel == "linear":
            out = np.dot(self.param,x)
        elif self.kernel == "rbf":
            out = np.exp(-(1/self.sig * 2)*np.linalg.norm(x - self.param)**2)
        return(out)
        
    def grad(self,x):
        """Gradient function"""
        if self.kernel == "linear":
            out = x
        elif self.kernel == "rbf":
            w = self.param
            out = np.exp(-(1/self.sig * 2)*np.linalg.norm(x - w)**2) \
                        * 1/self.sig * np.linalg.norm(x - w) * w
        return(out)
        
        
    
        
class DQN(QLearnCont):
    """Q Learning using experience replay and fixed Q-targets"""
    
    def __init__(self,feat_funct,size,actions,gamma,alpha,epsilon,
                 mini_batch = 100, num_update = 100,
                 learn_type = 'nn',**kwargs):
        """Constructor:
        Input:
            feat_funct: function to go from states to features
            size: number of elements in parameter vector (dim of feature space)
            actions: possible actions as list
            gamma: discounting factor
            alpha: training rate
            epsilon: epsilon greedy value
            mini_batch: size of minibatch
            num_update: number of iterations before updating the old learner
            learner: the type of learner to use
            kwargs: key word arguments for learner
        """
        
        self.feat_funct  = feat_funct
        self.actions = actions
        self.actions_dict = {action:i for i,action in enumerate(actions)}
        self.gamma = gamma
        self.alpha = alpha
        self.learn_type = learn_type
        self.size = size
        self.epsilon = epsilon
        
        #initialize the learner
        self.init_learner(**kwargs)
        #set old learner to current learner
        self.old_learn = copy.deepcopy(self.learner)
        #self.old_learn.evaulate = self.old_learn.activate
        #set number of iterations
        self.iteration = 1
        #initialize experience
        self.experience = deque([],100000)
        self.batch_size = mini_batch
        self.num_update = num_update
        
    def init_learner(self,**kwargs):
        """Randomly initializes a learner depending on type"""
        
        if self.learn_type == 'nn':
            #initialize neural network
            shape = kwargs["shape"]
            #initialize input layer
            model = Sequential()     
            #add hidden layers
            for i in range(len(shape)):
                if i == 0:
                    nb_input = self.size
                else:
                    nb_input = shape[i -1]
                nb_output = shape[i]
                model.add(Dense(nb_input,nb_output,init="he_normal",
                            activation = "tanh"))
                model.add(Dropout(.5))
            model.add(Dense(shape[-1],1,init = "he_normal",
                            activation = "linear"))
            model.compile(loss = 'mean_squared_error',optimizer = 'rmsprop')
            self.learner = model
        
        elif self.learn_type == 'linear':
            #initialize parameter
            self.learner = Linear(self.size,**kwargs)

    def comb_feat_action(self,feat,action):
        """Combine a feature vector with the action taken"""
        f = np.zeros((len(feat),len(self.actions)))
        col = self.actions_dict[action]
        f[:,col] = feat 
        return(feat)                   
            
    def Q(self,feat,action):
        """Q function"""
        f = self.comb_feat_action(feat,action)
        f = f.reshape(1,f.size)
        return(self.learner.predict(f))
        
    def oldQ(self,feat,action):
        """Old Q function"""
        f = self.comb_feat_action(feat,action)
        f = f.reshape(1,f.size)
        return(self.old_learn.predict(f))
        
    def qUpdate(self,state,action,reward,next_state):
        """Performs an update of Q"""
        #add to experience
        if next_state != "end":
            self.experience.append([self.feat_funct(state),action,
                                reward,self.feat_funct(next_state)])
        else:
            self.experience.append([self.feat_funct(state),action,
                                reward,next_state])            
        #print(state,action,reward,next_state)
        #get minibatch
        sample = np.random.randint(0,len(self.experience),self.batch_size)
        d = np.zeros((self.batch_size,self.size))
        y = np.zeros((self.batch_size,1))
        for i,row in enumerate(sample):
            state,action,reward,next_state = self.experience[row]
            #get feature vector
            d[i,:] = self.comb_feat_action(state,action)
            #get target
            #check if end of episoe
            if next_state == 'end':
                y[i] = reward
            else:
                y[i] = reward + self.gamma * self.maxOldQ(next_state)
            #print(row,next_state)
            #print(self.maxOldQ(next_state))
            
        loss = self.train(d,y)
        
        #update old learner if greater than num_update
        if self.iteration % self.num_update == 0:
            self.old_learn = copy.deepcopy(self.learner)
        self.iteration += 1
        
        return(loss)
    
    
    def train(self,features,y):
        """Trains the learner with the data"""
        
        if self.learn_type == "nn":
            #generate supervised dataset
            return(self.learner.train_on_batch(features,y))
        elif self.learn_type == "linear":
            grad = 0
            n = len(features)
            for i in range(n):
                #sum over the instances to get an estimate of the gradient
                print((y[i] - self.learner.activate(features[i])))
                grad -= (y[i] - self.learner.activate(features[i])) * \
                            self.learner.grad(features[i])
            grad /= n
            #update paramter
            param = np.copy(self.learner.param)
            self.learner.param = param -  self.alpha * grad
            #print(self.learner.param)
                
             
    def maxOldQ(self,feat):
        """Maximizes the old Q value"""
        maxQ = float('-inf')
        for a in self.actions:
            q = self.oldQ(feat,a)
            
            if q > maxQ:
                maxQ = q
        
        return(maxQ)        
        
    def maxQ(self,feat):
        """Find action which maximizes q. Return q and a"""
        
        maxQ = float('-inf')
        maxA = 0
        for a in self.actions:
            q = self.Q(feat,a)
            print(q,a)
            if q > maxQ:
                maxQ = q
                maxA = a
        return(maxQ,maxA)

    def chooseAction(self,state):
        """Choose action espilon greedy"""
        #generate float btwn 0-1
        choice = random.random()
        feat = self.feat_funct(state)
        #choose according to that number
        if choice > self.epsilon:
            return(self.maxQ(feat)[1])
        else:
            #choose randomly
            return(self.actions[random.randrange(0,len(self.actions))])            



def ipd(length,gamma1,epsilon1,gamma2,epsilon2):
    """Runs an iterated prisoner's dillema with two Q-learning agents"""
    #possible previous states (what each did in the last iteration)
    states = [("*","*"),("C","D"), ("C","C"), ("D","C"), ("D","D")]
    #actions: Defect or Cooperate
    actions = ["D","C"]
    #payoff matrix (as dict)
    payoff = {("C","D"): (-3,0), ("C","C"): (-1,-1), 
              ("D","C"): (0,-3), ("D","D"): (-2,-2)}
    #initialize learners   
    q1 = qLearn(states,actions,gamma1,epsilon1)
    q2 = qLearn(states,actions,gamma2,epsilon2)
    #initialize list of rewards
    rewards = []    
    #iterate through length states and run the game
    prevState = ("*","*")
    for i in range(length):
        #get actions
        #print("Iteration %i:" %i)
        #print("Previous State:", prevState)
        qa1 = q1.chooseAction(prevState)
        qa2 = q2.chooseAction(prevState)
        #print("Player 1 Action:",qa1)
        #print("Player 2 Action:",qa2)
        
        #find payoff
        newState = (qa1,qa2)
        reward = payoff[newState]
        rewards.append(sum(reward))
        #print("Player 1 Reward:", reward[0])
        #print("Player 2 Rewards:", reward[1])
        #assign reward and update Q params
        q1.qUpdate(prevState,qa1,reward[0],newState)
        q2.qUpdate(prevState,qa2,reward[1],newState)
        
        prevState = newState
    #print(q1.Q)
    #print(q2.Q)
    return(rewards)
    
def ipdTft(length,gamma,epsilon,alpha = .8):
    """Runs an iterated prisoners dilemma with a Q-learner and tit for tat"""
    #possible previous states (what each did in the last iteration)
    states = [("*","*"),("C","D"), ("C","C"), ("D","C"), ("D","D")]
    #actions: Defect or Cooperate
    actions = ["D","C"]
    #payoff matrix (as dict)
    payoff = {("C","D"): (-3,0), ("C","C"): (-1,-1), 
              ("D","C"): (0,-3), ("D","D"): (-2,-2)}
    #initialize learners   

    #q1 = qLearn(states,actions,gamma,alpha,epsilon)
    #q1 = QLearnCont(ipd_feats,10,actions,gamma,alpha,epsilon,kernel = 'linear')
    #q1 = DQN(ipd_feats,10,actions,.99,.5,.1,learn_type = 'linear')
    q1 = DQN(ipd_feats,10,actions,.99,.5,.1,shape = (10,10,1))
    #initialize list of rewards
    rewards = []
    #iterate through length states and run the game
    prevState = ("*","*")
    for i in range(length):
        #get actions
        print("Iteration %i:" %i)
        print("Previous State:", prevState)
        qa1 = q1.chooseAction(prevState)
        qa2 = tft(prevState[0])
        print("Player 1 Action:",qa1)
        print("Player 2 Action:",qa2)
        
        #find payoff
        newState = (qa1,qa2)
        reward = payoff[newState]
        rewards.append(reward[0])
        print("Player 1 Reward:", reward[0])
        print("Player 2 Rewards:", reward[1])
        print("Current average reward for Player 1:",np.mean(rewards))
        #assign reward and update Q params
        q1.qUpdate(prevState,qa1,reward[0],newState)
        
        prevState = newState
    #print(q1.Q)
    return(rewards,q1)
def ipd_feats(state,action):
    """feature vector representation of states and actions"""
    f = np.zeros((5,2))
    a = int(action == "C") 
    #dict for states to ints
    states = {("*","*"):0,("C","D"):1, ("C","C"):2, ("D","C"):3, ("D","D"):4}
    f[states[state],a] = 1
    f = f.flatten(1)
    return(f)
            
def tft(prevMove):
    """tit for tat"""
    if prevMove == "*":
        return("C")
    else:
        return(prevMove)
            
def repIpdTft(length,gammas,epsilon):
    """repeats the idp with a q-learner against a tit for tat player for 
        different values of gamma, and returns average reward in each group 
        of games"""
    avgRewards = []
    for gamma in gammas: 
        avgRewards.append(np.mean(ipdTft(length,gamma,epsilon)))
    return(avgRewards)
    
def repIpd(length,gammas,epsilon1,epsilon2):
    """repeats the idp with two qlearners for different values of gamma"""
    avgRewards = []
    for gamma in gammas: 
        avgRewards.append(np.mean(ipd(length,gamma,epsilon1,gamma,epsilon2)))
    return(avgRewards)
    
    
class AvoidBall():
    """Class that makes an avoiding ball(s) task"""
    
    
    def __init__(self,balls,gamma=.99,alpha=.8,epsilon=.1,mini_batch = 100,
                 num_update = 100, learn_type = 'nn', num_rots = 8
                 ,bound_box = [[-10,10],[-10,10]],**kwargs):
        """Constructor
        Input:
            boxes: list of list of ball locations in 2-d
            gamma: discounting factor
            alpha: training rate
            epsilon: epsilon greedy value
            mini_batch: size of minibatch
            num_update: number of iterations before updating the old learner
            learner: the type of learner to use
            num_rots: number of rotations 
            num_acc: number of accelerations
            bound_box: the bounds of the area the agent can go
            kwargs: key word arguments for learner
        """
        #get dimension
        self.dim = len(balls[0])
        
        
        self.balls = balls
        #actions
        #rotations
        rots = [2*i*np.pi / num_rots for i in range(num_rots)]
        #accelerations
        self.actions = rots
        size = len(self.actions) * self.dim
        #mapping from actions to integers
        self.action_dict = {a:i for i,a in enumerate(self.actions)}
        self.agent = DQN(self.feat_funct,size,self.actions,gamma,alpha,
                         epsilon,mini_batch,num_update,learn_type,**kwargs)
        #self.agent = QLearnCont(self.feat_funct,size,self.actions,gamma,
        #                        alpha,epsilon,**kwargs)
        #initialize velocity and position
        self.pos = np.random.randn(self.dim)
        
        #keep track of rewards
        self.rewards = []
        
        self.bound_box = bound_box
        
        
        
    def feat_funct(self,state,action):
        """returns a feature representation of the state and action"""
        #get position and velocity from state
        pos = state
        f = np.zeros((2,len(self.actions)))
        #assign position and velocity vectors to the column of the action
        col = self.action_dict[action]
        f[:,col] = pos
        #flatten along columns and return
        return(f.flatten(1))
        
    def update(self):
        """Updates one iteration"""
        #get current state
        state = self.pos
        #print(state)
        #get action
        action = self.agent.chooseAction(state)
        
        #get payoff (sum of opposite of rbf kernel btwn agent and balls)


            
        #reward = sum(self.neg_rbf(self.pos,ball) for ball in self.balls)
        #reward = min(np.linalg.norm(self.pos - np.array(ball)) for ball in self.balls)
        #print([np.linalg.norm(self.pos - np.array(ball)) for ball in self.balls])
        #print(reward)
        reward = 100 * np.exp(-np.linalg.norm(self.pos - np.array([1,1]))**2)
        #print(self.pos)
        #print(action)
        #print(reward)
        self.rewards.append(reward)
        #calculate new position and velocity
        #construct acceleration vector
        #print(action)
        #print(acc)                
        #update position
        #print(self.vel)
        pos = self.pos + np.array([np.cos(action),np.sin(action)])
        #don't let agent out of the box
        """xlim = self.bound_box[0]
        ylim = self.bound_box[1]        
        if pos[0] < xlim[0] or pos[0] > xlim[1] or \
            pos[1] < ylim[0] or pos[1] > ylim[1]: 
            pos = self.pos"""

        next_state = pos
        #print(self.pos,self.vel)
        #do qupdate
        #print(np.mean(self.rewards))
        #print(state,next_state)
        #print()
        self.agent.qUpdate(state,action,reward,next_state)
        self.pos = pos
        #print(pos)
        #update ball positions to follow the agent
        for i,ball in enumerate(self.balls):
            self.balls[i] = (ball - self.pos) * 3
    
    def neg_rbf(self,x,y):
        """Negative of rbf kernel"""
        return(-np.exp(-(1/5)*np.linalg.norm(x - y)**2))
        



