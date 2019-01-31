#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 15:15:18 2019

@author: connorgalbraith
"""

import numpy as np
from random import choices, random
from copy import deepcopy

# =============================================================================
# This file contains class definitions for the various learning algorithms that
# each agent in agents.py can use, initialising the with their specific learning
# parameters to make implementing heterogeneity easier between agents easier.
#
# All reward function crafting is performed within the agents themselves and 
# passed to the relevant algorithm's methods, whose only purpose is to update 
# and return each agent's propensity arrays. 
# =============================================================================

class VERA():
    """ The Variant Erev-Roth Algorithm (VERA) is a stateless, stochastic learning
    
    method first suggested in Erev, I. and Roth, A. E. (1995), and improved on in 
    its modified form in Nicolaisen et al. (2001).
    """
    
    def __init__(self, recency, expmt, num_actions):
        
        self.recency = recency
        self.expmt = expmt
        self.num_actions = num_actions
        
    
    def initialisePropensities(self, action_space, state_space, single_bm_ladder = False):
        
        if single_bm_ladder == True:
        
            return (np.ones(len(action_space)))
        
        return np.ones((len(state_space), len(action_space)))
    
    
    def update(self, props, reward, chosen_action):
        """ Takes an array of action propensities, a scalar reward, and the 
        
        index of the propensity array corresponding to the action chosen during
        the round in question.
        """
        
        new_props = deepcopy(props)
                        
        for i, prop in enumerate(new_props):
                
            if i == chosen_action:
                
                 new_props[i] = (1 - self.recency) * prop + ( 1 - self.expmt) * reward
                
            else:
                
                new_props[i] = (1 - self.recency) * prop + (self.expmt * prop / (self.num_actions - 1))
                 
        return new_props
    


class QLearning():
    """ Basic, non-state-aware Q-Learning, with a learning rate alhpa that decays
    
    for each action proportional to the number of times that action has been
    visited.
    """
    
    def __init__(self, num_actions):
        
        self.alpha = {x: 1 for x in range(num_actions)}
        
        self.action_visit_counts = {x: 0 for x in range(num_actions)}
        
        
    def initialisePropensities(self, action_space, state_space, single_bm_ladder = False):
        
        if single_bm_ladder == True:
        
            return (np.ones(len(action_space)))
        
        return np.ones((len(state_space), len(action_space)))
    
    
    def update(self, props, reward, chosen_action):
            
        new_props = deepcopy(props)

        new_props[chosen_action] = props[chosen_action] + self.alpha[chosen_action] * (reward - props[chosen_action])
                
        self.action_visit_counts[chosen_action] += 1
        
        self.alpha[chosen_action] = 1/self.action_visit_counts[chosen_action]
        
        return new_props
    
    
    
class Stateful_QLearning():
    """ Basic, non-state-aware Q-Learning, with a learning rate alhpa that decays
    
    for each action proportional to the number of times that action has been
    visited.
    """
    
    def __init__(self, num_actions):
        
        self.alpha = {x: 1 for x in range(num_actions)}
        
        self.action_visit_counts = {x: 0 for x in range(num_actions)}
        
        
    def initialisePropensities(self, action_space, state_space, single_bm_ladder = False):
        
        if single_bm_ladder == True:
        
            return (np.ones(len(action_space)))
        
        return np.ones((len(state_space), len(action_space)))
    
    
    def update(self, props, reward, chosen_action):
            
        new_props = deepcopy(props)

        new_props[chosen_action] = props[chosen_action] + self.alpha[chosen_action] * (reward - props[chosen_action])
                
        self.action_visit_counts[chosen_action] += 1
        
        self.alpha[chosen_action] = 1/self.action_visit_counts[chosen_action]
        
        return new_props
        
        
    
    
    
    
    
    
    
    
    
    
                     