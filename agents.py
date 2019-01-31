# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:20:15 2018
@author: conno
"""

import numpy as np
from random import choices, random
from copy import deepcopy
import math as m

import mesa_modified as mesa
import marl


class Generator(mesa.Agent):
    """ Properties should be assigned in model.py as simulation parameters, using
    the batch runner to iteratre when need, as well as the form of the markup set. 
    gen_id increments with each realisation of Generator and is assigned to each 
    instance as a unique identifier.
    """

    def __init__(self, model, fuel_id, params):

        super().__init__(fuel_id[1], model)

        # internalise simulation length to calculate temperature/epsilon decay rate
        self.days = params['days']
        self.day = 0
        
        # stores phyiscal parameters of the agent
        self.fuel = fuel_id[0]
        self.id = fuel_id[1]
        self.capacity = params[fuel_id[0]]['cap']
        self.marginal_cost = params[fuel_id[0]]['marginal_cost']
        self.startup_cost = params[fuel_id[0]]['startup_cost/MW'] * self.capacity
        self.cycles = params[fuel_id[0]]['cycles']
        self.min_gen = params[fuel_id[0]]['min_gen_prop'] * self.capacity
        self.emissions = params[fuel_id[0]]['emissions']
        
        # initialises the true wind profile for the given simulation day
        if self.fuel == 'wind':
            self.wind_sd = params['wind_sd']
            self.true_wind = params['true_wind']
            self.dynamic_imbalance = params['dynamic_imbalance']
        
        # learning parameters
        self.learning_mechanism = params['learning_mechanism']
        
        self.px_action_method = params['px_action_method']
        self.bm_action_method = params['bm_action_method']
        
        self.px_reward_method = params['px_reward_method']
        self.bm_reward_method = params['bm_reward_method']
        self.discount = params['discount']
        self.dampening_factor = params['dampening_factor']
        self.kernel_radius = params['kernel_radius']
        
        self.px_recency = params['px_recency']
        self.px_expmt = params['px_expmt']
        
        self.bm_recency = params['bm_recency']
        self.bm_expmt = params['bm_expmt']
        
        self.px_temperature_inf = params['px_temperature_inf']
        self.px_temperature_start = params['px_temperature_start']
        self.px_temperature_decay = params['px_temperature_decay']
        
        self.bm_temperature_inf = params['bm_temperature_inf']
        self.bm_temperature_start = params['bm_temperature_start']
        self.bm_temperature_decay = params['bm_temperature_decay']
        
        self.px_epsilon_inf = params['px_epsilon_inf']
        self.px_epsilon_start = params['px_epsilon_start']
        self.px_epsilon_decay = params['px_epsilon_decay']
        
        self.bm_epsilon_inf = params['bm_epsilon_inf']
        self.bm_epsilon_start = params['bm_epsilon_start']
        self.bm_epsilon_decay = params['bm_epsilon_decay']
        
        # offer_set very important. Limit to marginal_cost or not?
            
        self.px_price_offer_set = list(np.round(np.linspace(25, 125, 21), 0))
            
        self.px_volume_offer_set = list(np.round(np.linspace(0.5, 1, 21), 3))
        
        self.balancing_mechanism = params['balancing_mechanism']
        
        # list of bm intercept and gradient values
        
        self.bm_intercept_set = list(np.round(np.linspace(5, 100, 21), 2))
        self.bm_gradient_set = list(np.round(np.linspace(0, 10, 21), 1))
        
        self.bm_gradient_hold = params['bm_gradient_hold']
        self.single_bm_ladder = params['single_bm_ladder']
        
        
        # apply any heterogenous parameters specific to the agent
        
        if self.id in params['heterogeneous_parameters']:
            
            for param_name, param_value in params['heterogeneous_parameters'][self.id].items():
                
                setattr(self, param_name, param_value)
                print(self.learning_mechanism)
                
        
        # initialises propensities for PX price, PX quantities, and BM ladder
        # can also accept user-specified propensities, or from a previous run
        
        if self.learning_mechanism == 'VERA':
            
            self.px_price_RL = marl.VERA(self.px_recency, self.px_expmt, len(self.px_price_offer_set))
            self.px_volume_RL = marl.VERA(self.bm_recency, self.bm_expmt, len(self.px_volume_offer_set))
            self.bm_intercept_RL = marl.VERA(self.bm_recency, self.bm_expmt, len(self.bm_intercept_set))
            self.bm_gradient_RL = marl.VERA(self.bm_recency, self.bm_expmt, len(self.bm_gradient_set))
            
            self.state_space = range(48)
            
        elif self.learning_mechanism == 'QLearning':
            
            self.px_price_RL = marl.QLearning(len(self.px_price_offer_set))
            self.px_volume_RL = marl.QLearning(len(self.px_volume_offer_set))
            self.bm_intercept_RL = marl.QLearning(len(self.bm_intercept_set))
            self.bm_gradient_RL = marl.QLearning(len(self.bm_gradient_set))
            
            self.state_space = range(48)
            
        elif self.learning_mechanism == 'Stateful_QLearning':
            
            self.px_price_RL = marl.Stateful_QLearning(len(self.px_price_offer_set))
            self.px_volume_RL = marl.Stateful_QLearning(len(self.px_volume_offer_set))
            self.bm_intercept_RL = marl.Stateful_QLearning(len(self.bm_intercept_set))
            self.bm_gradient_RL = marl.Stateful_QLearning(len(self.bm_gradient_set))
            
            # this will need tweaking, but wil essentially become the proportion
            # of max available capacity that the demand is closest to
            
            self.max_capacity = params['max_capacity']
            self.state_space = list(np.linspace(0, 1, 21))
        
        if len(params['propensities'][self.id]) == 0:
            
            self.px_price_propensities = self.px_price_RL.initialisePropensities(self.px_price_offer_set,
                                                                                 self.state_space)
            
            self.px_volume_propensities = self.px_volume_RL.initialisePropensities(self.px_volume_offer_set,
                                                                                   self.state_space)
        
            self.bm_intercept_propensities = self.bm_intercept_RL.initialisePropensities(self.bm_intercept_set,
                                                                                         self.state_space,
                                                                                         single_bm_ladder = self.single_bm_ladder)
            
            self.bm_gradient_propensities = self.bm_gradient_RL.initialisePropensities(self.bm_gradient_set,
                                                                                       self.state_space,
                                                                                       single_bm_ladder = self.single_bm_ladder)
            
        else:
        
            self.px_price_propensities = params['propensities'][self.id]['px_price']
            self.px_volume_propensities = params['propensities'][self.id]['px_volume']
            self.bm_intercept_propensities = params['propensities'][self.id]['bm_intercept']
            self.bm_gradient_propensities = params['propensities'][self.id]['bm_gradient']
            
        
        # if the balancing mechanism is not in play, then volume offers are set
        # at full capacity
        
        if self.balancing_mechanism == False:
            
            self.px_volume_propensities = np.zeros((len(self.state_space), len(self.px_volume_offer_set)))
            self.px_volume_propensities[-1] = 1
            
            self.bm_period_profit = []
            self.bm_day_profit = []
            self.bm_intercept_choices = []
            self.bm_gradient_choices = []
            self.bm_intercept_props = []
            self.bm_gradient_props = []
        
        # reward signal scaled to the max profit the generator can make in a 
        # period, to make the reward independent of physical parameters
        
#        self.dampener = self.capacity * (self.px_price_offer_set[-1] - self.marginal_cost) * self.dampening_factor
        self.dampener = 1
        
        # keeps track of the offers the agent makes, their success rate, and
        # the average profit made from each to inform future actions. Each offer
        # key links to a list of the form [offer frequency, offer successes,
        # success ratio, average profit, expected_profit, change since previous
        # day, average reward, expected reward]
        
        self.expected_profits = {offer: {period: [0, 0, 0, 0, 0, 0, 0, 0] for period in range(48)} for offer in self.px_price_offer_set}
        
        
        # if imbalance is static, initialises the error profile for wind. If a
        # synthetic imbalance is specified, sets forecast_wind = true_wind
        
        if (len(params['synthetic_imbalance']) != 0) and (self.fuel == 'wind'):

            self.forecast_wind = self.true_wind/2
            
        elif (params['dynamic_imbalance'] == False) and (self.fuel == 'wind'):
            
            self.forecast_wind = np.zeros(48)
            
            self.forecast_wind[0] = (self.true_wind[0]/2) * (1 + np.random.normal(scale = self.wind_sd))
            self.forecast_wind[1] = self.forecast_wind[0] + 0.25*(self.true_wind[1]/2 - self.forecast_wind[0])
            
            for i in range(2, 48):
                
                self.forecast_wind[i] = self.forecast_wind[i-1]*(1 + np.random.normal(scale = self.wind_sd)) + \
                                   + 0.25*(self.true_wind[i]/2 - self.forecast_wind[i-1])
                                   
            
            
        
        
    def softmax(self, x, temperature):
        """Compute softmax values for each sets of scores in x."""
        
        e_x = np.exp((x - np.max(x))/temperature)
        
        return e_x / e_x.sum(axis=0)
    
    
    
    def create_kernel(self, radius, discount):
        """Creates a softmaxed smoothing kernel for the kernel_profit method, 
        
        with width 1 + 2*radius and a discount factor = discount.
        """
    
        kernel = np.ones(1 + 2*radius)
        
        for i in range(radius):
            kernel[i]  *= discount ** (radius - i)
            kernel[2*radius - i] *= discount ** (radius - i)
        
        return self.softmax(kernel, 0.2)
    
 
    
    def stateProxy(self, period, requested_props):
        """ A proxy function to ease the transition to state-aware learning.
        
        Accepts the period in question, and returns an index marking the tranche
        of the Q-values corresponding to the demand during that period. If a
        non-state-aware method is being used, just returns the requested props.
        """
        
        if (self.learning_mechanism == 'Stateful_QLearning'):
            
            if (self.single_bm_ladder == True) and (requested_props[0:2] == 'bm'):
            
                return getattr(self, requested_props)
            
            else:
        
                demand = self.demand[period]
                
                demand_tranche = np.argmin(np.abs(np.subtract(self.state_space, demand/self.max_capacity)))
                
                return getattr(self, requested_props)[demand_tranche]
        
        elif (self.single_bm_ladder == True) and (requested_props[0:2] == 'bm'):
        
            return getattr(self, requested_props)

        else:
                
            return getattr(self, requested_props)[period]
        
    
    
    def demandTranche(self, period):
        """ Accepted period, returns demand tranche. Stripped down version of
        
        stateProxy, without the propensity array return.
        """
        
        demand = self.demand[period]
            
        return np.argmin(np.abs(np.subtract(self.state_space, demand/self.max_capacity)))
    
    
    
    def make_px_offer(self):
        """ Applies softmax to price_offer_set based on propensities for each period
        
        and returns a price offer for each period of the next day. Offers are 
        of the form [Volume, Price, Maximum Generation, Minimum Generation]
        """
        
        if self.fuel == 'wind':
            # uses a simple AR(1) with mean-reversion to give a more realistic
            # forecast error term, scaled by the actual wind output so that
            # the errors are not grossly over-pronounced when wind is low
            
            if self.dynamic_imbalance == True:
                
                self.forecast_wind = np.zeros(48)
                
                self.forecast_wind[0] = (self.true_wind[0]/2) * (1 + np.random.normal(scale = self.wind_sd))
                self.forecast_wind[1] = self.forecast_wind[0] + 0.25*(self.true_wind[1]/2 - self.forecast_wind[0])
                
                for i in range(2, 48):
                    
                    self.forecast_wind[i] = self.forecast_wind[i-1]*(1 + np.random.normal(scale = self.wind_sd)) + \
                                       + 0.25*(self.true_wind[i]/2 - self.forecast_wind[i-1])
                                       
                     
            self.px_offer = [[self.forecast_wind[period], 0, self.forecast_wind[period], 0] for period in range(48)]

            return
        
                    
        self.px_offer = []
        

        if self.px_action_method == 'softmax':
            
            self.px_temperature = self.px_temperature_inf + (self.px_temperature_start - self.px_temperature_inf) \
                                  * np.exp(-self.day/(self.days/self.px_temperature_decay)) 
        
            for period in range(48):
                
                px_price_weights = self.softmax(self.stateProxy(period, 'px_price_propensities'), self.px_temperature)
                px_volume_weights = self.softmax(self.stateProxy(period, 'px_volume_propensities'), self.px_temperature)
                    
                if self.fuel == 'nuclear':
                    
                    self.px_offer.append([self.capacity/2, 
                                       choices(self.px_price_offer_set, px_price_weights)[0],
                                       self.capacity/2,
                                       self.capacity/2])

                else:
                    
                    self.px_offer.append([(choices(self.px_volume_offer_set, px_volume_weights)[0]) * self.capacity/2,
                                        choices(self.px_price_offer_set, px_price_weights)[0],
                                        self.capacity/2,
                                        self.min_gen/2])
        
        
        # this one chooses actions that are at or near the max_prop offer, based
        # on a using a kernel to weight the choices near it. Still retains the
        # completely random choice with probability epsilon.
        # Remember to change the size of the actions list if changing radius!
                                  
        elif self.px_action_method == 'epsilon_kernel':
            
            self.px_epsilon = self.px_epsilon_inf + (self.px_epsilon_start - self.px_epsilon_inf) \
                              * np.exp(-self.day/(self.days/self.px_epsilon_decay)) 
            
            price_kernel = self.create_kernel(2, 0.65)
            volume_kernel = self.create_kernel(2, 0.65)
        
            for period in range(48):
                
                if random() > self.px_epsilon:
                    
                    max_price_props = []
                    max_volume_props = []
                    
                    for i, prop in enumerate(self.stateProxy(period, 'px_price_propensities')):
                        if prop == np.max(self.stateProxy(period, 'px_price_propensities')):
                            max_price_props.append(i)
                            
                    for i, prop in enumerate(self.stateProxy(period, 'px_volume_propensities')):
                        if prop == np.max(self.stateProxy(period, 'px_volume_propensities')):
                            max_volume_props.append(i)
                    
                    max_price_prop = choices(max_price_props)[0]
                    max_volume_prop = choices(max_volume_props)[0]
                    
                    price_actions = np.linspace(max_price_prop - 2, max_price_prop + 2, 5)
                    volume_actions = np.linspace(max_volume_prop - 2, max_price_prop + 2, 5)
                    
                    for i, action in enumerate(price_actions):
                        
                        if (action < 0) or (len(self.px_price_offer_set) <= action):
                            
                            price_kernel[i] = 0
                    
                    for i, action in enumerate(volume_actions):
                        
                        if (action < 0) or (21 <= action):
                            
                            volume_kernel[i] = 0
                    
                    if self.fuel == 'nuclear':
                        
                        self.px_offer.append([self.capacity/2, 
                                              self.px_price_offer_set[int(choices(price_actions, price_kernel)[0])],
                                              self.capacity/2,
                                              self.capacity/2])
                        
                    else:
                        
                        self.px_offer.append([(self.px_volume_offer_set[int(choices(volume_actions, volume_kernel)[0])]) * self.capacity/2,
                                               self.px_price_offer_set[int(choices(price_actions, price_kernel)[0])],
                                               self.capacity/2,
                                               self.min_gen/2])

                else:   
                    
                    self.px_offer.append([0.85 * self.capacity/2, 
                                          choices(self.px_price_offer_set)[0],
                                          self.capacity/2,
                                          self.min_gen/2])
            
        
        # modified s.t if max propensities are the same, it chooses randomly from those
        
        elif self.px_action_method == 'epsilon_greedy':
            
            self.px_epsilon = self.px_epsilon_inf + (self.px_epsilon_start - self.px_epsilon_inf) \
                              * np.exp(-self.day/(self.days/self.px_epsilon_decay)) 
            
            for period in range(48):
                
                if random() > self.px_epsilon:
                    
                    max_price_props = []
                    max_volume_props = []
                    
                    for i, prop in enumerate(self.stateProxy(period, 'px_price_propensities')):
                        if prop == np.max(self.stateProxy(period, 'px_price_propensities')):
                            max_price_props.append(i)
                            
                    for i, prop in enumerate(self.stateProxy(period, 'px_volume_propensities')):
                        if prop == np.max(self.stateProxy(period, 'px_volume_propensities')):
                            max_volume_props.append(i)
                            
                    max_price_prop = choices(max_price_props)[0]
                    max_volume_prop = choices(max_volume_props)[0]
                        
                    if self.fuel == 'nuclear':
                        
                        self.px_offer.append([self.capacity/2, 
                                              self.px_price_offer_set[int(max_price_prop)],
                                              self.capacity/2,
                                              self.capacity/2])
                        
                    else:
                        
                        self.px_offer.append([(self.px_volume_offer_set[int(max_volume_prop)]) * self.capacity/2,
                                               self.px_price_offer_set[int(max_price_prop)],
                                               self.capacity/2,
                                               self.min_gen/2])
                    
                                   
                else:
                    
                    if self.fuel == 'nuclear':
                        
                        self.px_offer.append([self.capacity/2, 
                                              choices(self.px_price_offer_set)[0],
                                              self.capacity/2,
                                              self.capacity/2])
                        
                    else:
                        
                        self.px_offer.append([choices(self.px_volume_offer_set)[0] * self.capacity/2,
                                              choices(self.px_price_offer_set)[0],
                                              self.capacity/2,
                                              self.min_gen/2])
    
        elif self.px_action_method == 'SA-Q':
            
            self.px_temperature = self.px_temperature_inf + (self.px_temperature_start - self.px_temperature_inf) \
                                  * np.exp(-self.day/(self.days/self.px_temperature_decay)) 
            
            for period in range(48):
                
                random_price_prop = choices(range(len(self.stateProxy(period, 'px_price_propensities'))))[0]
                random_volume_prop = choices(range(len(self.stateProxy(period, 'px_volume_propensities'))))[0]
                
                max_price_props = []
                max_volume_props = []
                
                for i, prop in enumerate(self.stateProxy(period, 'px_price_propensities')):
                    if prop == np.max(self.stateProxy(period, 'px_price_propensities')):
                        max_price_props.append(i)
                        
                for i, prop in enumerate(self.stateProxy(period, 'px_volume_propensities')):
                    if prop == np.max(self.stateProxy(period, 'px_volume_propensities')):
                        max_volume_props.append(i)
                        
                greedy_price_prop = choices(max_price_props)[0]
                greedy_volume_prop = choices(max_volume_props)[0]
                
                xi_price = m.exp((self.stateProxy(period, 'px_price_propensities')[random_price_prop] - \
                                  self.stateProxy(period, 'px_price_propensities')[greedy_price_prop])/self.px_temperature)
                xi_volume = m.exp((self.stateProxy(period, 'px_volume_propensities')[random_volume_prop] - \
                                   self.stateProxy(period, 'px_volume_propensities')[greedy_volume_prop])/self.px_temperature)
                
                xi_price_compare = random()
                xi_volume_compare = random()
                
                if (xi_price_compare >= xi_price) and (xi_volume_compare >= xi_volume):
                    
                    if self.fuel == 'nuclear':
                            
                        self.px_offer.append([self.capacity/2, 
                                              self.px_price_offer_set[int(greedy_price_prop)],
                                              self.capacity/2,
                                              self.capacity/2])
                        
                    else:
                        
                        self.px_offer.append([(self.px_volume_offer_set[int(greedy_volume_prop)]) * self.capacity/2,
                                               self.px_price_offer_set[int(greedy_price_prop)],
                                               self.capacity/2,
                                               self.min_gen/2])
                
                elif (xi_price_compare >= xi_price) and (xi_volume_compare < xi_volume):
                    
                    if self.fuel == 'nuclear':
                            
                        self.px_offer.append([self.capacity/2, 
                                              self.px_price_offer_set[int(greedy_price_prop)],
                                              self.capacity/2,
                                              self.capacity/2])
                        
                    else:
                        
                        self.px_offer.append([(self.px_volume_offer_set[int(random_volume_prop)]) * self.capacity/2,
                                               self.px_price_offer_set[int(greedy_price_prop)],
                                               self.capacity/2,
                                               self.min_gen/2])
        
                elif (xi_price_compare < xi_price) and (xi_volume_compare >= xi_volume):
                    
                    if self.fuel == 'nuclear':
                            
                        self.px_offer.append([self.capacity/2, 
                                              self.px_price_offer_set[int(random_price_prop)],
                                              self.capacity/2,
                                              self.capacity/2])
                        
                    else:
                        
                        self.px_offer.append([(self.px_volume_offer_set[int(greedy_volume_prop)]) * self.capacity/2,
                                               self.px_price_offer_set[int(random_price_prop)],
                                               self.capacity/2,
                                               self.min_gen/2])
        
                else:
    
                    if self.fuel == 'nuclear':
                            
                        self.px_offer.append([self.capacity/2, 
                                              self.px_price_offer_set[int(random_price_prop)],
                                              self.capacity/2,
                                              self.capacity/2])
                        
                    else:
                        
                        self.px_offer.append([(self.px_volume_offer_set[int(random_volume_prop)]) * self.capacity/2,
                                               self.px_price_offer_set[int(random_price_prop)],
                                               self.capacity/2,
                                               self.min_gen/2])
                
    def construct_bm_ladder(self):
        """ Observes the agent's remaining capacities from the PX, and uses
        
        the bm market propensities to construct a linear function of capcacity
        and price offerings for each period of the day. Creates offers/bids in
        the form [id, volume, price], so that when they are sorted in the 
        market clearing function each bid/offer can be tied to its generator.
        """
        
        bm_offers = []
        bm_bids = []
        self.bm_intercept_choices = {}
        self.bm_gradient_choices = {}
        
        self.bm_temperature = self.bm_temperature_inf + (self.bm_temperature_start - self.bm_temperature_inf) \
                              * np.exp(-self.day/(self.days/self.bm_temperature_decay)) 
          
        self.bm_epsilon = self.bm_epsilon_inf + (self.bm_epsilon_start - self.bm_epsilon_inf) \
                          * np.exp(-self.day/(self.days/self.bm_epsilon_decay)) 
        
        
        if self.single_bm_ladder == True:
            
            if self.bm_action_method == 'softmax':
                
                bm_intercept_weights = self.softmax(self.stateProxy(0, 'bm_intercept_propensities'), self.bm_temperature)
                bm_gradient_weights = self.softmax(self.stateProxy(0, 'bm_gradient_propensities'), self.bm_temperature)
                
                bm_intercept_choice = choices(self.bm_intercept_set, bm_intercept_weights)[0]
                bm_gradient_choice = choices(self.bm_gradient_set, bm_gradient_weights)[0]
            
            elif self.bm_action_method == 'epsilon_greedy':
                
                if random() > self.bm_epsilon:
                        
                        max_intercept_props = []
                        
                        for i, prop in enumerate(self.stateProxy(0, 'bm_intercept_propensities')):
                            if prop == np.max(self.stateProxy(0, 'bm_intercept_propensities')):
                                max_intercept_props.append(i)
                        
                        bm_intercept_choice = self.bm_intercept_set[choices(max_intercept_props)[0]]
                    
                else:
                        
                    bm_intercept_choice = choices(self.bm_intercept_set)[0]
                    
                if random() > self.bm_epsilon:
                    
                    max_gradient_props = []
                    
                    for i, prop in enumerate(self.stateProxy(0, 'bm_gradient_propensities')):
                        if prop == np.max(self.stateProxy(0, 'bm_gradient_propensities')):
                            max_gradient_props.append(i)
                    
                    bm_gradient_choice = self.bm_gradient_set[choices(max_gradient_props)[0]]
                
                else:
                    
                    bm_gradient_choice = choices(self.bm_gradient_set)[0]
         
            for period in range(48):
                       
                if self.bm_gradient_hold != 0:
                    
                    self.bm_gradient_choices[period] = self.bm_gradient_hold
                    self.bm_intercept_choices[period] = bm_intercept_choice
                
                else:
                    
                    self.bm_gradient_choices[period] = bm_gradient_choice
                    self.bm_intercept_choices[period] = bm_intercept_choice
        
        else: 
            
            for period in range(48):
                
                if self.bm_action_method == 'softmax':
                
                    bm_intercept_weights = self.softmax(self.stateProxy(period, 'bm_intercept_propensities'), self.bm_temperature)
                    bm_gradient_weights = self.softmax(self.stateProxy(period, 'bm_gradient_propensities'), self.bm_temperature)
    
                    self.bm_intercept_choices[period] = choices(self.bm_intercept_set, bm_intercept_weights)[0]
                    self.bm_gradient_choices[period] = choices(self.bm_gradient_set, bm_gradient_weights)[0]
                
                elif self.bm_action_method == 'epsilon_greedy':
                    
                    if random() > self.bm_epsilon:
                        
                        max_intercept_props = []
                        
                        for i, prop in enumerate(self.stateProxy(period, 'bm_intercept_propensities')):
                            if prop == np.max(self.stateProxy(period, 'bm_intercept_propensities')):
                                max_intercept_props.append(i)
                        
                        self.bm_intercept_choices[period] = self.bm_intercept_set[choices(max_intercept_props)[0]]
                    
                    else:
                        
                        self.bm_intercept_choices[period] = choices(self.bm_intercept_set)[0]
                    
                    if random() > self.bm_epsilon:
                        
                        max_gradient_props = []
                        
                        for i, prop in enumerate(self.stateProxy(period, 'bm_gradient_propensities')):
                            if prop == np.max(self.stateProxy(period, 'bm_gradient_propensities')):
                                max_gradient_props.append(i)
                        
                        self.bm_gradient_choices[period] = self.bm_gradient_set[choices(max_gradient_props)[0]]
                    
                    else:
                        
                        self.bm_gradient_choices[period] = choices(self.bm_gradient_set)[0]
                
                if self.bm_gradient_hold != 0:
                    
                    self.bm_gradient_choices[period] = self.bm_gradient_hold
                
        
        for period in range(48):
            
            bm_offer_volume = self.bm_available_volume[period][0]
            bm_bid_volume = self.bm_available_volume[period][1]
        
            bm_bids.append([[self.id, round(-0.05*bm_bid_volume, 2), round(self.bm_intercept_choices[period] - self.bm_gradient_choices[period], 2)],
                            [self.id, round(-0.05*bm_bid_volume, 2), round(self.bm_intercept_choices[period] - 2*self.bm_gradient_choices[period], 2)],
                            [self.id, round(-0.15*bm_bid_volume, 2), round(self.bm_intercept_choices[period] - 3*self.bm_gradient_choices[period], 2)],
                            [self.id, round(-0.25*bm_bid_volume, 2), round(self.bm_intercept_choices[period] - 4*self.bm_gradient_choices[period], 2)],
                            [self.id, round(-0.5*bm_bid_volume, 2), round(self.bm_intercept_choices[period] - 5*self.bm_gradient_choices[period], 2)]])
            
            bm_offers.append([[self.id, round(0.05*bm_offer_volume, 2), round(self.bm_intercept_choices[period] + self.bm_gradient_choices[period], 2)],
                              [self.id, round(0.05*bm_offer_volume, 2), round(self.bm_intercept_choices[period] + 2*self.bm_gradient_choices[period], 2)],
                              [self.id, round(0.15*bm_offer_volume, 2), round(self.bm_intercept_choices[period] + 3*self.bm_gradient_choices[period], 2)],
                              [self.id, round(0.25*bm_offer_volume, 2), round(self.bm_intercept_choices[period] + 4*self.bm_gradient_choices[period], 2)],
                              [self.id, round(0.5*bm_offer_volume, 2), round(self.bm_intercept_choices[period] + 5*self.bm_gradient_choices[period], 2)]])
    
            
        return bm_bids, bm_offers
                        
        
    def update_px_propensities(self, generation, px_marginal_price):
        """ Takes the dispatched generation for each period of the previous
        day, returns an array of profit made in each period, and updates
        propensities using the modified Roth-Erev algorithm. Returns total
        and period profits for each day.
        """
        
        # generation is of the form period: dict{agent_id: [fuel, dispatch, offer]}
        # extract px_dispatch specific to the generator in question. This one 
        # DOES contain offers that are constrained to 1
        
        self.generation = []
        for period in range(48):
            self.generation.append(generation[period][self.id][1:3])
        
        # this part adds a startup cost whenever the accepted offers move from
        # 0 to 1, to incentivise nuclear to stay on for example. The penalty is
        # however applied to the period *before* startup so that the agent
        # is aware of where it needs to bid lower to avoid that cost again
        
        if self.fuel == 'wind':
            self.day += 1
            self.px_day_profit = 0
            self.px_period_profit = 0
            self.max_propensities = 0
            self.bm_bids = []
            self.bm_offers = []
            self.bm_intercept_choices = []
            self.bm_gradient_choices = []
            self.startup_penalties = list(np.zeros(48))
            return
        
        self.startup_penalties = []
        
        for period, dispatch in enumerate(self.generation[1:]):
            if (self.generation[period + 1][0] != 0) and (self.generation[period][0] == 0):
                self.startup_penalties.append(self.startup_cost)
            else:
                self.startup_penalties.append(0)
                
        self.startup_penalties.append(0)
        
        # remember that the price_offer here is not constrained to 1 if nuclear,
        # hence why profit is calculated using the figures from self.generation
        self.px_price_offer = [x[1] for x in self.px_offer]
        
        self.px_period_profit = np.multiply(np.subtract([x[1] for x in self.generation], 
                                                     self.marginal_cost), [x[0] for x in self.generation])
        self.px_period_profit -= np.array(self.startup_penalties)
        
        self.px_day_profit = sum(self.px_period_profit)
        
        # construct list of volumes carried forward to the balancing market,
        # and pass to the bm_offer_ladder function to create the offer ladder.
        # bm_offer_volumes is of the form [offer volume, bid volume]. If both
        # 0, then generator was not dispatched in the px and hence does not
        # participate in the balancing mechanism
        
        if self.balancing_mechanism == True:
            
            self.bm_available_volume = {}
            
            for period in range(48):
                
                if (self.generation[period][0] == 0) or (self.fuel == 'nuclear'):
                    
                    self.bm_available_volume[period] = [0, 0]
                
                else:
                    
                    self.bm_available_volume[period] = [self.capacity/2 - self.generation[period][0],
                                                        self.generation[period][0] - self.min_gen/2]
                
            self.bm_bids, self.bm_offers = self.construct_bm_ladder()
        
        else:
            
            self.day += 1
                    
        
        # convert period profit array into a reward array, that penalises generators
        # if they are not dispatched equal to the difference between their offer
        # and the system marginal cost for that period. also penalises generators 
        # if they exceed their cycling limits. This is somewhat arbitrary right 
        # now, will need to fiddle with value
        
        self.px_reward = deepcopy(self.px_period_profit)
        
        if len(px_marginal_price) != 48:
            print(px_marginal_price)
            raise Exception
                
        for period in range(48):
            if self.generation[period][0] == 0:
                self.px_reward[period] += (px_marginal_price[period] - self.px_price_offer[period]) * self.capacity/2
                if (len(self.startup_penalties) - self.startup_penalties.count(0)) > self.cycles:
                    self.px_reward[period] -= (len(self.startup_penalties) - self.startup_penalties.count(0)) * self.capacity * 2
        
        # add offer statistics and average profit to expected_profit dictionary
        # offer: [offer frequency, offer successes, success ratio, 
        # average profit, expected profit, change from previous day, average
        # reward, expected reward]
        
        self.expected_profits = deepcopy(self.expected_profits)
      
        for period, offer_dispatch in enumerate(self.generation):
            
            previous_expectation = self.expected_profits[self.px_price_offer[period]][period][4]
            
            self.expected_profits[self.px_price_offer[period]][period][0] += 1
            if offer_dispatch[0] != 0:
                self.expected_profits[self.px_price_offer[period]][period][1] += 1
            self.expected_profits[self.px_price_offer[period]][period][2] = self.expected_profits[self.px_price_offer[period]][period][1] / self.expected_profits[self.px_price_offer[period]][period][0]
            self.expected_profits[self.px_price_offer[period]][period][3] = (self.expected_profits[self.px_price_offer[period]][period][0] * self.expected_profits[self.px_price_offer[period]][period][3] \
                                                               + self.px_period_profit[period])/(self.expected_profits[self.px_price_offer[period]][period][0] + 1)
        
            self.expected_profits[self.px_price_offer[period]][period][4] = self.expected_profits[self.px_price_offer[period]][period][3] * self.expected_profits[self.px_price_offer[period]][period][2]
            self.expected_profits[self.px_price_offer[period]][period][5] = self.expected_profits[self.px_price_offer[period]][period][4] - previous_expectation
            self.expected_profits[self.px_price_offer[period]][period][6] = (self.expected_profits[self.px_price_offer[period]][period][0] * self.expected_profits[self.px_price_offer[period]][period][6] \
                                                               + self.px_reward[period])/(self.expected_profits[self.px_price_offer[period]][period][0] + 1)
            self.expected_profits[self.px_price_offer[period]][period][7] = self.expected_profits[self.px_price_offer[period]][period][6] * self.expected_profits[self.px_price_offer[period]][period][2]
                
        
        # updates propensities according to selected learning algorithm
        
        self.px_price_propensities = deepcopy(self.px_price_propensities)
        
        if self.px_reward_method == 'period_profit':
        
            self.px_reward /= self.dampener
        
        elif self.px_reward_method == 'day_profit':
            
            self.px_reward = sum(self.px_reward)/(self.dampener * 48)       
                         
        elif self.px_reward_method == 'kernel_profit':
            
            kernel = self.create_kernel(self.kernel_radius, self.discount)
            self.px_reward = np.convolve(self.px_reward, kernel, mode = 'same')/self.dampener
                   
        elif self.px_reward_method == 'expected_profit':
            
            self.px_reward = np.divide([self.expected_profits[self.px_price_offer[period]][period][5] for period in range(48)],
                                        self.dampener)          
            
                  
        for period in range(48):
            
            if self.learning_mechanism == 'Stateful_QLearning':
        
                state_proxy = self.demandTranche(period)
                
            else:
                
                state_proxy = period
            
            chosen_price = self.px_price_offer_set.index(self.px_price_offer[period])
            
            self.px_price_propensities[state_proxy] = self.px_price_RL.update(props = self.px_price_propensities[state_proxy],
                                                                              reward = self.px_reward[period],
                                                                              chosen_action = chosen_price)
        
        
        
    def update_bm_propensities(self, generation):
        """ Takes the successful dispatches from the balancing market, and uses
        
        them to update the propensities of both the BM offer ladder and the 
        volume offered in the PX, which has an outsized effect on the profit
        made in the BM and hence is updated here.
        """
       
        # generation is of the form period: dict{agent_id: [fuel, px_dispatch, 
        # px_offer, bm_dispatch, [list of successful bm bids/offers], total
        # dispatch]}
        # extract dispatch specific to the generator in question. This one 
        # DOES contain offers that are constrained to 1
         
        if (self.fuel == 'wind') or (self.fuel == 'nuclear'):
            
            self.day += 1
            self.bm_period_profit = [0 for i in range(48)]
            self.bm_day_profit = 0

            return

        self.generation = []
        for period in range(48):
            self.generation.append(generation[period][self.id][1:])
            
        # remember that the price_offer here is not constrained to 1 if nuclear,
        # hence why profit is calculated using the figures from self.generation
        # also, rewards for volume are given to the post-constrained volume
        # dispatch, rather than the actual volume offer that was made.
        
        self.true_volume_offer = [x[0] for x in self.px_offer]
        self.constrained_volume_offer = [x[0] for x in self.generation]
        
        self.accepted_bm_actions = [x[3] for x in self.generation]
        
        self.bm_period_profit = [sum([x[1] * x[2] for x in actions]) - (self.marginal_cost * self.generation[period][2]) for period, actions in enumerate(self.accepted_bm_actions)]
        
        self.bm_day_profit = sum(self.bm_period_profit)        
        
        self.px_volume_propensities = deepcopy(self.px_volume_propensities)
        self.bm_intercept_propensities = deepcopy(self.bm_intercept_propensities)
        self.bm_gradient_propensities = deepcopy(self.bm_gradient_propensities)
        
        
        # shape the reward function based on bm profits per period
        
        if self.bm_reward_method == 'kernel_profit':
            
            kernel = self.create_kernel(self.kernel_radius, self.discount)
            self.bm_reward = np.divide(np.convolve(self.bm_period_profit, kernel, mode = 'same'), self.dampener)
            
        
        elif self.bm_reward_method == 'period_profit':
            
            self.bm_reward = np.divide(self.bm_period_profit, self.dampener)
        
        # if a single bm-ladder is being used, then the day profit is used by default
        self.bm_day_reward = self.bm_day_profit/(self.dampener * 48)
                
        # update propensities according to selected learning algorithm
        
        for period in range(48):
            
            if self.learning_mechanism == 'Stateful_QLearning':
            
                state_proxy = self.demandTranche(period)
            
            else:
            
                state_proxy = period
            
            chosen_volume = np.argmin(np.abs(np.subtract(self.px_volume_offer_set, (self.constrained_volume_offer[period] /(self.capacity/2)))))

            chosen_intercept = self.bm_intercept_set.index(self.bm_intercept_choices[period])
            chosen_gradient = self.bm_gradient_set.index(self.bm_gradient_choices[period])
            
            self.px_volume_propensities[state_proxy] = self.px_volume_RL.update(props = self.px_volume_propensities[state_proxy],
                                                                                reward = self.bm_reward[period],
                                                                                chosen_action = chosen_volume)
            
            if (self.single_bm_ladder == True) and (period == 0):
                
                self.bm_intercept_propensities = self.bm_intercept_RL.update(props = self.bm_intercept_propensities,
                                                                                     reward = self.bm_day_reward,
                                                                                     chosen_action = chosen_intercept)
                
                self.bm_gradient_propensities = self.bm_gradient_RL.update(props = self.bm_gradient_propensities,
                                                                                   reward = self.bm_day_reward,
                                                                                   chosen_action = chosen_gradient)
                
            elif self.single_bm_ladder == False:
                
                self.bm_intercept_propensities[state_proxy] = self.bm_intercept_RL.update(props = self.bm_intercept_propensities[state_proxy],
                                                                                          reward = self.bm_reward[period],
                                                                                          chosen_action = chosen_intercept)
                
                self.bm_gradient_propensities[state_proxy] = self.bm_gradient_RL.update(props = self.bm_intercept_propensities[state_proxy],
                                                                                        reward = self.bm_reward[period],
                                                                                        chosen_action = chosen_gradient)
    
        self.day += 1
        
    
    
    
class Supplier():

    def __init__(self, supplier_params, demand_profile, params):
        
        self.market_share = supplier_params[0]
        self.forecast_error = supplier_params[1]
        self.strategy = supplier_params[2]
        
        self.dynamic_imbalance = params['dynamic_imbalance']
                
        if params['dynamic_imbalance'] == False:
            
            self.demand_forecast = (demand_profile/2 * self.market_share) * (1 + np.random.normal(scale = self.forecast_error, size = 48))
    
    def make_bid(self, demand, period):
        """ Accepts the true demand profile for the day and the period in which
        
        the market is to be cleared, in MW. Returns a tuple of forecast demand
        and the bid level.
        """
        if self.dynamic_imbalance == True:
            
            demand_forecast = (demand[period]/2 * self.market_share) * (1 + np.random.normal(scale = self.forecast_error))
        
        else:
        
            demand_forecast = self.demand_forecast[period]
            
        if self.strategy == 0:
            
            bid = choices([30, 40, 50])[0]
        
        elif self.strategy == 1:
            
            bid = 80 * demand[period]/max(demand)
            
        elif self.strategy == 2:
            
            bid = 30 * (1 + demand[period]/max(demand))
            
        return (demand_forecast, bid)
