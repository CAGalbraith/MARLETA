# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:20:15 2018

@author: conno
"""

import numpy as np
from random import choices, random

import mesa_modified as mesa


class Generator(mesa.Agent):
    """ Properties should be assigned in model.py as simulation parameters, using

    the batch runner to iteratre when need, as well as the form of the markup set. 
    gen_id increments with each realisation of Generator and is assigned to each 
    instance as a unique identifier.
    """

    def __init__(self, model, fuel_id, params):

        super().__init__(fuel_id[1], model)

        # internalise simulation length to calculate temperature decay rate
        self.days = params['days']
        
        # keeps track of days for the increment offer method, may deprecate
        self.day = 0
        
        # stores phyiscal parameters of the agent
        self.fuel = fuel_id[0]
        self.id = fuel_id[1]
        self.capacity = params[fuel_id[0]]['cap']
        self.marginal_cost = params[fuel_id[0]]['marginal_cost']
        self.startup_cost = params[fuel_id[0]]['startup_cost']
        self.cycles = params[fuel_id[0]]['cycles']
        self.min_gen = params[fuel_id[0]]['min_gen']
        self.emissions = params[fuel_id[0]]['emissions']
        
        # these two control the up/down time constraints
        self.run_time = -100
        self.last_dispatch = 0
        
        # initialises the true wind profile for the given simulation day
        if self.fuel == 'wind':
            self.wind_sd = params['wind_sd']
            self.wind_profile = params['wind_profile']
        
        # learning parameters
        self.action_method = params['action_method']
        self.offer_method = params['offer_method']
        
        self.reward_method = params['reward_method']
        self.discount = params['discount']
        self.dampening_factor = params['dampening_factor']
        self.kernel_radius = params['kernel_radius']
        
        self.epsilon = params['epsilon']
        self.recency = params['recency']
        self.expmt = params['expmt']
        
        self.temperature_inf = params['temperature_inf']
        self.temperature_start = params['temperature_start']
        self.temperature_decay = params['temperature_decay']

        
        # offer_set very important. Limit to marginal_cost or not?
        
        if params['offer_method'] == 'absolute':
            
            self.price_offer_set = list(np.linspace(25, 125, 21))
        
        elif params['offer_method'] == 'increment':
            
            self.price_offer_set = list(np.linspace(0.5, 1.5, 11))
            
        self.volume_offer_set = list(np.linspace(50, 100, 11))
        
        # initialises propensities for PX price, PX quantities, and BM ladder
        # volume offers go from 80% to 100% of capacity, = 21 options
        
        self.px_price_propensities = np.ones((48, len(self.price_offer_set)))
        self.px_volume_propensities = np.ones((48, len(self.volume_offer_set)))
        self.bm_propensities = np.ones(9)
        
        # reward signal scaled to the max profit the generator can make in a 
        # period, to make the reward independent of physical parameters
        
        self.dampener = self.capacity * (self.price_offer_set[-1] - self.marginal_cost)/self.dampening_factor
        
        # keeps track of the offers the agent makes, their success rate, and
        # the average profit made from each to inform future actions. Each offer
        # key links to a list of the form [offer frequency, offer successes,
        # success ratio, average profit, expected_profit]
        
        self.expected_profits = {offer: [0, 0, 0, 0, 0] for offer in self.price_offer_set}
    
   
    def constraint_mode(self):
        """ Keeps track of generator up/down time, returns whether or not the 
        
        generator is constrained during a period
        """
        
        if self.run_time <= -self.min_down or self.run_time >= self.min_up or self.run_time == 0:
            return('none')
        if self.run_time < 0 and self.run_time > -self.min_down:
            return('off')
        if self.run_time > 0 and self.run_time < self.min_up:
            return('on')

        
        
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
    
    
    
    def make_offer(self):
        """ Applies softmax to price_offer_set based on propensities for each period
        
        and returns a price offer for each period of the next day. Offers are 
        of the form [Volume, Price, Maximum Generation, Minimum Generation]
        """
        
        # TODO: use wind_sd to give some semblence of forecast error
        if self.fuel == 'wind':
            self.offer = [[self.wind_profile[period]/2, 0, self.wind_profile[period]/2, 0] for period in range(48)]
            return
        
        if (self.offer_method == 'increment' and self.day == 0):
            self.previous_offer = [(self.capacity, self.marginal_cost) for period in range(48)]
            
            
        self.temperature = self.temperature_inf + (self.temperature_start - self.temperature_inf) * np.exp(-self.day/(self.days/self.temperature_decay)) 
        
        self.offer = []
        
        
        # this is a hack for now. Markup is used in the Roth-Erev to keep track
        # of agent choices whether it's incremental or absolute. May need to 
        # change some of the checks in update_propensities if using incremental
        
        self.markup = [] 

        if self.action_method == 'softmax':
        
            for period in range(48):
                
                px_price_weights = self.softmax(self.px_price_propensities[period, :], self.temperature)
                px_volume_weights = self.softmax(self.px_volume_propensities[period, :], self.temperature)
                
                if self.offer_method == 'absolute':
                    
                    if self.fuel == 'nuclear':
                        
                        self.offer.append([self.capacity/2, 
                                           choices(self.price_offer_set, px_price_weights)[0],
                                           self.capacity/2,
                                           self.capacity/2])
                        self.markup.append(self.offer[-1])
                        
                    else:
                        
                        self.offer.append([(choices(self.volume_offer_set, px_volume_weights)[0]/100) * self.capacity/2,
                                            choices(self.price_offer_set, px_price_weights)[0],
                                            self.capacity/2,
                                            self.min_gen/2])
                        self.markup.append(self.offer[-1])
                    
                elif self.offer_method == 'increment':
                    
                    self.markup.append(choices(self.price_offer_set, px_price_weights)[0])
                    self.offer.append((self.capacity/2, self.markup[period] * self.previous_offer[period][1]))
        
        
        # this one chooses actions that are at or near the max_prop offer, based
        # on a using a kernel to weight the choices near it. Still retains the
        # completely random choice with probability epsilon.
        # Remember to change the size of the actions list if changing radius!
        
        elif self.action_method == 'epsilon_kernel':
            
            price_kernel = self.create_kernel(2, 0.65)
            volume_kernel = self.create_kernel(2, 0.65)
        
            for period in range(48):
                
                if random() > self.epsilon:
                    
                    max_price_props = []
                    max_volume_props = []
                    
                    for i, prop in enumerate(self.px_price_propensities[period]):
                        if prop == np.max(self.px_price_propensities[period]):
                            max_price_props.append(i)
                            
                    for i, prop in enumerate(self.px_volume_propensities[period]):
                        if prop == np.max(self.px_volume_propensities[period]):
                            max_volume_props.append(i)
                    
                    max_price_prop = choices(max_price_props)[0]
                    max_volume_prop = choices(max_volume_props)[0]
                    
                    price_actions = np.linspace(max_price_prop - 2, max_price_prop + 2, 5)
                    volume_actions = np.linspace(max_volume_prop - 2, max_price_prop + 2, 5)
                    
                    for i, action in enumerate(price_actions):
                        
                        if (action < 0) or (len(self.price_offer_set) <= action):
                            
                            price_kernel[i] = 0
                    
                    for i, action in enumerate(volume_actions):
                        
                        if (action < 0) or (21 <= action):
                            
                            volume_kernel[i] = 0
                    
                    if self.fuel == 'nuclear':
                        
                        self.offer.append([self.capacity/2, 
                                           self.price_offer_set[int(choices(price_actions, price_kernel)[0])],
                                           self.capacity/2,
                                           self.capacity/2])
                        
                    else:
                        
                        self.offer.append([(self.volume_offer_set[int(choices(volume_actions, volume_kernel)[0])]/100) * self.capacity/2,
                                           self.price_offer_set[int(choices(price_actions, price_kernel)[0])],
                                           self.capacity/2,
                                           self.min_gen/2])

                else:   
                    
                    self.offer.append([0.85 * self.capacity/2, 
                                       choices(self.price_offer_set)[0],
                                       self.capacity/2,
                                       self.min_gen/2])
            
        
        # TODO: update epsilon_greedy to include quantity offers too                
        # modified s.t if max propensities are the same, it chooses randomly from those
        
        elif self.action_method == 'epsilon_greedy':
            
            for period in range(48):
                
                if random() > self.epsilon:
                    
                    max_props = []
                    
                    for i, prop in enumerate(self.px_price_propensities[period]):
                        if prop == np.max(self.px_price_propensities[period]):
                            max_props.append(i)
                    
                    if self.offer_method == 'absolute':
                        
                        self.offer.append([self.capacity/2, self.price_offer_set[choices(max_props)[0]]])
                        self.markup.append(self.offer[-1])
                    
                    elif self.offer_method == 'increment':
                        
                        self.markup.append[self.price_offer_set[choices(max_props)[0]]]
                        self.offer.append([self.capacity/2, self.markup[period] * self.previous_offer[period][1]])
                                    
                else:
                    
                    if self.offer_method == 'absolute':
                        
                        self.offer.append([self.capacity/2, choices(self.price_offer_set)[0]])
                        self.markup.append(self.offer[-1])
                        
                    elif self.offer_method == 'increment':
                        
                        self.markup.append(choices(self.price_offer_set)[0])
                        self.offer.append([self.capacity/2, self.markup[period] * self.previous_offer[period][1]])
                            
        
    def update_px_propensities(self, generation, system_marginal_cost):
        """ Takes the dispatched generation for each period of the previous

        day, returns an array of profit made in each period, and updates
        propensities using the modified Roth-Erev algorithm. Returns total
        and period profits for each day.
        """
        
        # generation is of the form period: dict{agent_id: [fuel, dispatch, offer]}
        # extract dispatch specific to the generator in question. This one 
        # DOES contain offers that are constrained to 1
        
        self.generation = []
        for period in range(48):
            self.generation.append(generation[period][self.id][1:])
        
        # this part adds a startup cost whenever the accepted offers move from
        # 0 to 1, to incentivise nuclear to stay on for example. The penalty is
        # however applied to the period *before* startup so that the agent
        # is aware of where it needs to bid lower to avoid that cost again
        
        if self.fuel == 'wind':
            self.day += 1
            self.day_profit = 0
            self.period_profit = 0
            self.max_propensities = 0
            return
        
        startup_penalties = []
        
        for period, dispatch in enumerate(self.generation[1:]):
            if (self.generation[period + 1][0] != 0) and (self.generation[period][0] == 0):
                startup_penalties.append(self.startup_cost)
            else:
                startup_penalties.append(0)
                
        startup_penalties.append(0)
        
        # remember that the price_offer here is not constrained to 1 if nuclear,
        # hence why profit is calculated using the figures from self.generation
        self.price_offer = [x[1] for x in self.offer]
        
        self.period_profit = np.multiply(np.subtract([x[1] for x in self.generation], 
                                                     self.marginal_cost), [x[0] for x in self.generation])
        self.period_profit -= np.array(startup_penalties)
        
        self.day_profit = sum(self.period_profit)
        
        # add offer statistics and average profit to expected_profit dictionary
        # offer: [offer frequency, offer successes, success ratio, 
        # average profit, expected profit]
        
        for period, offer_dispatch in enumerate(self.generation):
            
            self.expected_profits[self.price_offer[period]][0] += 1
            if offer_dispatch[0] != 0:
                self.expected_profits[self.price_offer[period]][1] += 1
            self.expected_profits[self.price_offer[period]][2] = self.expected_profits[self.price_offer[period]][1] / self.expected_profits[self.price_offer[period]][0]
            self.expected_profits[self.price_offer[period]][3] = (self.expected_profits[self.price_offer[period]][0] * self.expected_profits[self.price_offer[period]][3] \
                                                             + self.period_profit[period])/(self.expected_profits[self.price_offer[period]][0] + 1)
        
            self.expected_profit[self.price_offer[period]][4] = self.expected_profits[self.price_offer[period]][3] * self.expected_profits[self.price_offer[period]][2]
        
        # construct list of volumes carried forward to the balancing market
        
        self.bm_offer_volumes = [self.capacity/2 - self.generation]
        
        # convert period profit array into a reward array, that penalises generators
        # if they are not dispatched equal to the difference between their offer
        # and the system marginal cost for that period. also penalises generators 
        # if they exceed their cycling limits. This is somewhat arbitrary right 
        # now, will need to fiddle with value
        
        self.reward = self.period_profit
                
        for period in range(48):
            if self.generation[period][0] == 0:
                self.reward[period] += (system_marginal_cost[period] - self.price_offer[period]) * self.capacity/2
                if (len(startup_penalties) - startup_penalties.count(0)) > self.cycles:
                    self.reward[period] -= (len(startup_penalties) - startup_penalties.count(0)) * self.capacity * 2
        
        
        # updates propensities according to Roth-Erev
        
        if self.reward_method == 'period_profit':
        
            for period, reward in enumerate(self.reward):
                
                for j, prop in enumerate(self.px_price_propensities[period]):
                    
                    if j == self.price_offer_set.index(self.price_offer[period]):
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * reward/(self.dampener)
                        
                    else:
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.price_offer_set) - 1))
        
        elif self.reward_method == 'day_profit':
            
            for period in range(48):
                
                for j, prop in enumerate(self.px_price_propensities[period]):
                    
                    if j == self.price_offer_set.index(self.price_offer[period]):
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * sum(reward)/(self.dampener * 48)
                        
                    else:
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.price_offer_set) - 1))
                        
        elif self.reward_method == 'discounted_profit':
            
            for period in range(48):
                
                discounted_reward = sum(reward * self.discount ** n for n, reward in enumerate(self.reward[period:]))
                
                for j, prop in enumerate(self.px_price_propensities[period]):
                    
                    if j == self.price_offer_set.index(self.price_offer[period]):
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * discounted_reward/(self.dampener * (48 - period))
                        
                    else:
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.price_offer_set) - 1))
        
        elif self.reward_method == 'kernel_profit':
            
            kernel = self.create_kernel(self.kernel_radius, self.discount)
            kernel_reward = np.convolve(self.reward, kernel, mode = 'same')
            
            for period in range(48):
                
                for j, prop in enumerate(self.px_price_propensities[period]):
                    
                    if j == self.price_offer_set.index(self.price_offer[period]):
                                                    
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * kernel_reward[period]/(self.dampener * sum(kernel))
                        
                    else:
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.price_offer_set) - 1))
                        
            
        self.max_propensities = [np.argmax(x) for x in self.px_price_propensities]
        
        self.previous_offer = self.offer
        
#        self.day += 1
        
        
    def update_bm_propensities(self, generation, bm_generation):
        """ Takes the successful dispatches from the balancing market, and uses
        
        them to update the propensities of both the BM offer ladder and the 
        volume offered in the PX, which has an outsized effect on the profit
        made in the BM and hence is updated here.
        
        """
        # For now, this only updates the volume offers of the agent during the
        # PX. Once the BM clearing mechanisms is added, that will be added also.
        
        
        # generation is of the form period: dict{agent_id: [fuel, dispatch, offer]}
        # extract dispatch specific to the generator in question. This one 
        # DOES contain offers that constrained to 1
        
        
        if self.fuel == 'wind':
            
            self.day += 1

            return

        
        # remember that the price_offer here is not constrained to 1 if nuclear,
        # hence why profit is calculated using the figures from self.generation
        
        self.volume_offer = [x[0] for x in self.offer]
    
        self.bm_reward = self.period_profit
        
        if self.reward_method == 'kernel_profit':
            
            kernel = self.create_kernel(self.kernel_radius, self.discount)
            kernel_reward = np.convolve(self.reward, kernel, mode = 'same')
            
            for period in range(48):
                
                for j, prop in enumerate(self.px_volume_propensities[period]):
                    
                    if j == self.volume_offer_set.index((self.volume_offer[period] * 100 /(self.capacity/2)).__round__()):
                                                    
                        self.px_volume_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * kernel_reward[period]/(self.dampener * sum(kernel))
                        
                    else:
                        
                        self.px_volume_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / 20)
                        
        
        self.day += 1
        
    
    
    
class Supplier():

    def __init__(self, params):
        
        self.market_share = params[0]
        self.forecast_error = params[1]
        self.strategy = params[2]
        
        self.demand = 0
    
    def make_bid(self, demand, period):
        """ Accepts the true demand profile for the day and the period in which
        
        the market is to be cleared, in MW. Returns a tuple of forecast demand
        and the bid level.
        """
        
        demand_forecast = (demand[period]/2 * self.market_share) * (1 + np.random.normal(scale = self.forecast_error))
        
        if self.strategy == 0:
            
            bid = choices([30, 40, 50])[0]
        
        elif self.strategy == 1:
            
            bid = 80 * demand[period]/max(demand)
            
        elif self.strategy == 2:
            
            bid = 30 * (1 + demand[period]/max(demand))
            
        return (demand_forecast, bid)
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
    
    
    