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
#        self.last_dispatch = 0
        
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
        
        # list of increment tuples for the intercept and gradient values,
        # and initialises the first intercept and gradient choices via index
        
        self.bm_increment_set = [(-1, -1), (-1, 0), (-1, 1),
                                 (0, -1), (0, 0), (0, 1),
                                 (1, -1), (1, 0), (1, 1)]
        
        self.bm_intercept_set = list(np.linspace(5, 50, 46))
        self.bm_gradient_set = list(np.linspace(0, 10, 51))
        
        self.bm_intercept_choices = [choices(np.linspace(18, 28, 11))[0] for i in range(48)]
        self.bm_gradient_choices = [choices(np.linspace(20, 30, 11))[0] for i in range(48)]
        
        # initialises propensities for PX price, PX quantities, and BM ladder,
        # which is given a kernel centered on no change ([0 ,0])
        
        self.px_price_propensities = np.ones((48, len(self.price_offer_set)))
        self.px_volume_propensities = np.ones((48, len(self.volume_offer_set)))
        self.bm_propensities = [list(self.create_kernel(4, 0.7) * 2) for i in range(48)]
        
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
    
    
    
    def make_px_offer(self):
        """ Applies softmax to price_offer_set based on propensities for each period
        
        and returns a price offer for each period of the next day. Offers are 
        of the form [Volume, Price, Maximum Generation, Minimum Generation]
        """
        
        if self.fuel == 'wind':
            # uses a simple AR(2) with mean-reversion to give a more realistic
            # forecast error term
            
            wind_errors = np.zeros(48)
            wind_errors[0] = np.random.normal(scale = self.wind_sd)
            wind_errors[1] = 0.2 * wind_errors[0] + 0.5 * np.random.normal(scale = self.wind_sd) 
            for i in range(2, 48):
                wind_errors[i] = wind_errors[i-1] + 0.2*wind_errors[i-2] + 0.25*(0 - wind_errors[i-1]) + 0.25*np.random.normal(scale = self.wind_sd)
            
            self.px_offer = [[(self.wind_profile[period]/2) + wind_errors[period], 0, (self.wind_profile[period]/2) + wind_errors[period], 0] for period in range(48)]
 
            return
        
        if (self.offer_method == 'increment' and self.day == 0):
            self.previous_offer = [(self.capacity, self.marginal_cost) for period in range(48)]
            
            
        self.temperature = self.temperature_inf + (self.temperature_start - self.temperature_inf) * np.exp(-self.day/(self.days/self.temperature_decay)) 
        
        self.px_offer = []
        
        
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
                        
                        self.px_offer.append([self.capacity/2, 
                                           choices(self.price_offer_set, px_price_weights)[0],
                                           self.capacity/2,
                                           self.capacity/2])
                        self.markup.append(self.px_offer[-1])
                        
                    else:
                        
                        self.px_offer.append([(choices(self.volume_offer_set, px_volume_weights)[0]/100) * self.capacity/2,
                                            choices(self.price_offer_set, px_price_weights)[0],
                                            self.capacity/2,
                                            self.min_gen/2])
                        self.markup.append(self.px_offer[-1])
                    
                elif self.offer_method == 'increment':
                    
                    self.markup.append(choices(self.price_offer_set, px_price_weights)[0])
                    self.px_offer.append((self.capacity/2, self.markup[period] * self.previous_offer[period][1]))
        
        
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
                        
                        self.px_offer.append([self.capacity/2, 
                                           self.price_offer_set[int(choices(price_actions, price_kernel)[0])],
                                           self.capacity/2,
                                           self.capacity/2])
                        
                    else:
                        
                        self.px_offer.append([(self.volume_offer_set[int(choices(volume_actions, volume_kernel)[0])]/100) * self.capacity/2,
                                           self.price_offer_set[int(choices(price_actions, price_kernel)[0])],
                                           self.capacity/2,
                                           self.min_gen/2])

                else:   
                    
                    self.px_offer.append([0.85 * self.capacity/2, 
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
                        
                        self.px_offer.append([self.capacity/2, self.price_offer_set[choices(max_props)[0]]])
                        self.markup.append(self.px_offer[-1])
                    
                    elif self.offer_method == 'increment':
                        
                        self.markup.append[self.price_offer_set[choices(max_props)[0]]]
                        self.px_offer.append([self.capacity/2, self.markup[period] * self.previous_offer[period][1]])
                                    
                else:
                    
                    if self.offer_method == 'absolute':
                        
                        self.px_offer.append([self.capacity/2, choices(self.price_offer_set)[0]])
                        self.markup.append(self.px_offer[-1])
                        
                    elif self.offer_method == 'increment':
                        
                        self.markup.append(choices(self.price_offer_set)[0])
                        self.px_offer.append([self.capacity/2, self.markup[period] * self.previous_offer[period][1]])
    


    def construct_bm_ladder(self):
        """ Observes the agent's remaining capacities from the PX, and uses
        
        the bm market propensities to construct a linear function of capcacity
        and price offerings for each period of the day. Creates offers/bids in
        the form [id, volume, price], so that when they are sorted in the 
        market clearing function each bid/offer can be tied to its generator.
        """
        
        bm_offers = []
        bm_bids = []
        self.bm_increment_choice = {}

        for period in range(48):
            
            if self.action_method == 'softmax':
            
                bm_ladder_weights = self.softmax(self.bm_propensities[period], self.temperature)
                
                self.bm_increment_choice[period] = choices(self.bm_increment_set, bm_ladder_weights)[0]
                self.bm_intercept_choices[period] += self.bm_increment_choice[period][0]
                self.bm_gradient_choices[period] += self.bm_increment_choice[period][1]
                
            if self.bm_intercept_choices[period] < 0:
                self.bm_intercept_choices[period] = 0
            if self.bm_intercept_choices[period] > (len(self.bm_intercept_set) -1):
                self.bm_intercept_choices[period] = len(self.bm_intercept_set) -1
            if self.bm_gradient_choices[period] < 0:
                self.bm_gradient_choices[period] = 0
            if self.bm_gradient_choices[period] > (len(self.bm_gradient_set) -1):
                self.bm_gradient_choices[period] = len(self.bm_gradient_set) -1
            
            intercept = self.bm_intercept_set[int(self.bm_intercept_choices[period])]
            gradient = self.bm_gradient_set[int(self.bm_gradient_choices[period])]
            
            bm_offer_volume = self.bm_available_volume[period][0]
            bm_bid_volume = self.bm_available_volume[period][1]
        
            bm_bids.append([[self.id, round(-0.05*bm_bid_volume, 2), round(intercept - gradient, 2)],
                            [self.id, round(-0.05*bm_bid_volume, 2), round(intercept - 2*gradient, 2)],
                            [self.id, round(-0.15*bm_bid_volume, 2), round(intercept - 3*gradient, 2)],
                            [self.id, round(-0.25*bm_bid_volume, 2), round(intercept - 4*gradient, 2)],
                            [self.id, round(-0.5*bm_bid_volume, 2), round(intercept - 5*gradient, 2)]])
            
            bm_offers.append([[self.id, round(0.05*bm_offer_volume, 2), round(intercept + gradient, 2)],
                              [self.id, round(0.05*bm_offer_volume, 2), round(intercept + 2*gradient, 2)],
                              [self.id, round(0.15*bm_offer_volume, 2), round(intercept + 3*gradient, 2)],
                              [self.id, round(0.25*bm_offer_volume, 2), round(intercept + 4*gradient, 2)],
                              [self.id, round(0.5*bm_offer_volume, 2), round(intercept + 5*gradient, 2)]])
        
        return bm_bids, bm_offers
                        
        
    def update_px_propensities(self, generation, px_marginal_cost):
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
        self.px_price_offer = [x[1] for x in self.px_offer]
        
        self.px_period_profit = np.multiply(np.subtract([x[1] for x in self.generation], 
                                                     self.marginal_cost), [x[0] for x in self.generation])
        self.px_period_profit -= np.array(startup_penalties)
        
        self.px_day_profit = sum(self.px_period_profit)
        
        # add offer statistics and average profit to expected_profit dictionary
        # offer: [offer frequency, offer successes, success ratio, 
        # average profit, expected profit]
        
        for period, offer_dispatch in enumerate(self.generation):
            
            self.expected_profits[self.px_price_offer[period]][0] += 1
            if offer_dispatch[0] != 0:
                self.expected_profits[self.px_price_offer[period]][1] += 1
            self.expected_profits[self.px_price_offer[period]][2] = self.expected_profits[self.px_price_offer[period]][1] / self.expected_profits[self.px_price_offer[period]][0]
            self.expected_profits[self.px_price_offer[period]][3] = (self.expected_profits[self.px_price_offer[period]][0] * self.expected_profits[self.px_price_offer[period]][3] \
                                                               + self.px_period_profit[period])/(self.expected_profits[self.px_price_offer[period]][0] + 1)
        
            self.expected_profits[self.px_price_offer[period]][4] = self.expected_profits[self.px_price_offer[period]][3] * self.expected_profits[self.px_price_offer[period]][2]
        
        # construct list of volumes carried forward to the balancing market,
        # and pass to the bm_offer_ladder function to create the offer ladder.
        # bm_offer_volumes is of the form [offer volume, bid volume]. If both
        # 0, then generator was not dispatched in the px and hence does not
        # participate in the balancing mechanism
        
        self.bm_available_volume = {}
        
        for period in range(48):
            
            if (self.generation[period][0] == 0) or (self.fuel == 'nuclear'):
                
                self.bm_available_volume[period] = [0, 0]
            
            else:
                
                self.bm_available_volume[period] = [self.capacity/2 - self.generation[period][0],
                                                    self.generation[period][0] - self.min_gen/2]
            
        self.bm_bids, self.bm_offers = self.construct_bm_ladder()
                    
        
        # convert period profit array into a reward array, that penalises generators
        # if they are not dispatched equal to the difference between their offer
        # and the system marginal cost for that period. also penalises generators 
        # if they exceed their cycling limits. This is somewhat arbitrary right 
        # now, will need to fiddle with value
        
        self.px_reward = self.px_period_profit
                
        for period in range(48):
            if self.generation[period][0] == 0:
                self.px_reward[period] += (px_marginal_cost[period] - self.px_price_offer[period]) * self.capacity/2
                if (len(startup_penalties) - startup_penalties.count(0)) > self.cycles:
                    self.px_reward[period] -= (len(startup_penalties) - startup_penalties.count(0)) * self.capacity * 2
        
        
        # updates propensities according to Roth-Erev
        
        if self.reward_method == 'period_profit':
        
            for period, reward in enumerate(self.px_reward):
                
                for j, prop in enumerate(self.px_price_propensities[period]):
                    
                    if j == self.price_offer_set.index(self.px_price_offer[period]):
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * reward/(self.dampener)
                        
                    else:
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.price_offer_set) - 1))
        
        elif self.reward_method == 'day_profit':
            
            for period in range(48):
                
                for j, prop in enumerate(self.px_price_propensities[period]):
                    
                    if j == self.price_offer_set.index(self.px_price_offer[period]):
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * sum(reward)/(self.dampener * 48)
                        
                    else:
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.price_offer_set) - 1))
                        
        elif self.reward_method == 'discounted_profit':
            
            for period in range(48):
                
                discounted_reward = sum(reward * self.discount ** n for n, reward in enumerate(self.px_reward[period:]))
                
                for j, prop in enumerate(self.px_price_propensities[period]):
                    
                    if j == self.price_offer_set.index(self.px_price_offer[period]):
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * discounted_reward/(self.dampener * (48 - period))
                        
                    else:
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.price_offer_set) - 1))
        
        elif self.reward_method == 'kernel_profit':
            
            kernel = self.create_kernel(self.kernel_radius, self.discount)
            kernel_reward = np.convolve(self.px_reward, kernel, mode = 'same')
            
            for period in range(48):
                
                for j, prop in enumerate(self.px_price_propensities[period]):
                    
                    if j == self.price_offer_set.index(self.px_price_offer[period]):
                                                    
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * kernel_reward[period]/(self.dampener * sum(kernel))
                        
                    else:
                        
                        self.px_price_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.price_offer_set) - 1))
                        
            
        self.max_propensities = [np.argmax(x) for x in self.px_price_propensities]
        
        self.previous_offer = self.px_offer
        
#        self.day += 1
        
        
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
        
        self.volume_offer = [x[0] for x in self.px_offer]
        
        self.accepted_bm_actions = [x[3] for x in self.generation]
        
        self.bm_period_profit = [sum([abs(x[1]) * x[2] for x in actions]) - (self.marginal_cost * self.generation[period][2]) for period, actions in enumerate(self.accepted_bm_actions)]

        self.bm_reward = self.bm_period_profit
        
        self.bm_day_profit = sum(self.bm_period_profit)
        
        
        if self.reward_method == 'kernel_profit':
            
            kernel = self.create_kernel(self.kernel_radius, self.discount)
            kernel_reward = np.convolve(self.bm_reward, kernel, mode = 'same')
            
            for period in range(48):
                
                for j, prop in enumerate(self.px_volume_propensities[period]):
                    
                    if j == self.volume_offer_set.index((self.volume_offer[period] * 100 /(self.capacity/2)).__round__()):
                                                    
                        self.px_volume_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * kernel_reward[period]/(self.dampener/3)
                        
                    else:
                        
                        self.px_volume_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / 20)
                
                
                for k, prop in enumerate(self.bm_propensities[period]):
                    
                    if k == self.bm_increment_set.index(self.bm_increment_choice[period]):
                                                    
                        self.bm_propensities[period][k] = (1 - self.recency) * prop + ( 1 - self.expmt) * kernel_reward[period]/(self.dampener/3)
                        
                    else:
                        
                        self.bm_propensities[period][k] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.bm_increment_set) - 1))
                        
        
        elif self.reward_method == 'period_profit':
            
            for period in range(48):
                
                for j, prop in enumerate(self.px_volume_propensities[period]):
                    
                    if j == self.volume_offer_set.index((self.volume_offer[period] * 100 /(self.capacity/2)).__round__()):
                                                    
                        self.px_volume_propensities[period][j] = (1 - self.recency) * prop + ( 1 - self.expmt) * self.bm_reward[period]/(self.dampener/3)
                        
                    else:
                        
                        self.px_volume_propensities[period][j] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.bm_increment_set) - 1))
                
                
                for k, prop in enumerate(self.bm_propensities[period]):
                    
                    if k == self.bm_increment_set.index(self.bm_increment_choice[period]):
                                                    
                        self.bm_propensities[period][k] = (1 - self.recency) * prop + ( 1 - self.expmt) * self.bm_reward[period]/(self.dampener/3)
                        
                    else:
                        
                        self.bm_propensities[period][k] = (1 - self.recency) * prop + ((self.expmt * prop) / (len(self.bm_increment_set) - 1))
                       
        
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
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
    
    
    