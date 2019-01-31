# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:18:18 2018
@author: conno
"""

import numpy as np
import pandas as pd
import math as m
from collections import OrderedDict
from random import choices, shuffle
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from copy import deepcopy
from tqdm import tqdm
import mesa_modified as mesa
from mesa_modified.time import StagedActivation
from mesa_modified.datacollection import DataCollector
from mesa_modified.batchrunner import BatchRunner

import agents
from aggregated_data import genAggregator



class MarketModel(mesa.Model):
    
    def __init__(self, params):

        self.running = True
                
        self.schedule = StagedActivation(self, 
                                         stage_list = params['stage_list'],
                                         shuffle = False,
                                         shuffle_between_stages = False)
        
        # internalise initial parameter set
        
        self.params = params
        
        # get typical demand profile and scale to total capacity of generators times a 
        # specified peak allowable margin. genAggregator return MWh, so need the * 2
        # assuming a 30% capacity factor for wind. Wind is added to self.params
        # rather than params, and so won't be printed onto parameter csv
        
        self.system_params = genAggregator(params['date'][0], params['date'][1], False)
        self.system_params['Supply']['Total Wind'] = self.system_params['Supply']['Wind'] + \
                                                     self.system_params['Supply']['Embedded Wind']
        
        self.true_wind = self.system_params['Intermittent']['Wind'] * 2
        self.true_wind *=  (params['wind']['cap'] * params['num_wind']) / max(self.true_wind)
        
        self.params['true_wind'] = self.true_wind
        
        ### initialise generator fleet and demand profile

        generators = [agents.Generator(self, fuel_id, self.params) for fuel_id in 
              zip([item for sublist in [['nuclear']*self.params['num_nuclear'],
                                        ['wind']*self.params['num_wind'],
                                        ['coal']*self.params['num_coal'],
                                        ['ccgt']*self.params['num_ccgt']]
                                            for item in sublist], 
                                            range(self.params['num_agents']))] 
    
        for agent in generators:
            self.schedule.add(agent)
        
        self.max_capacity = params['max_capacity']
            
        # TODO: standardise whether we're dealing in MW or MWh, these 2s 
        # everywhere are getting confusing
        
        if params['synthetic_demand'] == 0:
        
            self.demand = self.system_params['Demand']['Transmission Demand'] * 2  
            self.demand *= self.max_capacity / (params['peak_margin'] * max(self.demand))
            self.params['demand_profile'] = self.demand
            
        else:
            
            self.demand = np.array(params['synthetic_demand'])
            self.params['demand_profile'] = self.demand

        # initialise supply agents; [market share, forecast error, strategy]
        # TODO: initialise suppliers using params['num_suppliers']
        
        if self.params['use_supplier_agents'] == True:
        
            supplier_params = [[0.3, 0.03, 1], [0.2, 0.02, 0], [0.2, 0.02, 2],
                               [0.15, 0.04, 1], [0.1, 0.02, 0], [0.05, 0.01, 2]]
            
            self.suppliers = [agents.Supplier(supplier_params, self.demand, self.params) for supplier_params in supplier_params]
        
        # internalise a labeled list of generators' id, fuel type, and capacity,
        # and assign each a unique colour from the xkcd colourset (949 + 10 standard)
        
        self.gen_labels = {}
        
        colours = ['C0', 'C1', 'C2', 'C7', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9']
        colours += list(mcolors.XKCD_COLORS.values())
        
        # set labels, and pass some derived parameters to each agent
        
        for agent in self.schedule.agents:
            
            self.gen_labels[agent.id] = [agent.fuel, agent.capacity, colours[agent.id]]
            setattr(agent, 'demand', self.demand)   
            setattr(agent, 'max_capacity', self.max_capacity)
        
        # initialises constraints
        
        self.constraints = params['constraints']                    
            
        
        # initialise Mesa's datacollector functionality

        agent_reporters = {'px_day_profit': 'px_day_profit',
                           'bm_day_profit': 'bm_day_profit',
                           'px_offers': 'px_offer',
                           'bm_offers': 'bm_offers',
                           'bm_bids': 'bm_bids'}
        model_reporters = {'px_dispatch': 'px_dispatch',
                           'generation': 'generation',
                           'imbalance': 'imbalance',
                           'px_marginal_price': 'px_marginal_price',
                           'bm_marginal_price': 'bm_marginal_price',
                           'system_costs': 'system_costs',
                           'emissions': 'emissions',
                           'gen_labels': 'gen_labels'}
        
        if params['verbose'] == True:
        
       
            agent_reporters.update({'px_period_profit': 'px_period_profit',
                                    'px_volume_props': 'px_volume_propensities',
                                    'px_price_props': 'px_price_propensities',
                                    'startup_penalties': 'startup_penalties',
                                    'bm_intercept_props' : 'bm_intercept_propensities',
                                    'bm_gradient_props': 'bm_gradient_propensities',
                                    'bm_period_profit': 'bm_period_profit',
                                    'expected_profits': 'expected_profits',
                                    'bm_intercept_choices': 'bm_intercept_choices',
                                    'bm_gradient_choices': 'bm_gradient_choices'})
            model_reporters.update({'offer_curtailments': 'constrained_schedule',})
            
        self.datacollector = DataCollector(agent_reporters = agent_reporters,
                                           model_reporters = model_reporters)


    def constrainedScheduler(self, unconstrained_schedule, period):
        """ Takes an unconstrained schedule for a given period, checks it against
        
        a rolling count of up/down time for each generator, and returns a 
        constrained schedule.
        """
        
        constrained_schedule = OrderedDict()
        
        for agent_id, dispatch in unconstrained_schedule.items():

            agent = self.schedule.agents[agent_id]

            if agent.constraint_mode() == 'none':
                if dispatch == 0:
                    agent.run_time -= 1
                    constrained_schedule[agent_id] = 0
                else:
                    agent.run_time += 1
                    constrained_schedule[agent_id] = 1
            
            if agent.constraint_mode() == 'on':
                if dispatch == 0:
                    agent.run_time += 1
                    constrained_schedule[agent_id] = 1
                    self.px_period_offers[period][agent_id] = agent.marginal_cost
                else:
                    agent.run_time += 1
                    constrained_schedule[agent_id] = 1
            
            if agent.constraint_mode() == 'off':
                if dispatch == 0:
                    agent.run_time -= 1
                    constrained_schedule[agent_id] = 0
                else:
                    agent.run_time -= 1
                    constrained_schedule[agent_id] = 0
            
            if constrained_schedule[agent_id] == 1 and agent.last_dispatch == 0:
                agent.run_time = 1
            if constrained_schedule[agent_id] == 0 and agent.last_dispatch == 1:
                agent.run_time = -1
            
                    
        # realigns constrained schedule and checks to see if demand is still
        # satisfied, chopping off those offers that are not constrained that 
        # go over
        
        constrained_schedule = OrderedDict(sorted(constrained_schedule.items(), 
                                                  key = lambda x: (-x[1], self.px_period_offers[period][x[0]])))
        
        constrained_cumulative_capacity = np.cumsum(
                    [self.schedule.agents[gen_id].capacity for gen_id in constrained_schedule.keys()])
        
        satisfying_demand = list(np.where(constrained_cumulative_capacity <= self.demand[period], 
                                          1, 0))
          
        if 0 in satisfying_demand:
            self.px_marginal_price[period] = self.px_period_offers[period][satisfying_demand.index(0)]
            satisfying_demand[satisfying_demand.index(0)] = 1
            
        for i, (agent_id, dispatch) in enumerate(constrained_schedule.items()):
            constrained_schedule[agent_id] = satisfying_demand[i]
        
        for agent in self.schedule.agents:
            agent.last_dispatch = constrained_schedule[agent_id]
        
        
        
        return constrained_schedule
    
    
    
    def clearMarket(self, offers, demand, period):
        """ Accepts a list of generator offer tuples of the form 
        
        (quantity, price), the demand profile for the day, and the period in
        which the market is to be cleared. Returns an array of values between
        for each generator, indicating the proportion of their offer
        that is dispatched (can be greater than 1 if constrained up!)
        """
        
        # TODO: should accepted_offers be returned as a dict to save the next
        # step in the main market body? 
        # TODO: make inclusion of wind error more generalised to accommodate
        # more than one wind farm with id 1
        
        # the addition of the wind offer and the actual wind profile allows 
        # the AR(2) error from the wind forecasting to be included in the demand
        # forecast, alongside the demand forecasting error. Currently assumes
        # a single wind farm with an id of 1.
        
        if self.params['use_supplier_agents'] == True:
        
            bids = []
            
            for supplier in self.suppliers:
            
                bids.append(supplier.make_bid(demand, period))
            
            bids = sorted(bids, key = lambda tup: tup[1], reverse = True)
    
            demand_forecast = sum([bid[0] for bid in bids])
        
        else:
            
            demand_forecast = demand[period]/2

        # bids are tuples of forecast demand in MWh and bid price in £/MWh. For
        # now, the bid price is ignored and the generators are paid what they
        # offered.
        
        volume_offers = [offer[1][0] for offer in offers.items()]
            
        cumulative_offers = np.cumsum(volume_offers)
        
        accepted_offers = list(np.where(cumulative_offers <= demand_forecast, 1, 0))
        
        # if bounded rationality is in play and nuclear is not dispatched, its
        # offer is re-assigned to £1/MWh and the market is recleared.
        # TODO: this currently assumes one nuclear plant with id of 0, will 
        # need to make this more general when there is more than one of them
        
        if self.constraints == 'all':
            if accepted_offers[list(offers.keys()).index(0)] == 0:
                offers[0][1] = 1
                offers = OrderedDict(sorted(offers.items(), 
                                            key = lambda x: (x[1][1], self.schedule.agents[x[0]].marginal_cost)))
                self.px_period_offers[period] = offers
                return self.clearMarket(offers, demand, period)
            
        # SIMPLE MARGINAL DISPATCH, IGNORES MINIMUM GENERATION LIMITS
        # the marginal generator thus dispatches partially to meet forecast
        # demand
#            
#        if 0 in accepted_offers:
#                
#            self.px_marginal_price[period] = list(offers.items())[accepted_offers.index(0)][1][1]
#            accepted_offers[accepted_offers.index(0)] = (demand_forecast - cumulative_offers[accepted_offers.index(0) - 1]) \
#                                                        / list(offers.items())[accepted_offers.index(0)][1][0]
#                                                        
#        else:
#            
#            self.px_marginal_price[period] = list(self.px_period_offers[period].items())[-1][1][1]
#        
        
        if self.params['px_clearing_method'] == 'simple':
            
            # simple clearing method, that only dispatches the marginal generator
            # if it decreases the difference between total PX dispatch and foreacst
            # demand, sending the rest to the BM. 
            
            if 0 in accepted_offers:
                
                self.px_marginal_price.append(list(offers.items())[accepted_offers.index(0)][1][1])
                
                marginal_volume = demand_forecast - cumulative_offers[accepted_offers.index(0) - 1]
                marginal_gen = self.schedule.agents[list(offers.items())[accepted_offers.index(0)][0]]
                
                if 2 * marginal_volume >= marginal_gen.px_offer[period][3]:
                    
                    accepted_offers[accepted_offers.index(0)] = 1
                    marginal_volume -= marginal_gen.px_offer[period][3]
            
            else:
            
                self.px_marginal_price.append(list(offers.items())[-1][1][1])
                marginal_volume = demand_forecast - cumulative_offers[-1]
            
            # If marginal_gen is +ve, then system is short, therefore deduct from
            # demand imbalance

            demand_imbalance = demand_forecast - demand[period]/2
            intermittent_imbalance = offers[1][0] - self.true_wind[period]/2
            px_clearing_imbalance = marginal_volume
            
            self.imbalance.append(demand_imbalance - px_clearing_imbalance - intermittent_imbalance)
            
            return accepted_offers  
        
                
        elif self.params['px_clearing_method'] == 'advanced':
            
            # ADVANCED MARGINAL GENERATION WITH CASES AND CONSTRAINTS
            
            if 0 in accepted_offers:
                
                self.px_marginal_price.append(list(offers.items())[accepted_offers.index(0)][1][1])
                
                marginal_volume = demand_forecast - cumulative_offers[accepted_offers.index(0) - 1]
                marginal_gen = self.schedule.agents[list(offers.items())[accepted_offers.index(0)][0]]
    
                # case 1: first tries to increase volume of generators already
                # dispatched, excluding wind, starting from the cheapest
                
                if sum([offer[1][2] - offer[1][0] for offer in list(offers.items())[1:(accepted_offers.index(0))]]) >= marginal_volume:
                    
                    for i, offer in enumerate(list(offers.items())[1:(accepted_offers.index(0))]):
                        
                        if self.schedule.agents[offer[0]] == 'nuclear':
                            
                            pass
                        
                        elif offer[1][2] - offer[1][0] >= marginal_volume:
                            
                            accepted_offers[i + 1] = (marginal_volume + offer[1][0])/offer[1][0]
                            marginal_volume = 0
                            break
                        
                        else:
                            
                            accepted_offers[i + 1] = offer[1][2]/offer[1][0]
                            marginal_volume -= (offer[1][2] - offer[1][0])
                            
                
                # case 2: dispatches the marginal generator somewhere between its
                # minimum and maximum generation
                
                elif (sum([offer[1][2] - offer[1][0] for offer in list(offers.items())[1:(accepted_offers.index(0))]]) < marginal_volume) \
                    and (marginal_gen.px_offer[period][3] <= marginal_volume <= marginal_gen.px_offer[period][2]):
                        
                    accepted_offers[accepted_offers.index(0)] = marginal_volume/(marginal_gen.px_offer[period][0])
                    
                
                # case 3: marginal generator needs to be dispatched, but other generators
                # must also be constrained down to fit the marginal generator's
                # minimum generation limit
                
                elif (sum([offer[1][0] - offer[1][3] for offer in list(offers.items())[1:(accepted_offers.index(0))]]) >= (marginal_gen.px_offer[period][3] - marginal_volume)) \
                    and (marginal_gen.px_offer[period][3] >= marginal_volume):
                    
                    accepted_offers[accepted_offers.index(0)] = marginal_gen.px_offer[period][3]/(marginal_gen.px_offer[period][0])
                    
                    # strictly the marginal volume is now negative, but we're keeping
                    # it positive here for ease of comparison
                    marginal_volume = marginal_gen.px_offer[period][3] - marginal_volume
                    
                    for i, offer in enumerate(list(offers.items())[1:(accepted_offers.index(0))]):
                        
                        if self.schedule.agents[offer[0]] == 'nuclear':
                            
                            pass
                        
                        elif offer[1][0] - offer[1][3] >= marginal_volume:
                            
                            accepted_offers[i + 1] = (offer[1][0] - marginal_volume)/offer[1][0]
                            marginal_volume = 0
                            break
                        
                        else:
                            
                            accepted_offers[i + 1] = offer[1][3]/offer[1][0]
                            marginal_volume -= (offer[1][0] - offer[1][3])
                
                    # case 4: if there is still marginal volume remaining, constrain
                    # wind generators to make up the difference
                    
                    if marginal_volume != 0:
                        
                        accepted_offers[0] = (list(offers.items())[0][1][0] - marginal_volume)/list(offers.items())[0][1][0]
                        marginal_volume = 0
            
            else:
                
                # covers the somewhat common case where all generators are 
                # dispatched, but the cumulative volume is lower than forecast
                # demand, meaning that the only option is to turn some up
                
                self.px_marginal_price.append(list(offers.items())[-1][1][1])
                
                marginal_volume = demand_forecast - cumulative_offers[-1]
                                    
                for i, offer in enumerate(list(offers.items())[1:]):
                    
                    if self.schedule.agents[offer[0]] == 'nuclear':
                        
                        pass
                    
                    elif offer[1][2] - offer[1][0] >= marginal_volume:
                        
                        accepted_offers[i + 1] = (marginal_volume + offer[1][0])/offer[1][0]
                        marginal_volume = 0
                        break
                    
                    else:
                        
                        accepted_offers[i + 1] = offer[1][2]/offer[1][0]
                        marginal_volume -= (offer[1][2] - offer[1][0])
                        
            # TODO: store imbalance as two separate components, due to uncertainty
            # demand forecasts and uncertainty from intermittent generators.
            # store as (total_imbal, intermittent_imbal, demand_imbal).
            # Positive imbalance means system is long, negative means short. 
            
            # if demand_imbalance is +ve, system is long; if intermittent_imbalance
            # is +ve, system is short. Therefore total imbalance equals demand -
            # intermittent imbalance
            
            if len(self.params['synthetic_imbalance']) == 0:
                
                demand_imbalance = demand_forecast - demand[period]/2
                intermittent_imbalance = offers[1][0] - self.true_wind[period]/2
                
                self.imbalance.append(demand_imbalance - intermittent_imbalance)
            
            else:
                
                self.imbalance.append(self.params['synthetic_imbalance'][period])
            
            return accepted_offers
    
    
    def clearBalancingMarket(self, imbalance, bm_offers, bm_bids, period):
        """ Accepts the outstanding physical imbalance for the period and the
        
        the generator bids and offers. Clears the market by matching the
        cheapest offers if the system is short or the most expensive bids if
        the system is long, returning an array of accepted actions and stores the 
        marginal cost of the most expensive action taken. Marginal cost given
        as [system_position (0 for short, 1 for long), price]
        """
        
        if (imbalance == 0) or (bm_offers == []) or (bm_bids == []):
            
            return []

        # if system is long, need to use bids to reduce the px generation
        if imbalance > 0:
            
            volume_bids = [bid[1] for bid in bm_bids]
            
            cumulative_bids = np.cumsum(volume_bids)
        
            bm_actions = list(np.where(cumulative_bids >= -imbalance, 
                                       1, 0))
            
            if bm_actions[0] == 0:
                
                self.bm_marginal_price.append([1, bm_bids[bm_actions.index(0)][2]])
                bm_actions[0] = imbalance / -bm_bids[0][1]
                
            elif 0 in bm_actions:
                
                self.bm_marginal_price.append([1, bm_bids[bm_actions.index(0)][2]])
                bm_actions[bm_actions.index(0)] = (imbalance + cumulative_bids[bm_actions.index(0) - 1]) \
                                                            / (-bm_bids[bm_actions.index(0)][1])
                                                        
            else:
            
                self.bm_marginal_price.append([1, bm_bids[-1][2]])
            
            accepted_bm_actions = []
            
            for i, action in enumerate(bm_actions):
                if action != 0:
                    accepted_bm_actions.append(bm_bids[i])
        
        
        # likewise, if system is short, need to use offers to increase px generation
        elif imbalance < 0:
            
            volume_offers = [offer[1] for offer in bm_offers]

            cumulative_offers = np.cumsum(volume_offers)
        
            bm_actions = list(np.where(cumulative_offers <= -imbalance, 
                                       1, 0))

            if bm_actions[0] == 0:
                
                self.bm_marginal_price.append([0, bm_offers[bm_actions.index(0)][2]])
                bm_actions[0] = -imbalance / bm_offers[0][1]
                
            elif 0 in bm_actions:
                
                self.bm_marginal_price.append([0, bm_offers[bm_actions.index(0)][2]])
                bm_actions[bm_actions.index(0)] = (-imbalance - cumulative_offers[bm_actions.index(0) - 1]) \
                                                            / (bm_offers[bm_actions.index(0)][1])
                                                        
            else:
            
                self.bm_marginal_price.append([0, bm_offers[-1][2]])
                
            accepted_bm_actions = []
            
            for i, action in enumerate(bm_actions):
                if action != 0:
                    accepted_bm_actions.append(bm_offers[i])        
        
        # modify the marginal accepted action if it's a proportion of the
        # original bid/offer
        
        if 0 in bm_actions:
            accepted_bm_actions[-1][1] *= bm_actions[bm_actions.index(0) - 1]
        else:
            accepted_bm_actions[-1][1] *= bm_actions[-1]
        
        return accepted_bm_actions
    
    
    
    def step(self):
        """ Each Model step induces N staged agent steps, where N = the length
        
        of the staged_activation parameter list.
        """
        
        ### DAY AHEAD MARKET CLEARING ###
        
        day_offers = {}

        # collect each agent's offers for the day ahead. deepcopy is used here
        # because when nuclear is constrained to offer = 1 if not successful 
        # in the market, Roth-Erev needs to know what the offer was that failed
        # in order to update the propensities appropriately
        
        self.schedule.step(self.params['stage_list'][0])
        
        for agent in self.schedule.agents:
            day_offers[agent.id] = deepcopy(agent.px_offer)
            
        
        # the master ordereddict that contains every generator, what they bid,
        # and how much they generated for each period
        
        self.generation = OrderedDict()

        # clear the market according to cheapest offers per period
        
        self.px_period_offers = OrderedDict()
        self.px_marginal_price = []
        self.bm_marginal_price = []
        self.system_cost = []
        self.imbalance = []
        
        self.constrained_schedule = OrderedDict()
        
        for period in range(48):
        
            self.px_period_offers[period] = OrderedDict() 
    
            for gen_id in day_offers.keys():
                
                self.px_period_offers[period][gen_id] = day_offers[gen_id][period]
                
            # sorts by offer; if there is a tie, sorts by marginal cost    
            self.px_period_offers[period] = OrderedDict(
                    sorted(self.px_period_offers[period].items(), 
                           key = lambda x: (x[1][1], self.schedule.agents[x[0]].marginal_cost))) 
                    

            # passes generator offers to market clearning mechanism, where they
            # are matched against static supply agent bids
            
            accepted_offers = self.clearMarket(self.px_period_offers[period], 
                                                self.demand,
                                                period)
                
            # convert list of accepted offers to ordered dict of agent:0/1, to 
            # pass to constraint function
            
            unconstrained_schedule = OrderedDict()
            
            for i, agent in enumerate(list(self.px_period_offers[period].keys())):
                unconstrained_schedule[agent] = accepted_offers[i]
                
            
            # pass dict of unconstrained offers to function that checks against
            # each agent's individual constraints, and returns a constrained 
            # schedule, if enabled
            
#            if self.constraints == 'all':
#                
#                constrained_schedule[period] = self.constrainedScheduler(unconstrained_schedule, period)
#                
#            elif self.constraints == 'none':
                
            self.constrained_schedule[period] = unconstrained_schedule 
            
            # save each agent's offer, what they dispatched in MWh, and what they
            # recieved (i.e. constrained payments), and store their last dispatch
            # state to update the constraint function
            # [fuel, px_dispatch, px_offer, bm_dispatch, [list of successful bm 
            # offers], total_dispatch, emissions]
            
            self.generation[period] = OrderedDict()
            for agent in self.schedule.agents:
#                agent.last_dispatch = self.constrained_schedule[period][agent.id]
                if agent.fuel == 'wind':
                    
                    self.generation[period].update(
                            {agent.id: [agent.fuel,
                                        agent.px_offer[period][0],
                                        0, 0, [],
                                        self.true_wind[period]/2,
                                        0]})
                else:
                        
                    self.generation[period].update(
                            {agent.id: [agent.fuel,
                                        agent.px_offer[period][0] * self.constrained_schedule[period][agent.id],
                                        self.px_period_offers[period][agent.id][1],
                                        0, [],
                                        agent.px_offer[period][0] * self.constrained_schedule[period][agent.id],
                                        0]})
            
        
        # create per-period dispatch schedules, consisting of tuples of the gen
        # id, and its dispatch for that period in MWh, in order of price offer,
        # as amended by any rational constraints. Note that this is the pre-
        # balancing mechanism dispatch, i.e. the FPN!
        
        self.px_dispatch = OrderedDict()
    
        for period in range(48):
        
            self.px_dispatch[period] = []
            
            for gen, accepted in self.constrained_schedule[period].items():
                if accepted != 0:
                    self.px_dispatch[period].append((gen, self.generation[period][gen][1]))
                else:
                    break
            
            
        # update generator propensities by passing it back its dispatch
        
        self.schedule.step(self.params['stage_list'][1], 
                           self.generation,
                           self.px_marginal_price)
        
        
        ### BALANCING MECHANISM CLEARING ###
        
        if self.params['balancing_mechanism'] == True:
            
            bm_day_offers = {}
            bm_day_bids = {}
            
            self.bm_period_offers = OrderedDict()
            self.bm_period_bids = OrderedDict()
            
            self.bm_schedule = OrderedDict()
            
            for agent in self.schedule.agents:
                
                if (agent.fuel != 'wind') and (agent.fuel != 'nuclear'):
                    
                    bm_day_offers[agent.id] = agent.bm_offers
                    bm_day_bids[agent.id] = agent.bm_bids
            
            for period in range(48):
                
                self.bm_period_offers[period] = []
                self.bm_period_bids[period] = []
            
                for gen_id in bm_day_offers.keys():
                    for action in bm_day_offers[gen_id][period]:
                        if action[1] != 0:
                            self.bm_period_offers[period].append(action)
                    for action in bm_day_bids[gen_id][period]:
                        if action[1] != 0:
                            self.bm_period_bids[period].append(action)
                
                # first shuffles the list so that individual generators are not
                # favoured for having proximate prices between their offers
                
                shuffle(self.bm_period_offers[period])
                shuffle(self.bm_period_bids[period])
                
                # sorts by offer price   
                self.bm_period_offers[period] = sorted(self.bm_period_offers[period], 
                                                       key = lambda x: (x[2]))
                
                # bids are sorted in decending order, because the SO prefers those
                # who would pay the most to lower their output first
                self.bm_period_bids[period] = sorted(self.bm_period_bids[period], 
                                                     key = lambda x: (x[2]),
                                                     reverse = True)
            
                
                accepted_bm_actions = self.clearBalancingMarket(self.imbalance[period],
                                                                  self.bm_period_offers[period],
                                                                  self.bm_period_bids[period],
                                                                  period)
                
            
                # accepted_bm_actions of the form [agent.id, volume, price], in 
                # order of price
                            
                for action in accepted_bm_actions:
                    self.generation[period][action[0]][3] += action[1]
                    self.generation[period][action[0]][4].append(action)
                    self.generation[period][action[0]][5] += action[1]
                
                self.bm_schedule[period] = accepted_bm_actions
                
                # also adds emissions per generator to the end of generation
                
                for agent in self.schedule.agents:
                    
                    self.generation[period][agent.id][6] = self.generation[period][agent.id][5] * agent.emissions
            
            # calculates the system cost for each period of the day, equal to 
            # the total cost of fuel burnt, the startup costs, the bm costs
            # to the SO, and the carbon tax if applicable
            # TODO: add carbon tax parameter, using emissions
            
            fuel_costs = np.array([sum([self.generation[period][gen_id][5] * self.schedule.agents[gen_id].marginal_cost 
                                    for gen_id in range(self.params['num_agents'])]) for period in range(48)])
            
            startup_costs = np.array([sum([self.schedule.agents[gen_id].startup_penalties[period]
                                      for gen_id in range(self.params['num_agents'])]) for period in range(48)])
            
            bm_SO_costs = np.array([sum([sum([self.generation[period][gen_id][4][i][1] * self.generation[period][gen_id][4][i][2] 
                                    for i in range(len(self.generation[period][gen_id][4]))]) 
                                    for gen_id in range(self.params['num_agents'])])
                                    for period in range(48)])
            
            self.system_costs = fuel_costs + startup_costs + bm_SO_costs
            
            self.emissions = [sum([self.generation[period][agent_id][6] for agent_id in range(self.params['num_agents'])])
                                                                        for period in range(48)]
            
            # update generator bm_mechanism propensities        
            
            self.schedule.step(self.params['stage_list'][2],
                               self.generation)
        
        self.datacollector.collect(self)
    


def movingAverage(x, N):
    
    return pd.DataFrame(x).rolling(N).mean()[N:]
            



# runs the simulation, using mesa's batchrunner class to take in parameters to
# to vary during each run, the number of runs to make, and the elements to 
# return. Returns a graph of profit for each generator over time, a dispatch 
# graph in order to bid price for the final day, and an averaged generation
# fuel mix over the final 50 days

def runSimulation(model_class, params, days, name = None, verbose = False):
    
    params['days'] = days
    params['verbose'] = verbose

    model = model_class(params)
    
    for i in tqdm(range(days), desc = 'Running Model: "{0}"'.format(name)):
        model.step()
    
    # agent variables from datacollector
    px_day_profit = model.datacollector.get_agent_vars_dataframe()['px_day_profit']
    bm_day_profit = model.datacollector.get_agent_vars_dataframe()['bm_day_profit']
    px_offers = model.datacollector.get_agent_vars_dataframe()['px_offers']
    bm_offers = model.datacollector.get_agent_vars_dataframe()['bm_offers']
    bm_bids = model.datacollector.get_agent_vars_dataframe()['bm_bids']
    
    # model variables from datacollector
    px_dispatch = model.datacollector.get_model_vars_dataframe()['px_dispatch']
    generation = model.datacollector.get_model_vars_dataframe()['generation']
    imbalance = model.datacollector.get_model_vars_dataframe()['imbalance']
    bm_marginal_price = model.datacollector.get_model_vars_dataframe()['bm_marginal_price']
    px_marginal_price = model.datacollector.get_model_vars_dataframe()['px_marginal_price']
    system_costs = model.datacollector.get_model_vars_dataframe()['system_costs']
    emissions= model.datacollector.get_model_vars_dataframe()['emissions']
    gen_labels = model.datacollector.get_model_vars_dataframe()['gen_labels']
    
    if verbose == True:    
        
        px_period_profit = model.datacollector.get_agent_vars_dataframe()['px_period_profit']
        px_volume_props = model.datacollector.get_agent_vars_dataframe()['px_volume_props']
        bm_intercept_props = model.datacollector.get_agent_vars_dataframe()['bm_intercept_props']
        bm_gradient_props = model.datacollector.get_agent_vars_dataframe()['bm_gradient_props']
        expected_profits = model.datacollector.get_agent_vars_dataframe()['expected_profits']
        bm_period_profit = model.datacollector.get_agent_vars_dataframe()['bm_period_profit']
        bm_intercept_choices = model.datacollector.get_agent_vars_dataframe()['bm_intercept_choices']
        bm_gradient_choices = model.datacollector.get_agent_vars_dataframe()['bm_gradient_choices']
        px_price_props = model.datacollector.get_agent_vars_dataframe()['px_price_props']
        startup_penalties = model.datacollector.get_agent_vars_dataframe()['startup_penalties']

        offer_curtailments = model.datacollector.get_model_vars_dataframe()['offer_curtailments']
       
   
    if params['capture_props'] == True:
        
        final_props = {}
        
        for agent in model.schedule.agents:
            
            final_props[agent.id] = {'px_price': agent.px_price_propensities,
                                     'px_volume': agent.px_volume_propensities,
                                     'bm_intercept': agent.bm_intercept_propensities,
                                     'bm_gradient': agent.bm_gradient_propensities}
    
    # The below code generates data averaging the dispatch mix, average
    # bid price, and peak bid price of dispatched generators 
    # per period averaged over the final 100 days of the simulation, along with
    # a 25-day MA of avg peak/off bid price for each fuel type over the whole sim
    
    avg_fuel_mix = pd.DataFrame()
    avg_offer_price = pd.DataFrame()
    peak_offer_price = pd.DataFrame()
    avg_bm_marginal_price = pd.DataFrame()
    
    for period in tqdm(range(48), desc = 'Crunching data...'):
        all_gen = pd.DataFrame({agent_id: np.zeros(50) for agent_id in gen_labels.iloc[-1].keys()})
        all_offers = pd.DataFrame({agent_id: np.zeros(50) for agent_id in gen_labels.iloc[-1].keys()})
        
        for day in range(50):
            for agent_id in gen_labels.iloc[-1].keys():
                all_gen.iloc[day][agent_id] = generation.iloc[-1 -day][period][agent_id][5]
                all_offers.iloc[day][agent_id] = np.where(generation.iloc[-1 -day][period][agent_id][1] != 0, 
                                                          generation.iloc[-1 -day][period][agent_id][2],
                                                          np.nan)
        
        all_gen.columns = all_gen.columns.to_series().apply(lambda gen: gen_labels.iloc[-1][gen][0])
        all_gen = all_gen.groupby(all_gen.columns, axis = 1).sum()
        all_gen = all_gen.mean(axis = 0)
        avg_fuel_mix = avg_fuel_mix.append(all_gen, ignore_index = True)
                
        all_offers.columns = all_offers.columns.to_series().apply(lambda gen: gen_labels.iloc[-1][gen][0])
        peak_offers = all_offers.groupby(all_offers.columns, axis = 1).max()
        peak_offers = peak_offers.max(axis = 0)
        
        avg_offers = all_offers.groupby(all_offers.columns, axis = 1).mean()
        avg_offers = avg_offers.mean(axis = 0, skipna = True)
        
        avg_offer_price = avg_offer_price.append(avg_offers, ignore_index = True)
        peak_offer_price = peak_offer_price.append(peak_offers, ignore_index = True)
        
        if params['balancing_mechanism'] == True: 
            avg_bm_marginal_price = avg_bm_marginal_price.append([[bm_marginal_price.iloc[-1][period][0], np.mean([bm_marginal_price.iloc[-day][period][1] for day in range(50)])]], ignore_index = True)
    
    if params['balancing_mechanism'] == True:
        avg_bm_marginal_price.columns = ['Direction', 'Price']
    
    rolling_offers_peak = pd.DataFrame({agent_id: np.zeros(days) for agent_id in gen_labels.iloc[-1].keys()})
    rolling_offers_off = pd.DataFrame({agent_id: np.zeros(days) for agent_id in gen_labels.iloc[-1].keys()})
    rolling_offers_volume = pd.DataFrame({agent_id: np.zeros(days) for agent_id in gen_labels.iloc[-1].keys()})
    avg_rolling_offers = pd.DataFrame({period: np.zeros(days) for period in range(48)})
    
    for day in tqdm(range(days), desc = 'Still crunching data...'):
            for agent_id in gen_labels.iloc[-1].keys():
                
                off_offers = []
                peak_offers = []
                volume_offers = []
                
                for period in range(48):
                    if (period < 32 or period >= 42):
                        off_offers.append(float(np.where(list(generation.iloc[day].values())[period][agent_id][1] != 0, 
                                                         list(generation.iloc[day].values())[period][agent_id][2],
                                                         np.nan)))
                    elif (32 <= period < 42):
                        peak_offers.append(float(np.where(list(generation.iloc[day].values())[period][agent_id][1] != 0, 
                                                          list(generation.iloc[day].values())[period][agent_id][2],
                                                          np.nan)))
                    
                    volume_offers.append(float(np.where(list(generation.iloc[day].values())[period][agent_id][1] != 0, 
                                                        px_offers[day, agent_id][period][0]/(gen_labels.iloc[-1][agent_id][1]/2),
                                                        np.nan)))
                
                rolling_offers_peak.iloc[day][agent_id] = pd.Series(peak_offers).mean(skipna = True)
                rolling_offers_off.iloc[day][agent_id] = pd.Series(off_offers).mean(skipna = True) 
                rolling_offers_volume.iloc[day][agent_id] = pd.Series(volume_offers).mean(skipna = True)
            
            for period in range(48):
                
                all_offers_temp = []
                
                for agent_id in gen_labels.iloc[-1].keys():
                    
                    all_offers_temp.append(float(np.where(list(generation.iloc[day].values())[period][agent_id][1] != 0, 
                                                          list(generation.iloc[day].values())[period][agent_id][2],
                                                          np.nan)))
                    
                avg_rolling_offers.iloc[day][period] = pd.Series(all_offers_temp).mean(skipna = True)
    
    rolling_offers_off.columns = rolling_offers_off.columns.to_series().apply(lambda gen: gen_labels.iloc[-1][gen][0])
    rolling_offers_off = rolling_offers_off.groupby(rolling_offers_off.columns, axis = 1).mean()
    
    rolling_offers_peak.columns = rolling_offers_peak.columns.to_series().apply(lambda gen: gen_labels.iloc[-1][gen][0])
    rolling_offers_peak = rolling_offers_peak.groupby(rolling_offers_peak.columns, axis = 1).mean()
    
    rolling_offers_volume.columns = rolling_offers_volume.columns.to_series().apply(lambda gen: gen_labels.iloc[-1][gen][0])
    rolling_offers_volume = rolling_offers_volume.groupby(rolling_offers_volume.columns, axis = 1).mean()
            
    rolling_offers_off = movingAverage(rolling_offers_off, 25)
    rolling_offers_peak = movingAverage(rolling_offers_peak, 25)
    rolling_offers_volume = movingAverage(rolling_offers_volume, 5)
    
    # calculates proportional dispatch values
    
    model_demand = model.demand
    
    true_dispatch = model.system_params['Supply'][['Nuclear', 'Total Wind', 'Coal', 'CCGT']]
    true_dispatch.columns = ['nuclear', 'wind', 'coal', 'ccgt']
    
    prop_true_total = true_dispatch.sum(axis = 1)
    prop_true_dispatch = true_dispatch.apply(lambda x: x/prop_true_total)
    
    prop_model_dispatch = avg_fuel_mix[['nuclear', 'wind', 'coal', 'ccgt']]
    prop_model_total = avg_fuel_mix[['nuclear', 'wind', 'coal', 'ccgt']].sum(axis = 1)
    prop_model_dispatch = prop_model_dispatch.apply(lambda x: x/prop_model_total)
    
    # calculates the error between the true and model dispatch for comparison
    # between runs. If model dispatch is lower, error is negative. For absolute
    # dispatch, error is in MW. 
    
    prop_dispatch_error = prop_model_dispatch - prop_true_dispatch
    
    dispatch_error = (avg_fuel_mix - true_dispatch) * 2
    dispatch_error['total'] = dispatch_error.sum(axis = 1)
    
    rms_dispatch_error = m.sqrt(sum(dispatch_error['total']) ** 2)
    
    results = {'name': name,
               'px_day_profit': px_day_profit,
               'px_dispatch': px_dispatch,
               'px_offers': px_offers,
               'bm_offers': bm_offers,
               'bm_bids': bm_bids,
               'demand': model_demand,
               'true_dispatch': true_dispatch,
               'bm_day_profit': bm_day_profit,
               'gen_labels': gen_labels,
               'generation': generation,
               'imbalance': imbalance,
               'bm_marginal_price': bm_marginal_price,
               'px_marginal_price': px_marginal_price,
               'system_costs': system_costs,
               'emissions': emissions,
               'avg_bm_marginal_price': avg_bm_marginal_price,
               'avg_offer_price': avg_offer_price,
               'avg_fuel_mix': avg_fuel_mix,
               'rolling_offers_off': rolling_offers_off,
               'rolling_offers_peak': rolling_offers_peak,
               'rolling_offers_volume': rolling_offers_volume,
               'avg_rolling_offers': avg_rolling_offers,
               'prop_model_dispatch': prop_model_dispatch,
               'prop_true_dispatch': prop_true_dispatch,
               'prop_dispatch_error':prop_dispatch_error,
               'dispatch_error': dispatch_error,
               'rms_dispatch_error': rms_dispatch_error}
    
    if verbose == True:
        
        results.update({'px_period_profit': px_period_profit,
                        'px_price_props': px_price_props,
                        'px_volume_props': px_volume_props,
                        'bm_intercept_props': bm_intercept_props,
                        'bm_gradient_props': bm_gradient_props,
                        'bm_intercept_choices': bm_intercept_choices,
                        'bm_gradient_choices': bm_gradient_choices,
                        'offer_curtailments': offer_curtailments,
                        'expected_profits': expected_profits,
                        'bm_period_profit': bm_period_profit,
                        'startup_penalties': startup_penalties})
        
    if params['capture_props'] == True:
        
        results.update({'final_props': final_props})
        
        
    return results
  
    
        







def batchrunSimulation(model_class, fixed_params, days, 
                        variable_params = None, 
                        iterations = 1):
    
    fixed_params['days'] = days
    params = {**fixed_params, **variable_params}

    # adjusts for the batchrunner treating each stage method as a simulation 
    # cycle, when we only want complete days
    
    cycles = days * len(params['stage_list'])

    batch_run = BatchRunner(model_cls = model_class,
                            fixed_parameters = fixed_params,
                            variable_parameters = variable_params,
                            iterations = iterations,
                            max_steps = cycles,
                            model_reporters = {'Datacollector': lambda m: m.datacollector})
    
    batch_run.run_all()
    
    results = batch_run.get_model_vars_dataframe()
    
    
    # for the batchrunner each step only equals half a day because of the edit
    # made to the StagedActivation class
    
    for i, value in enumerate(results[results.columns[0]]):
        
        profit = results['Datacollector'].iloc[i].get_agent_vars_dataframe()['px_day_profit']
        px_dispatch = results['Datacollector'].iloc[i].get_model_vars_dataframe()['px_dispatch']
        gen_labels = results['Datacollector'].iloc[i].get_model_vars_dataframe()['gen_labels'][days - 1]
        
        figure = plt.figure(i)
        for j in range(params['num_agents']):
            plt.plot(range(days - 5), movingAverage(profit[:, j], 5))
            plt.title('Profit 5-Day MA with {0}: {1}'.format(results.columns[0], value))
            plt.legend([item for sublist in [['ccgt']*params['num_ccgt'], ['coal']*params['num_coal'], ['nuclear']*params['num_nuclear']] 
              for item in sublist])
        plt.show()
        
        x = i + len(results[results.columns[0]])
    
        # TODO: integrate 'generation' dataframe to properly sum up lower dispatches
        figure = plt.figure(x)    
        for period in range(48):
            for k, (gen, dispatch) in enumerate(px_dispatch[days - 1][period]):
                plt.bar(range(period, period + 1),
                        dispatch,
                        label = gen_labels[gen][0],
                        color = gen_labels[gen][2],
                        bottom = sum([gen_labels[h][1] for h in px_dispatch[days - 1][period][:k]]))
        plt.title('Final Dispatch Schedule with {0}: {1}'.format(results.columns[0], value))
        
    return results



