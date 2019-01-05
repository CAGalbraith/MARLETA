# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:18:18 2018

@author: conno
"""

import numpy as np
import pandas as pd
import math as m
from collections import OrderedDict
from random import choices
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from copy import deepcopy

from tqdm import tqdm
import json
import csv
from os.path import getsize
from os import makedirs

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
        
        self.wind_profile = self.system_params['Intermittent']['Wind'] * 2
        self.wind_profile *=  (params['wind']['cap'] * params['num_wind']) / max(self.wind_profile)
        
        self.params['wind_profile'] = self.wind_profile
        
        ### initialise generator fleet and demand profile

        generators = [agents.Generator(self, fuel_id, self.params) for fuel_id in 
              zip([item for sublist in [['nuclear']*self.params['num_nuclear'],
                                        ['wind']*self.params['num_wind'],
                                        ['coal']*self.params['num_coal'],
                                        ['ccgt']*self.params['num_ccgt']]
                                            for item in sublist], 
                                            range(self.params['num_gen']))] 
    
        for agent in generators:
            self.schedule.add(agent)
            
        max_capacity = sum(agent.capacity for agent in generators)       
        
        # TODO: standardise whether we're dealing in MW or MWh, these 2s 
        # everywhere are getting confusing
        
        self.demand = self.system_params['Demand']['Transmission Demand'] * 2  
        self.demand *= max_capacity / (params['peak_margin'] * max(self.demand))
        self.params['demand_profile'] = self.demand

        # initialise supply agents; [market share, forecast error, strategy]
        # TODO: initialise suppliers using params['num_suppliers'] 
        
        supplier_params = [[0.3, 0.03, 1], [0.2, 0.02, 0], [0.2, 0.02, 2],
                           [0.15, 0.04, 1], [0.1, 0.02, 0], [0.05, 0.01, 2]]
        
        self.suppliers = [agents.Supplier(params) for params in supplier_params]
        
        # internalise a labeled list of generators' id, fuel type, and capacity,
        # and assign each a unique colour from the xkcd colourset (949 + 10 standard)
        
        self.gen_labels = {}
        
        colours = ['C0', 'C1', 'C2', 'C7', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9']
        colours += list(mcolors.XKCD_COLORS.values())
        
        for agent in self.schedule.agents:
            
            self.gen_labels[agent.id] = [agent.fuel, agent.capacity, colours[agent.id]]
        
        
        
        # initialises constraints
        
        self.constraints = params['constraints']
                
        
        
        # initialise Mesa's datacollector functionality
        
        self.datacollector = DataCollector(
                agent_reporters = {'px_day_profit': 'px_day_profit',
                                   'px_period_profit': 'px_period_profit',
                                   'max_props': 'max_propensities',
                                   'px_price_props': 'px_price_propensities',
                                   'px_volume_props': 'px_volume_propensities',
                                   'bm_props' : 'bm_propensities',
                                   'expected_profits': 'expected_profits',
                                   'px_offers': 'px_offer',
                                   'bm_period_profit': 'bm_period_profit',
                                   'bm_day_profit': 'bm_day_profit'},
                model_reporters = {'dispatch_schedules': 'dispatch_schedules',
                                   'offer_curtailments': 'constrained_schedule',
                                   'gen_labels': 'gen_labels',
                                   'generation': 'generation',
                                   'imbalance': 'imbalance'})


    def constrained_scheduler(self, unconstrained_schedule, period):
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
            self.px_marginal_cost[period] = self.px_period_offers[period][satisfying_demand.index(0)]
            satisfying_demand[satisfying_demand.index(0)] = 1
            
        for i, (agent_id, dispatch) in enumerate(constrained_schedule.items()):
            constrained_schedule[agent_id] = satisfying_demand[i]
        
        for agent in self.schedule.agents:
            agent.last_dispatch = constrained_schedule[agent_id]
        
        
        
        return constrained_schedule
    
    
    
    def clear_market(self, offers, demand, period):
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
    
            demand_forecast = sum([bid[0] for bid in bids]) + offers[1][0] - self.wind_profile[period]/2
        
        else:
            
            demand_forecast = demand[period]/2 + offers[1][0] - self.wind_profile[period]/2

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
                return self.clear_market(offers, demand, period)
            
        # SIMPLE MARGINAL DISPATCH, IGNORES MINIMUM GENERATION LIMITS
        # the marginal generator thus dispatches partially to meet forecast
        # demand
#            
#        if 0 in accepted_offers:
#                
#            self.px_marginal_cost[period] = list(offers.items())[accepted_offers.index(0)][1][1]
#            accepted_offers[accepted_offers.index(0)] = (demand_forecast - cumulative_offers[accepted_offers.index(0) - 1]) \
#                                                        / list(offers.items())[accepted_offers.index(0)][1][0]
#                                                        
#        else:
#            
#            self.px_marginal_cost[period] = list(self.px_period_offers[period].items())[-1][1][1]
#        
        
        # ADVANCED MARGINAL GENERATION WITH CASES AND CONSTRAINTS
        
        if 0 in accepted_offers:
            
            self.px_marginal_cost[period] = list(offers.items())[accepted_offers.index(0)][1][1]
            
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
        
        
        # TODO: store imbalance as two separate components, due to uncertainty
        # demand forecasts and uncertainty from intermittent generators.
        # store as (total_imbal, intermittent_imbal, demand_imbal).
        # Positive imbalance means system is long, negative means short. 
        
        self.imbalance.append(demand_forecast - demand[period]/2)
        
        return accepted_offers
    
    
    def clear_balancing_market(self, imbalance, bm_offers, bm_bids, period):
        """ Accepts the outstanding physical imbalance for the period and the
        
        the generator bids and offers. Clears the market by matching the
        cheapest offers if the system is short or the most expensive bids if
        the system is long, returning an array of accepted actions and stores the 
        marginal cost of the most expensive action taken.
        """

        # if system is long, need to use bids to reduce the px generation
        if imbalance > 0:
            
            volume_bids = [bid[1] for bid in bm_bids]
            
            cumulative_bids = np.cumsum(volume_bids)
        
            bm_actions = list(np.where(cumulative_bids >= -imbalance, 
                                       1, 0))
            
            if 0 in bm_actions:
                
                self.bm_marginal_price[period] = bm_bids[bm_actions.index(0)][2]
                bm_actions[bm_actions.index(0)] = (imbalance + cumulative_bids[bm_actions.index(0) - 1]) \
                                                            / (-bm_bids[bm_actions.index(0)][1])
                                                        
            else:
            
                self.bm_marginal_price[period] = bm_bids[-1][2]
            
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
            
            if 0 in bm_actions:
                
                self.bm_marginal_price[period] = bm_offers[bm_actions.index(0)][2]
                bm_actions[bm_actions.index(0)] = (-imbalance + cumulative_offers[bm_actions.index(0) - 1]) \
                                                            / (bm_offers[bm_actions.index(0)][1])
                                                        
            else:
            
                self.bm_marginal_price[period] = bm_offers[-1][2]
                
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
        self.px_marginal_cost = OrderedDict()
        self.bm_marginal_price = OrderedDict()
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
            
            accepted_offers = self.clear_market(self.px_period_offers[period], 
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
#                constrained_schedule[period] = self.constrained_scheduler(unconstrained_schedule, period)
#                
#            elif self.constraints == 'none':
                
            self.constrained_schedule[period] = unconstrained_schedule 
            
            # save each agent's offer, what they dispatched in MWh, and what they
            # recieved (i.e. constrained payments), and store their last dispatch
            # state to update the constraint function
            # [fuel, px_dispatch, px_offer, bm_dispatch, [list of successful bm 
            # offers], total_dispatch]
            
            self.generation[period] = OrderedDict()
            for agent in self.schedule.agents:
#                agent.last_dispatch = self.constrained_schedule[period][agent.id]
                self.generation[period].update(
                        {agent.id: [agent.fuel,
                                    agent.px_offer[period][0] * self.constrained_schedule[period][agent.id],
                                    self.px_period_offers[period][agent.id][1],
                                    0, [],
                                    agent.px_offer[period][0] * self.constrained_schedule[period][agent.id]]})
            
        
        # create per-period dispatch schedules, consisting of tuples of the gen
        # id, and its dispatch for that period in MWh, in order of price offer,
        # as amended by any rational constraints. Note that this is the pre-
        # balancing mechanism dispatch, i.e. the FPN!
        
        self.dispatch_schedules = OrderedDict()
    
        for period in range(48):
        
            self.dispatch_schedules[period] = []
            
            for gen, accepted in self.constrained_schedule[period].items():
                if accepted != 0:
                    self.dispatch_schedules[period].append((gen, self.generation[period][gen][1]))
                else:
                    break
            
            
        # update generator propensities by passing it back its dispatch
        
        self.schedule.step(self.params['stage_list'][1], 
                           self.generation,
                           self.px_marginal_cost)
        
        
        ### BALANCING MECHANISM CLEARING ###
        
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
            
            # flattens list of lists
#            self.bm_period_offers[period] = [item for sublist in self.bm_period_offers[period] for item in sublist]
#            self.bm_period_bids[period] = [item for sublist in self.bm_period_bids[period] for item in sublist]
            
            # sorts by offer price   
            self.bm_period_offers[period] = sorted(self.bm_period_offers[period], 
                                                   key = lambda x: (x[2]))
            
            # bids are sorted in decending order, because the SO prefers those
            # who would pay the most to lower their output first
            self.bm_period_bids[period] = sorted(self.bm_period_bids[period], 
                                                 key = lambda x: (x[2]),
                                                 reverse = True)
        
            
            accepted_bm_actions = self.clear_balancing_market(self.imbalance[period],
                                                              self.bm_period_offers[period],
                                                              self.bm_period_bids[period],
                                                              period)
            
        
            # convert accepted_bm_actions to an ordered dictionary of 
            # agent: accepted actions, ready to update the generation dict
                        
            for action in accepted_bm_actions:
                self.generation[period][action[0]][3] += action[1]
                self.generation[period][action[0]][4].append(action)
                self.generation[period][action[0]][5] += action[1]
            
            self.bm_schedule[period] = accepted_bm_actions
        
        
        # update generator bm_mechanism propensities                
        
        self.schedule.step(self.params['stage_list'][2],
                           self.generation)
        
        self.datacollector.collect(self)
    


def running_mean(x, N):
    
    return pd.DataFrame(x).rolling(N).mean()[N:]
            



# runs the simulation, using mesa's batchrunner class to take in parameters to
# to vary during each run, the number of runs to make, and the elements to 
# return. Returns a graph of profit for each generator over time, a dispatch 
# graph in order to bid price for the final day, and an averaged generation
# fuel mix over the final 50 days

def run_simulation(model_class, params, days, show_graphs = False, save_graphs = False, iterate = False, name = None):
    
    params['days'] = days

    model = model_class(params)
    
    for i in tqdm(range(days), desc = 'Running Model: "{0}"'.format(name)):
        model.step()
    
    # agent variables from datacollector    
    px_day_profit = model.datacollector.get_agent_vars_dataframe()['px_day_profit']
    px_period_profit = model.datacollector.get_agent_vars_dataframe()['px_period_profit']
    px_price_props = model.datacollector.get_agent_vars_dataframe()['px_price_props']
    px_volume_props = model.datacollector.get_agent_vars_dataframe()['px_volume_props']
    bm_props = model.datacollector.get_agent_vars_dataframe()['bm_props']
    max_props = model.datacollector.get_agent_vars_dataframe()['max_props']
    px_offers = model.datacollector.get_agent_vars_dataframe()['px_offers']
    expected_profits = model.datacollector.get_agent_vars_dataframe()['expected_profits']
    bm_period_profit = model.datacollector.get_agent_vars_dataframe()['bm_period_profit']
    bm_day_profit = model.datacollector.get_agent_vars_dataframe()['bm_day_profit']
    
    # model variables from datacollector
    dispatch_schedules = model.datacollector.get_model_vars_dataframe()['dispatch_schedules']
    offer_curtailments = model.datacollector.get_model_vars_dataframe()['offer_curtailments']
    generation = model.datacollector.get_model_vars_dataframe()['generation']
    imbalance = model.datacollector.get_model_vars_dataframe()['imbalance']
    gen_labels = model.datacollector.get_model_vars_dataframe()['gen_labels']
    
    # The below code generates data averaging the dispatch mix, average
    # bid price, and peak bid price of dispatched generators 
    # per period averaged over the final 100 days of the simulation, along with
    # a 25-day MA of avg peak/off bid price for each fuel type over the whole sim
    
    avg_fuel_mix = pd.DataFrame()
    avg_offer_price = pd.DataFrame()
    peak_offer_price = pd.DataFrame()
    
    for period in tqdm(range(48), desc = 'Crunching data...'):
        all_gen = pd.DataFrame({agent_id: np.zeros(100) for agent_id in gen_labels.iloc[-1].keys()})
        all_offers = pd.DataFrame({agent_id: np.zeros(100) for agent_id in gen_labels.iloc[-1].keys()})
        
        for day in range(100):
            for agent_id in gen_labels.iloc[-1].keys():
                all_gen.iloc[day][agent_id] = generation.iloc[-day][period][agent_id][5]
                all_offers.iloc[day][agent_id] = np.where(generation.iloc[-day][period][agent_id][1] != 0, 
                                                          generation.iloc[-day][period][agent_id][2],
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
    
     
    rolling_offers_peak = pd.DataFrame({agent_id: np.zeros(days) for agent_id in gen_labels.iloc[-1].keys()})
    rolling_offers_off = pd.DataFrame({agent_id: np.zeros(days) for agent_id in gen_labels.iloc[-1].keys()})
    
    for day in tqdm(range(days), desc = 'Still crunching data...'):
            for agent_id in gen_labels.iloc[-1].keys():
                off_offers = []
                peak_offers = []
                for period in range(48):
                    if (period < 32 or period >= 42):
                        off_offers.append(float(np.where(list(generation.iloc[day].values())[period][agent_id][1] != 0, 
                                                   list(generation.iloc[day].values())[period][agent_id][2],
                                                   np.nan)))
                    elif (32 <= period < 42):
                        peak_offers.append(float(np.where(list(generation.iloc[day].values())[period][agent_id][1] != 0, 
                                                    list(generation.iloc[day].values())[period][agent_id][2],
                                                    np.nan)))
                        
                rolling_offers_peak.iloc[day][agent_id] = pd.Series(peak_offers).mean(skipna = True)
                rolling_offers_off.iloc[day][agent_id] = pd.Series(off_offers).mean(skipna = True) 
    
    rolling_offers_off.columns = rolling_offers_off.columns.to_series().apply(lambda gen: gen_labels.iloc[-1][gen][0])
    rolling_offers_off = rolling_offers_off.groupby(rolling_offers_off.columns, axis = 1).mean()
    
    rolling_offers_peak.columns = rolling_offers_peak.columns.to_series().apply(lambda gen: gen_labels.iloc[-1][gen][0])
    rolling_offers_peak = rolling_offers_peak.groupby(rolling_offers_peak.columns, axis = 1).mean()
            
    rolling_offers_off = running_mean(rolling_offers_off, 25)
    rolling_offers_peak = running_mean(rolling_offers_peak, 25)
    
    # calculates proportional dispatch values
    
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
    abs_prop_dispatch_error = sum(abs(prop_dispatch_error).sum(axis = 1))
    
    dispatch_error = (avg_fuel_mix - true_dispatch) * 2
    dispatch_error['total'] = dispatch_error.sum(axis = 1)
    
    rms_dispatch_error = m.sqrt(sum(dispatch_error['total']) ** 2)
    
    results = {'px_day_profit': px_day_profit,
               'px_period_profit': px_period_profit,
               'px_offers': px_offers,
               'px_price_props': px_price_props,
               'px_volume_props': px_volume_props,
               'bm_props': bm_props,
               'max_props': max_props,
               'dispatch_schedules': dispatch_schedules,
               'offer_curtailments': offer_curtailments,
               'expected_profits': expected_profits,
               'bm_period_profit': bm_period_profit,
               'bm_day_profit': bm_day_profit,
               'gen_labels': gen_labels,
               'generation': generation,
               'imbalance': imbalance,
               'avg_offer_price': avg_offer_price,
               'avg_fuel_mix': avg_fuel_mix,
               'rolling_off_price': rolling_offers_off,
               'rolling_peak_price': rolling_offers_peak,
               'dispatch_error': dispatch_error,
               'prop_dispatch_error': prop_dispatch_error,
               'rms_dispatch_error': rms_dispatch_error,
               'abs_prop_dispatch_error': abs_prop_dispatch_error}
    
    
    if show_graphs == True:
        
        # This graph produces the 5-day MA profit for each generator over the whole
        # simulation
        
        fig1 = plt.figure(1)
        ax1 = plt.subplot(111)
        for i in range(params['num_gen']):
            ax1.plot(range(days - 25), running_mean(px_day_profit[:,i], 25)/1000,
                    color = gen_labels[days-1][i][2],
                    label = gen_labels[days-1][i][0:2])
        ax1.set(title = '25-day MA Profit for Each Generator per Iteration',
                xlabel = 'Iteration (Day)',
                ylabel = 'Profit (£\'000)')
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        fig1 = ax1.get_figure()
                
        
        # This graph gives the actual dispatch of the final day, with gens
        # stacked in order of the price of their winning offer
            
        fig2 = plt.figure(2)
        ax2 = plt.subplot(111)
        for period in range(48):
            for i, (gen, dispatch) in enumerate(dispatch_schedules[days - 1][period]):
                ax2.bar(range(period, period+1),
#                        dispatch * 2,
                        generation.iloc[days-1][period][gen][5] * 2,
                        label = gen_labels[days-1][gen][0],
                        color = gen_labels[days-1][gen][2],
                        bottom = sum([generation.iloc[days-1][period][j][5] * 2 for j in [dispatch_schedules[days - 1][period][z][0] for z in range(i)]]))
        ax2.plot(model.demand, 'b-', linewidth = 2)
        ax2.set(title = 'Final Day Dispatch in PX Offer Price Order with Demand',
                xlabel = 'Iteration (Day)',
                ylabel = 'Generation and Demand (MW)')
        fig2 = ax2.get_figure()
        
        # Dispatch by fuel type, averaged over final 100 days 
                        
        fig3 = plt.figure(3)
        ax3 = (avg_fuel_mix.loc[:, ['nuclear', 'wind', 'coal', 'ccgt']] * 2).plot.bar(stacked = True) 
        ax3.set(xlabel = 'Period',
               xticks = list(range(0, 47, 2)),
               xticklabels = list(range(0, 47, 2)),
               ylabel = 'Generation (MW)', 
               title = 'Average Dispatch over Final 100 Days');
        fig3 = ax3.get_figure()
        
        # the true dispatch for the given date in params
        
        fig4 = plt.figure(4)
        ax4 = (model.system_params['Supply'][['Nuclear', 'Total Wind', 'Coal', 'CCGT']] * 2).plot.bar(stacked = True) 
        ax4.set(xlabel = 'Period',
               xticks = list(range(0, 47, 2)),
               xticklabels = list(range(0, 47, 2)),
               ylabel = 'Generation (MW)', 
               title = 'True Dispatch for {0}'.format(params['date'][0]));
        fig4 = ax4.get_figure()
        
        # average successful offer price for each fuel type over final 100 days
        
        fig5 = plt.figure(5)
        ax5 = avg_offer_price.loc[:, ['nuclear', 'wind', 'coal', 'ccgt']].plot()
        ax5.set(xlabel = 'Period',
                xticks = list(range(0, 47, 2)),
                xticklabels = list(range(0, 47, 2)),
                ylabel = 'Offer Price (£/MWh)', 
                title = 'Average Successful Offer Price over Final 100 Days');
        fig5 = ax5.get_figure()
        
        # rolling 25-day mean of successful offers per fuel type, for periods
        # 0-31 and 42-47
        
        fig6 = plt.figure(6)
        ax6 = rolling_offers_off.loc[:, ['nuclear', 'wind', 'coal', 'ccgt']].plot()
        ax6.set(xlabel = 'Day',
                xticks = list(range(25, days, 100)),
                xticklabels = list(range(25, days, 100)),
                ylabel = 'Offer Price (£/MWh)',
                title = '25-Day MA for Average Off-Peak Successful Offers')
        fig6 = ax6.get_figure()
        
        # as above, for periods 32-41
        
        fig7 = plt.figure(7)
        ax7 = rolling_offers_peak.loc[:, ['nuclear', 'wind', 'coal', 'ccgt']].plot()
        ax7.set(xlabel = 'Day',
                xticks = list(range(25, days, 100)),
                xticklabels = list(range(25, days, 100)),
                ylabel = 'Offer Price (£/MWh)',
                title = '25-Day MA for Average Peak Successful Offers')
        fig7 = ax7.get_figure()
        
        # proportional dispatch versions of the averaged and true dispatch graphs
        
        fig8 = plt.figure(8)
        ax8 = prop_model_dispatch[['nuclear', 'wind', 'coal', 'ccgt']].plot.bar(stacked = True) 
        ax8.set(xlabel = 'Period',
                xticks = list(range(0, 47, 2)),
                xticklabels = list(range(0, 47, 2)),
                ylabel = 'Proportion of Generation', 
                title = 'Average Proportional Dispatch over Final 100 Days');
        fig8 = ax8.get_figure()
        
        fig9 = plt.figure(9)
        ax9 = prop_true_dispatch[['nuclear', 'wind', 'coal', 'ccgt']].plot.bar(stacked = True) 
        ax9.set(xlabel = 'Period',
                xticks = list(range(0, 47, 2)),
                xticklabels = list(range(0, 47, 2)),
                ylabel = 'Proportion of Generation', 
                title = 'True Proportional Dispatch for {0}'.format(params['date'][0]));
        fig9 = ax9.get_figure()
        
        # error between true and model proportional dispatch as fuel- and 
        # total-time-series
        
        fig10 = plt.figure(10)
        ax10 = prop_dispatch_error[['nuclear', 'wind', 'coal', 'ccgt']].plot()
        ax10.set(xlabel = 'Period',
                 xticks = list(range(0, 47, 2)),
                 xticklabels = list(range(0, 47, 2)),
                 ylabel = 'Absolute Proportion Error (%)', 
                 title = 'Error Between True and Model Proportional Dispatch for {0}'.format(params['date'][0]));
        fig10 = ax10.get_figure()
        
        fig11 = plt.figure(11)
        ax11 = dispatch_error[['nuclear', 'wind', 'coal', 'ccgt', 'total']].plot()
        ax11.set(xlabel = 'Period',
                 xticks = list(range(0, 47, 2)),
                 xticklabels = list(range(0, 47, 2)),
                 ylabel = 'Absolute Error (MW)', 
                 title = 'Error Between True and Model Dispatch for {0}'.format(params['date'][0]));
        fig11 = ax11.get_figure()
        
    if save_graphs == True:
        
        # saves all graphs and a json dump of 'results' to a new directory named
        # after params['name'], along with the sim params to a csv indexed
        # by that name
        
        makedirs('Results/{0}'.format(name))
        fig1.savefig('Results/{0}/Profits.jpg'.format(name), dpi = 200)
        fig2.savefig('Results/{0}/Final Dispatch.jpg'.format(name), dpi = 200)
        fig3.savefig('Results/{0}/Average Dispatch.jpg'.format(name), dpi = 200)
        fig4.savefig('Results/{0}/True Dispatch.jpg'.format(name), dpi = 200)
        fig5.savefig('Results/{0}/Final Offer Prices.jpg'.format(name), dpi = 200)
        fig6.savefig('Results/{0}/Off-Peak Rolling Offers.jpg'.format(name), dpi = 200)
        fig7.savefig('Results/{0}/Peak Rolling Offers.jpg'.format(name), dpi = 200)
        fig8.savefig('Results/{0}/Average Proportional Dispatch.jpg'.format(name), dpi = 200)
        fig9.savefig('Results/{0}/True Proportional Dispatch.jpg'.format(name), dpi = 200)
        fig10.savefig('Results/{0}/Proportional Dispatch Error.jpg'.format(name), dpi = 200)
        fig11.savefig('Results/{0}/Absolute Dispatch Error.jpg'.format(name), dpi = 200)
        
        if iterate == True:
            plt.close('all')
        
        params['name'] = name
        params = orderHeaders(params)
        del params['wind_profile']
        del params['demand_profile']
        with open('Results/Graph Parameters.csv', 'a+', newline = '') as file:
            w = csv.DictWriter(file, params.keys())
            
            if getsize('Results/Graph Parameters.csv') == 0:
                w.writeheader()    
            else: 
                csv.writer(file).writerow([])

            w.writerow(params)
        
#        json_results = {key: value.to_json() for key, value in results}
#        
#        with open('Results/{0}/Results.json'.format(name), 'w') as dump:
#            dump.write(json.dumps(json_results))
        
    return results



def batchrun_simulation(model_class, fixed_params, days, 
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
        dispatch_schedules = results['Datacollector'].iloc[i].get_model_vars_dataframe()['dispatch_schedules']
        gen_labels = results['Datacollector'].iloc[i].get_model_vars_dataframe()['gen_labels'][days - 1]
        
        figure = plt.figure(i)
        for j in range(params['num_gen']):
            plt.plot(range(days - 5), running_mean(profit[:, j], 5))
            plt.title('Profit 5-Day MA with {0}: {1}'.format(results.columns[0], value))
            plt.legend([item for sublist in [['ccgt']*params['num_ccgt'], ['coal']*params['num_coal'], ['nuclear']*params['num_nuclear']] 
              for item in sublist])
        plt.show()
        
        x = i + len(results[results.columns[0]])
    
        # TODO: integrate 'generation' dataframe to properly sum up lower dispatches
        figure = plt.figure(x)    
        for period in range(48):
            for k, (gen, dispatch) in enumerate(dispatch_schedules[days - 1][period]):
                plt.bar(range(period, period + 1),
                        dispatch,
                        label = gen_labels[gen][0],
                        color = gen_labels[gen][2],
                        bottom = sum([gen_labels[h][1] for h in dispatch_schedules[days - 1][period][:k]]))
        plt.title('Final Dispatch Schedule with {0}: {1}'.format(results.columns[0], value))
        
    return results


def orderHeaders(params):
    param_order = ['name','days','peak_margin','demand_sd','date','num_gen','num_wind','wind','wind_sd','num_ccgt','ccgt','num_coal','coal',
                   'num_nuclear','nuclear', 'use_supplier_agents', 'num_suppliers','action_method','offer_method','constraints','temperature_inf','temperature_start','temperature_decay',
                   'expmt','recency','epsilon','dampening_factor','reward_method','discount','kernel_radius','stage_list']
    ordered_headers = OrderedDict([(k, None) for k in param_order if k in params])
    ordered_headers.update(params)
    return ordered_headers






    
    
    
    
    
    
    