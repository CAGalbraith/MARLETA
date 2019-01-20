#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:34:23 2018

@author: connorgalbraith
"""
import numpy as np
import pandas as pd
import math as m

import model

# the parameters below are default fixed values; if they are to be made variable
# in mesa's batchrunner, then a separate variable parameter dictionary is
# specified and the corresponding keys removed from the fixed parameter
# dictionary passed

# CHANGE ORDERED HEADERS FUNCTION WHEN NEW PARAMETERS ARE ADDED

### learning parameters ###

learning_params = {}

# chooses between softmax, epsilon_greedy, or epsilon_kernel for converting propensities to actions
learning_params['px_action_method'] = 'epsilon_greedy'
learning_params['bm_action_method'] = 'epsilon_greedy'

# either holds the bm-ladder gradient at a fixed value, or lets the agents 
# learn their own for each period (in which case, use 0)
learning_params['bm_gradient_hold'] = 10

# the hotter it gets, the more likely the agents will explore new strategies.
# for constant T, set both _start and _inf as equal
learning_params['px_temperature_inf'] = 0.05
learning_params['px_temperature_start'] = 1
learning_params['px_temperature_decay'] = 1.25 * (m.log(
        (learning_params['px_temperature_start'] -
         learning_params['px_temperature_inf']) /
        (0.1 * learning_params['px_temperature_inf'])))
    
learning_params['bm_temperature_inf'] = 0.005
learning_params['bm_temperature_start'] = 0.5
learning_params['bm_temperature_decay'] = 1.25 * (m.log(
        (learning_params['bm_temperature_start'] -
         learning_params['bm_temperature_inf']) /
        (0.1 * learning_params['bm_temperature_inf'])))

# 'experimentation', controls degree that profit increases the propensity of that action
learning_params['px_expmt'] = 0.95
learning_params['bm_expmt'] = 0.5

# how quickly propensities decay. Recency must be greater than expmt/(len(action_set) - 1) 
learning_params['px_recency'] = 0.05
learning_params['bm_recency'] = 0.05

# epsilon if epsilon-greedy is being used.
learning_params['px_epsilon_inf'] = 0.05
learning_params['px_epsilon_start'] = 0.2
learning_params['px_epsilon_decay'] = 0.5 * (m.log(
        (learning_params['px_epsilon_start'] -
         learning_params['px_epsilon_inf']) /
        (0.1 * learning_params['px_epsilon_inf'])))
    
learning_params['bm_epsilon_inf'] = 0.05
learning_params['bm_epsilon_start'] = 0.2
learning_params['bm_epsilon_decay'] = 0.5 * (m.log(
        (learning_params['bm_epsilon_start'] -
         learning_params['bm_epsilon_inf']) /
        (0.1 * learning_params['bm_epsilon_inf'])))

# switches between using the per-period profit or the day-total profit as 
# reward, either 'day_profit', 'period_profit', 'discounted_profit', 
# 'kernel_profit', or 'expected_profit'
learning_params['px_reward_method'] = 'period_profit'
learning_params['bm_reward_method'] = 'period_profit'

learning_params['discount'] = 0.4
learning_params['kernel_radius'] = 1

# controls the magnitude of the rewards that the agents receive for a given,
# in the denominator
learning_params['dampening_factor'] = 1.5


### physical parameters: MW, £marginal/MWh, £/startup, cycles, minimum up-time, 
### minimum down-time, tCO2/MWh
phys_params = {
    'wind': {'cap': 500, 'marginal_cost': 0, 'startup_cost': 0, 'min_gen': 0, 'cycles': 0, 'emissions': 0},
    'ccgt': {'cap': 800, 'marginal_cost': 39, 'startup_cost': 13920, 'min_gen': 400, 'cycles': 3, 'emissions': 0.487},
    'coal': {'cap': 1800, 'marginal_cost': 30, 'startup_cost': 55440, 'min_gen': 900, 'cycles': 2, 'emissions': 0.87},
    'nuclear': {'cap': 2400, 'marginal_cost': 10, 'startup_cost': 200000, 'min_gen': 2400, 'cycles': 0, 'emissions': 0},
    }

### simulation parameters ###

sim_params = {}

# defines the agent methods that are going to be called during each step
sim_params['stage_list'] = ['make_px_offer',
                            'update_px_propensities',
                            'update_bm_propensities']

# controls whether mesa stores all variables in the model or limits to critical
sim_params['verbose_data_collection'] = True

# specifies whether the balancing market is used - should only be True when there
# is no other source of imbalance
sim_params['balancing_mechanism'] = True

# specifies whether a synthetic demand profile is being used. If not, choose 0,
# otherwise specifiy in MW
sim_params['synthetic_demand'] = 0

# specifies whether run-time constraints are in effect; 'none', 'soft' for cycling
# penalties only, or 'all' to include boundedly-rational heuristics
sim_params['constraints'] = 'all'

# determines whether the agents learn against a fixed or a dynamic imbalance
# profile, i.e. whether a new set of demand/wind errors are calculated each day
sim_params['dynamic_imbalance'] = False

# controls whether stochastic additions are made to demand and wind, and their severity
sim_params['demand_sd'] = 0

sim_params['wind_sd'] = 0.01

# scales the demand such that total generation capacity = max demand * margin
sim_params['peak_margin'] = 1.36

# the date from which to take the demand and wind profiles
sim_params['date'] = ['2016-03-15', '2016-03-16']

# define number of and types of generators
sim_params['num_wind'] = 1
sim_params['num_ccgt'] = 4
sim_params['num_coal'] = 2
sim_params['num_nuclear'] = 1
sim_params['num_agents'] = (
        sim_params['num_wind'] + 
        sim_params['num_ccgt'] + 
        sim_params['num_coal'] + 
        sim_params['num_nuclear'])

sim_params['use_supplier_agents'] = False
sim_params['num_suppliers'] = 6

params = {**learning_params, **phys_params, **sim_params}


# =============================================================================
# Single model runs, and functions for additional analysis from model.py
# =============================================================================

results = model.run_simulation(model.MarketModel, 
                               params, 
                               days = 250, 
                               show_graphs = True,
                               save_graphs = False,
                               iterate = False,
                               name = 'BM Test')

agent_id = 2
period = 42
num_agents = params['num_agents']
days = 250
step_size = 25
name = 'Coal'

model.additional_graphs(results, num_agents, days, agent_id)


agent_ids = [2,4]
periods = [2,16,24,38]
one_gen_per_graph = True

model.individual_offers(results, agent_ids, periods, one_gen_per_graph)


# make dir to store gif images; there can be many is results is large
makedirs('Strategy Evolutions/{0}'.format(name))

model.px_strategy_evolution(results, agent_id, num_agents, days, step_size, name)

model.bm_ladder_evolution(results, agent_id, period)

model.system_cost_graphs(results)

model.final_five_day_dispatch(results)




# =============================================================================
# For the batchrunner
# =============================================================================

# defines which variables to vary over each iteration, and returns a dictionary
# of fixed_variables to be passed to the batchrunner

variable_params = {'kernel_radius': [2, 3]}


def without_keys(d, variable_params):
    return {x: d[x] for x in d if x not in variable_params}


fixed_params = without_keys(params, variable_params)
results = model.batchrun_simulation(model_class = model.MarketModel,
                                    fixed_params = fixed_params,
                                    days = 1000,
                                    iterations = 1,
                                    variable_params = variable_params)



# =============================================================================
# Custom batchrunner designed for savign graphs only - results is overwritten
# =============================================================================
for scenario in [[1000, 2, 6], [2500, 1, 6], [5000, 1, 3]]:
    params['wind'][0] = scenario[0]
    params['wind_sd'] = scenario[0]/8
    params['num_coal'] = scenario[1]
    params['num_ccgt'] = scenario[2]
    params['num_agents'] = params['num_coal'] + params['num_wind'] + params['num_nuclear'] + params['num_ccgt']
    wind_prop = round(100 * params['wind'][0] / (params['num_coal']*1800 + 2400 + params['num_ccgt']*800 + scenario[0]), ndigits = 1)
    model.run_simulation(model.MarketModel, 
                         params, 
                         days = 10000, 
                         show_graphs = True,
                         save_graphs = True,
                         iterate = True,
                         name = 'Wind Prop 2 = {0}%'.format(wind_prop))
















