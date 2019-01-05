#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:34:23 2018

@author: connorgalbraith
"""
import model
import math as m

# the parameters below are default fixed values; if they are to be made variable
# in mesa's batchrunner, then a separate variable parameter dictionary is
# specified and the corresponding keys removed from the fixed parameter
# dictionary passed

# CHANGE ORDERED HEADERS FUNCTION WHEN NEW PARAMETERS ADDED

### learning parameters ###

learning_params = {}

# chooses between softmax, epsilon_greedy, or epsilon_kernel for converting propensities to actions
learning_params['action_method'] = 'softmax'

# the hotter it gets, the more likely the agents will explore new strategies.
# for constant T, set both _start and _inf as equal
learning_params['temperature_inf'] = 1.999
learning_params['temperature_start'] = 2
learning_params['temperature_decay'] = 1.25 * (m.log(
        (learning_params['temperature_start'] -
         learning_params['temperature_inf']) /
        (0.1 * learning_params['temperature_inf'])))

# 'experimentation', controls degree that profit increases the propensity of that action
learning_params['expmt'] = 0.95

# how quickly propensities decay. Recency must be greater than expmt/(len(action_set) - 1) 
learning_params['recency'] = 0.05

# epsilon if epsilon-greedy is being used. TO DO: experiment with lower propensity
# limit below which epsilon-greedy won't consider actions
learning_params['epsilon'] = 0.05

# switches between using the per-period profit or the day-total profit as 
# reward, either 'day_profit', 'period_profit', 'discounted_profit', or
# 'kernel_profit'
learning_params['reward_method'] = 'kernel_profit'

learning_params['discount'] = 0.3
learning_params['kernel_radius'] = 1

# controls the magnitude of the rewards that the agents receive for a given,
# in the denominator
learning_params['dampening_factor'] = 3


### physical parameters: MW, £marginal/MWh, £/startup, cycles, minimum up-time, 
### minimum down-time, tCO2/MWh
phys_params = {
    'wind': {'cap': 4250, 'marginal_cost': 0, 'startup_cost': 0, 'min_gen': 0, 'cycles': 0, 'emissions': 0},
    'ccgt': {'cap': 2600, 'marginal_cost': 39, 'startup_cost': 40000, 'min_gen': 1300, 'cycles': 3, 'emissions': 0.487},
    'coal': {'cap': 4600, 'marginal_cost': 30, 'startup_cost': 200000, 'min_gen': 2300, 'cycles': 2, 'emissions': 0.87},
    'nuclear': {'cap': 7000, 'marginal_cost': 10, 'startup_cost': 200000, 'min_gen': 7000, 'cycles': 0, 'emissions': 0},
    }

### simulation parameters ###

sim_params = {}

# defines the agent methods that are going to be called during each step
sim_params['stage_list'] = ['make_px_offer',
                            'update_px_propensities',
                            'update_bm_propensities']

# specify whether the agents use absolute offers or increments on the bid they
# made during that period in the previous day. 'absolute' or 'increment'
sim_params['offer_method'] = 'absolute'

# specifies whether run-time constraints are in effect; 'none', 'soft' for cycling
# penalties only, or 'all' to include boundedly-rational heuristics
sim_params['constraints'] = 'all'

# controls whether stochastic additions are made to demand and wind, and their severity
sim_params['demand_sd'] = 0

sim_params['wind_sd'] = phys_params['wind']['cap']/12

# scales the demand such that total generation capacity = max demand * margin
sim_params['peak_margin'] = 1.36

# the date from which to take the demand and wind profiles
sim_params['date'] = ['2016-03-15', '2016-03-16']

# define number of and types of generators
sim_params['num_wind'] = 1
sim_params['num_ccgt'] = 4
sim_params['num_coal'] = 2
sim_params['num_nuclear'] = 1
sim_params['num_gen'] = (
        sim_params['num_wind'] + 
        sim_params['num_ccgt'] + 
        sim_params['num_coal'] + 
        sim_params['num_nuclear'])

sim_params['use_supplier_agents'] = False
sim_params['num_suppliers'] = 6

params = {**learning_params, **phys_params, **sim_params}


# defines which variables to vary over each iteration, and returns a dictionary
# of fixed_variables to be passed to the batchrunner

variable_params = {'kernel_radius': [2, 3]}


def without_keys(d, variable_params):
    return {x: d[x] for x in d if x not in variable_params}


fixed_params = without_keys(params, variable_params)



results = model.run_simulation(model.MarketModel, 
                               params, 
                               days = 500, 
                               show_graphs = True,
                               save_graphs = False,
                               iterate = False,
                               name = 'BM Test')




results = model.batchrun_simulation(model_class = model.MarketModel,
                                    fixed_params = fixed_params,
                                    days = 1000,
                                    iterations = 1,
                                    variable_params = variable_params)


for scenario in [[1000, 2, 6], [2500, 1, 6], [5000, 1, 3]]:
    params['wind'][0] = scenario[0]
    params['wind_sd'] = scenario[0]/8
    params['num_coal'] = scenario[1]
    params['num_ccgt'] = scenario[2]
    params['num_gen'] = params['num_coal'] + params['num_wind'] + params['num_nuclear'] + params['num_ccgt']
    wind_prop = round(100 * params['wind'][0] / (params['num_coal']*1800 + 2400 + params['num_ccgt']*800 + scenario[0]), ndigits = 1)
    model.run_simulation(model.MarketModel, 
                         params, 
                         days = 10000, 
                         show_graphs = True,
                         save_graphs = True,
                         iterate = True,
                         name = 'Wind Prop 2 = {0}%'.format(wind_prop))
















