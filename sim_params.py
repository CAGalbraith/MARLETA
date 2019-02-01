"""
# Created on Wed Oct 17 13:34:23 2018
# @author: connorgalbraith
"""
import numpy as np
import pandas as pd
import math as m
from os import makedirs
import seaborn as sns

import model
import visualisations as vis

# the parameters below are default fixed values; if they are to be made variable
# in mesa's batchrunner, then a separate variable parameter dictionary is
# specified and the corresponding keys removed from the fixed parameter
# dictionary passed

# CHANGE ORDERED HEADERS FUNCTION WHEN NEW PARAMETERS ARE ADDED

### learning parameters ###

learning_params = {}

# for now, this sets the learning algorithm for all agents, need to update to 
# specify a unique method for each agent. Choose from 'VERA', 'QLearning',
# 'Stateful_QLearning'
learning_params['learning_mechanism'] = 'Stateful_QLearning'

# chooses between softmax, epsilon_greedy, epsilon_kernel, or Simulated-Annealing
# Q-Learning (SA-Q) for converting propensities to actions
learning_params['px_action_method'] = 'SA-Q'
learning_params['bm_action_method'] = 'epsilon_greedy'

# either holds the bm-ladder gradient at a fixed value, or lets the agents 
# learn their own for each period (in which case, use 0)
learning_params['bm_gradient_hold'] = 0

# forces agents to learn only a single bid/offer ladder for the entire day, 
# resulting in a dramatically simpler but under-optimised BM
learning_params['single_bm_ladder'] = False

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
learning_params['px_epsilon_start'] = 0.25
learning_params['px_epsilon_decay'] = 1.2 * (m.log(
        (learning_params['px_epsilon_start'] -
         learning_params['px_epsilon_inf']) /
        (0.1 * learning_params['px_epsilon_inf'])))
    
learning_params['bm_epsilon_inf'] = 0.05
learning_params['bm_epsilon_start'] = 0.25
learning_params['bm_epsilon_decay'] = 1.2 * (m.log(
        (learning_params['bm_epsilon_start'] -
         learning_params['bm_epsilon_inf']) /
        (0.1 * learning_params['bm_epsilon_inf'])))

# switches between using the per-period profit or the day-total profit as 
# reward, either 'day_profit', 'period_profit', 'kernel_profit', or 'expected_profit'
learning_params['px_reward_method'] = 'period_profit'
learning_params['bm_reward_method'] = 'period_profit'

learning_params['discount'] = 0.4
learning_params['kernel_radius'] = 1

# controls the magnitude of the rewards that the agents receive for a given 
# action, with the dampener used to divide the reward
learning_params['dampening_factor'] = 0.5


### physical parameters: MW, £marginal/MWh, £/startup, cycles, minimum up-time, 
### minimum down-time, tCO2/MWh
phys_params = {
    'wind': {'cap': 1000, 'marginal_cost': 0, 'startup_cost/MW': 0, 'min_gen_prop': 0, 'cycles': 0, 'emissions': 0},
    'ccgt': {'cap': 800, 'marginal_cost': 39, 'startup_cost/MW': 17.4, 'min_gen_prop': 0.5, 'cycles': 3, 'emissions': 0.487},
    'coal': {'cap': 1800, 'marginal_cost': 30, 'startup_cost/MW': 30.8, 'min_gen_prop': 0.5, 'cycles': 2, 'emissions': 0.87},
    'nuclear': {'cap': 2400, 'marginal_cost': 10, 'startup_cost/MW': 4000, 'min_gen_prop': 1, 'cycles': 0, 'emissions': 0},
    }

### simulation parameters ###

sim_params = {}

# defines the agent methods that are going to be called during each step
sim_params['stage_list'] = ['make_px_offer',
                            'update_px_propensities',
                            'update_bm_propensities']

# specifies whether the balancing market is used - should only be True when there
# is no other source of imbalance
sim_params['balancing_mechanism'] = True

# specifies whether the SO applied the 4-case curtailement approach in the PX
# to meet forecast demand entirely, if set to 'advanced'. If set to 'simple',
# SO will try to minimise the different between PX dispatch and forecast demand
# by either dispatching the marginal generator or not, within phyiscal limits
sim_params['px_clearing_method'] = 'advanced' 

# controls whether stochastic additions are made to demand and wind, and their 
# severity. If defining synthetic imbalance, set both sd values to 0.
sim_params['demand_sd'] = 0

sim_params['wind_sd'] = 0.01

sim_params['synthetic_imbalance'] = []

# specifies whether a synthetic demand profile is being used. If not, leave [],
# otherwise specifiy in MW
sim_params['synthetic_demand'] = []

# specifies whether run-time constraints are in effect; 'none', 'soft' for cycling
# penalties only, or 'all' to include boundedly-rational heuristics
sim_params['constraints'] = 'all'

# flag as True to capture the propensities of each agent for the last day of
# the run for transfer purposes, rather than save every day's props
sim_params['capture_props'] = True

# factors in an asymmetric penalty for agents if the system is left short by 
# the end of the balancing mechanism
sim_params['VoLL'] = 9000

# determines whether the agents learn against a fixed or a dynamic imbalance
# profile, i.e. whether a new set of demand/wind errors are calculated each day
sim_params['dynamic_imbalance'] = False

# scales the demand such that total generation capacity = max demand * margin
sim_params['peak_margin'] = 1.36

# the date from which to take the demand and wind profiles
sim_params['date'] = ['2015-08-01', '2015-08-02']

# define number of and types of generators
sim_params['num_wind'] = 1
sim_params['num_ccgt'] = 6
sim_params['num_coal'] = 2
sim_params['num_nuclear'] = 1
sim_params['num_agents'] = (
        sim_params['num_wind'] + 
        sim_params['num_ccgt'] + 
        sim_params['num_coal'] + 
        sim_params['num_nuclear'])

sim_params['max_capacity'] = sum([sim_params['num_wind'] * phys_params['wind']['cap'] + 
                                  sim_params['num_ccgt'] * phys_params['ccgt']['cap'] + 
                                  sim_params['num_coal'] * phys_params['coal']['cap'] +
                                  sim_params['num_nuclear'] * phys_params['nuclear']['cap']])

sim_params['use_supplier_agents'] = False
sim_params['num_suppliers'] = 6

# a dictionary of user-specified propensity sets, indexed to [gen_id], allowing
# learning from previous runs to be transferred. For a uniform set of ones, set
# as: {x: {} for x in range(sim_params['num_agents'])}, otherwise for manual:
# e.g. {0: {'px_price': [], 'px_volume': [], 'bm_intercept': [], 'bm_gradient': []}} 
# or transfer: results['last_props']
     
learning_params['propensities'] = {x: {} for x in range(sim_params['num_agents'])}

# an additional dict of agent_id: {prop_name: value} to specifiy agent-specific
# heterogeneous properties
learning_params['heterogeneous_parameters'] = {}

params = {**learning_params, **phys_params, **sim_params}


# =============================================================================
# Single model runs, and functions for additional analysis from model.py
# =============================================================================

results = model.runSimulation(model.MarketModel, 
                              params, 
                              days = 500, 
                              name = 'State Test',
                              verbose = False)



# =============================================================================
# Various visualisation options
# =============================================================================

makedirs('Results/{0}'.format(results['name']))

sns.set_style('dark', {'axes.grid': True,
                       'xtick.bottom': True,
                       'ytick.left':True})

vis.basicGraphs(results, 
                params, 
                save_graphs = False)

vis.bmGraphs(results, 
             periods = [25, 38],
             save_graphs = False)

vis.individualOffers(results, 
                     agent_ids = [2, 3], 
                     periods = [24], 
                     one_gen_per_graph = True, 
                     ma = 10,
                     save_graphs = False)

vis.pxStrategyEvolution(results, 
                        agent_id = 2, 
                        step_size = 25)

# specify if verbose, in order to get intercept and gradient choices
vis.bmLadderEvolution(results, 
                      agent_id = 4, 
                      period = 20, 
                      verbose = False,
                      save_graphs = False)

vis.systemPrices(results, 
                 cost_graphs = False,
                 periods = [2, 14, 36],
                 save_graphs = False)

vis.rawDispatch(results, 
                days = 1, 
                save_graphs = False)

vis.emissions(results,
              periods = [2, 14, 36],
              save_graphs = False)




# =============================================================================
# Custom batchrunner, only works currently for two variable parameters at a
# time, and is generally just shit
# =============================================================================

variable_params = {}
run_count = 0
exp_run_count = np.prod([len(x) for x in variable_params.values()])

def iterateSimulation(params, variable_params, days, verbose):
    """ A custom version of Mesa's batchrunner, that allows for agent reporters
    
    to collect data alongside the model reporters during each run. Accepts a
    nested dictionary of parameter-name: [values] pairs to iterate through,
    returns the results from each run in a dictionary indexed by name of run.
    """
    
    global run_count
    global exp_run_count
        
    # create array of all possible combinations of variable parameters
    
    results_dict = dict()
    
    if len(list(variable_params.keys())) != 1:
                
        variable_param_name = list(variable_params.keys())[0]
        variable_param_values = variable_params.pop(variable_param_name)
                            
        for param_value in variable_param_values:
            
            name = '{0}: {1}'.format(variable_param_name, param_value)
            params[variable_param_name] = param_value
            
            results_dict[name] = iterateSimulation(params, variable_params, days, verbose)

    
    else:
        
        variable_param_name = list(variable_params.keys())[0]
        variable_param_values = variable_params[variable_param_name]
        
        run_count += len(variable_param_values)
        
        if run_count > exp_run_count:
        
            return results_dict 
        
        else:
            
            for param_value in variable_param_values:
                
                name = '{0}: {1}'.format(variable_param_name, param_value)
                
                params[variable_param_name] = param_value
                
                results_dict[name] = model.runSimulation(model.MarketModel, 
                                                         params, 
                                                         days = days, 
                                                         name = name,
                                                         verbose = verbose)
        
    return results_dict



# =============================================================================
# For the batchrunner
# =============================================================================

# defines which variables to vary over each iteration, and returns a dictionary
# of fixed_variables to be passed to the batchrunner

variable_params = {'kernel_radius': [2, 3]}


def without_keys(d, variable_params):
    return {x: d[x] for x in d if x not in variable_params}


fixed_params = without_keys(params, variable_params)
results = model.batchrunSimulation(model_class = model.MarketModel,
                                   fixed_params = fixed_params,
                                   days = 1000,
                                   iterations = 1,
                                   variable_params = variable_params)



# =============================================================================
# Custom batchrunner designed for saving graphs only - results is overwritten
# TODO: include new differentiated graphing functions here too
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


