#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:45:55 2019

@author: connorgalbraith
"""
import numpy as np
from matplotlib import pyplot as plt


# =============================================================================
# Functions for diagnostics
# =============================================================================

def softmax(x, temperature):
    """Compute softmax values for each sets of scores in x."""
    
    e_x = np.exp((x - np.max(x))/temperature)
    
    return e_x / e_x.sum(axis=0)


def create_kernel(radius, discount):
    """Creates a softmaxed smoothing kernel for the kernel_profit method, 
    
    with width 1 + 2*radius and a discount factor = discount.
    """

    kernel = np.ones(1 + 2*radius)
    
    for i in range(radius):
        kernel[i]  *= discount ** (radius - i)
        kernel[2*radius - i] *= discount ** (radius - i)
    
    return softmax(kernel, 0.2)


def moving_average(x, N):
    
    return pd.DataFrame(x).rolling(N).mean()[N:]


# =============================================================================
# Autoregressive and mean-reverting forecast error creation
# =============================================================================

e = np.zeros(100)
e[0] = 0.2*np.random.normal(scale = 10)
e[1] = 0.5*e[0] - 0.3*e[0] + 0.2*np.random.normal(scale = 10)
e[2] = 0.5*e[1] + 0.3*e[0] - 0.3*e[1] + 0.2*np.random.normal(scale = 10)

for i in range(3, 100):
    e[i] = e[i-1] + 0.2 * e[i-2] + 0.25*(0 - e[i-1]) + 0.1*np.random.normal(scale = 10)

plt.plot(e)

wind_errors = np.zeros(48)
wind_errors[0] = np.random.normal(scale = 25)
wind_errors[1] = 0.2 * wind_errors[0] + 0.5 * np.random.normal(scale = 25) 
for i in range(2, 48):
    wind_errors[i] = wind_errors[i-1] + 0.2*wind_errors[i-2] + 0.25*(0 - wind_errors[i-1]) + 0.25*np.random.normal(scale = 25)
plt.plot(wind_errors)


# =============================================================================
# Analysing BM Ladder formation using increments vs other methods
# =============================================================================
            
# decomposes the bm_parameters part of results into individual arrays of intercept
# and gradient for each agent and period

agent_id = 2
bm_intercepts = {}
bm_gradients = {}

for period in [4, 24]:
    bm_intercepts[period] = [results['bm_intercept_choices'][agent_id::10][j].loc[agent_id][period] for j in range(1000)]
    bm_gradients[period] = [results['bm_gradient_choices'][agent_id::10][j].loc[agent_id][period] for j in range(1000)]

plt.plot(bm_intercepts[4])
plt.plot(bm_intercepts[24])
plt.plot(bm_gradients[4])
plt.plot(bm_gradients[24])


sbp_long = []
for _ in sbp:
    if _[12][0] == 1:
        sbp_long.append(_[12][1])

plt.plot(sbp_long)
        
sbp_short = []
for _ in sbp:
    if _[8][0] == 0:
        sbp_short.append(_[8][1])

plt.plot(sbp_short)    

avg_bm_marginal_price = []
for period in range(48):
    avg_bm_marginal_price.append([np.mean([bm_marginal_price.iloc[-day][period][1] for day in range(100)]), bm_marginal_price.iloc[-1][period][0]])


fig12 = plt.figure(12)
plt.show()
plt.clf()
fig12 = plt.figure(12)
direction = np.array([avg_bm_marginal_price[period][0] for period in range(48)])
color = np.where(direction == 0, 'r', 'b')
fig12 = plt.plot([avg_bm_marginal_price[period][1] for period in range(48)])
fig12 = plt.scatter(range(48), [avg_bm_marginal_price[period][1] for period in range(48)], color = color) 
red = mpatches.Patch(color = 'red', label = 'Short')
blue = mpatches.Patch(color = 'blue', label = 'Long')
plt.xlabel('Period')
plt.xticks(list(range(0, 47, 2)))
plt.ylabel('Price (£/MWh)')
plt.legend(handles = [red, blue])
plt.title('Average Balancing Mechanism Marginal Price over Final 100 Days for {0}'.format(params['date'][0]));
 
# =============================================================================
# More attempts at different visualisations
# =============================================================================

# like the tracking of avea





    
# =============================================================================
# Experimenting with actions based on expected profits. For some reason, despite
# being sampled every day, it overwrites all previous samples with the new
# reading, so they're all identical to the final day's readings. Need to change
# the storage method to deliberately keep a new copy for each day to track changes
# =============================================================================

exp_profits = results['expected_profits']

agent_id = 2
offer = 45
period = 34

exp_profits_agent = np.array([x[4] for x in [exp_profits[y][agent_id][offer][period] for y in range(101)]])
exp_reward_agent = np.array([x[7] for x in [exp_profits[y][agent_id][offer][period] for y in range(101)]])
plt.plot(exp_profits_agent)
plt.plot(exp_reward_agent)

# before transforming to a pdf, subtract minimium from each value so there're 
# no negatives, and the worst action goes to zero probability

exp_profits_agent -= min(exp_profits_agent)

weights = exp_profits_agent/np.linalg.norm(exp_profits_agent, ord = 1)

plt.plot(weights)


px_epsilon_inf = 0.01
px_epsilon_start = 0.1
px_epsilon_decay = 0.75 * (m.log((px_epsilon_start -px_epsilon_inf) /(0.1 * px_epsilon_inf)))
days = 500
px_epsilon = []

for day in range(days):
    px_epsilon.append(px_epsilon_inf + (px_epsilon_start - px_epsilon_inf) \
                      * np.exp(-day/(days/px_epsilon_decay)))
plt.plot(px_epsilon)





























