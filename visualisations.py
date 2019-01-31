#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:45:28 2019

@author: connorgalbraith
"""
import numpy as np
import pandas as pd
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns

import csv
from os.path import getsize
from os import makedirs

def orderHeaders(params):
    """ A very unsatisfactory way of making sure the various parameters in 
    
    params line up with each other in the csv file header. Be sure to add to 
    when new parameters are added.
    """
    
    param_order = ['name','days','peak_margin','demand_sd','date','num_agents','num_wind','wind','wind_sd',
                   'num_ccgt','ccgt','num_coal','coal','num_nuclear','nuclear', 'use_supplier_agents', 
                   'num_suppliers','max_capacity','synthetic_demand','synthetic_imbalance','px_clearing_method','dynamic_imbalance',
                   'px_action_method','bm_action_method','bm_gradient_hold','single_bm_ladder','constraints','VoLL','balancing_mechanism','px_temperature_inf',
                   'px_temperature_start','px_temperature_decay','bm_temperature_inf','bm_temperature_start','bm_temperature_decay',
                   'learning_mechanism','px_expmt','bm_expmt','px_recency','bm_recency','px_epsilon_inf','px_epsilon_start',
                   'px_epsilon_decay','bm_epsilon_inf','bm_epsilon_start','bm_epsilon_decay','dampening_factor',
                   'px_reward_method','bm_reward_method','discount','kernel_radius','heterogeneous_params','stage_list']
    
    ordered_headers = OrderedDict([(k, None) for k in param_order if k in params])
    ordered_headers.update(params)
    
    return ordered_headers



def movingAverage(x, N):
    
    return pd.DataFrame(x).rolling(N).mean()[N:]




def basicGraphs(results, params, save_graphs):        
    
    # model inputs
    gen_labels = results['gen_labels']
    model_demand = results['demand']
    true_dispatch = results['true_dispatch']
    days = len(results['generation'])
    name = results['name']
    
    # raw model outputs
    px_day_profit = results['px_day_profit']
    px_dispatch = results['px_dispatch']
    generation = results['generation']
    
    # derived model outputs computed in run_simulation()
    avg_fuel_mix = results['avg_fuel_mix']
    avg_offer_price = results['avg_offer_price']
    rolling_offers_off = results['rolling_offers_off']
    rolling_offers_peak = results['rolling_offers_peak']
    rolling_offers_volume = results['rolling_offers_volume']
    prop_model_dispatch = results['prop_model_dispatch']
    prop_true_dispatch = results['prop_true_dispatch']
    prop_dispatch_error = results['prop_dispatch_error']
    dispatch_error = results['dispatch_error']
    
                
    # This graph produces the 25-day MA profit for each generator over the whole
    # simulation
    
    fig1 = plt.figure(1)
    ax1 = plt.subplot(111)
    for i in range(params['num_agents']):
        ax1.plot(range(days - 25), movingAverage(px_day_profit[:,i], 25)/1000,
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
        for i, (gen, dispatch) in enumerate(px_dispatch[days - 1][period]):
            ax2.bar(range(period, period+1),
#                        dispatch * 2,
                    generation.iloc[days-1][period][gen][5] * 2,
                    label = gen_labels[days-1][gen][0],
                    color = gen_labels[days-1][gen][2],
                    bottom = sum([generation.iloc[days-1][period][j][5] * 2 for j in [px_dispatch[days - 1][period][z][0] for z in range(i)]]))
    ax2.plot(model_demand, 'b-', linewidth = 2)
    ax2.set(title = 'Final Day Dispatch in PX Offer Price Order with Demand',
            xlabel = 'Period',
            ylabel = 'Generation and Demand (MW)',
            xticks = list(range(0, 47, 2)),
            xticklabels = list(range(0, 47, 2)))
    fig2 = ax2.get_figure()
    
    # Dispatch by fuel type, averaged over final 50 days 
                    
    fig3 = plt.figure(3)
    ax3 = (avg_fuel_mix.loc[:, ['nuclear', 'wind', 'coal', 'ccgt']] * 2).plot.bar(stacked = True) 
    ax3.set(xlabel = 'Period',
            xticks = list(range(0, 47, 2)),
            xticklabels = list(range(0, 47, 2)),
            ylabel = 'Generation (MW)', 
            title = 'Average Dispatch over Final 50 Days');
    fig3 = ax3.get_figure()
    
    # the true dispatch for the given date in params
    
    fig4 = plt.figure(4)
    ax4 = (true_dispatch[['nuclear', 'wind', 'coal', 'ccgt']] * 2).plot.bar(stacked = True) 
    ax4.set(xlabel = 'Period',
           xticks = list(range(0, 47, 2)),
           xticklabels = list(range(0, 47, 2)),
           ylabel = 'Generation (MW)', 
           title = 'True Dispatch for {0}'.format(params['date'][0]));
    fig4 = ax4.get_figure()
    
    # average successful offer price for each fuel type over final 50 days
    
    fig5 = plt.figure(5)
    ax5 = avg_offer_price.loc[:, ['nuclear', 'wind', 'coal', 'ccgt']].plot()
    ax5.set(xlabel = 'Period',
            xticks = list(range(0, 47, 2)),
            xticklabels = list(range(0, 47, 2)),
            ylabel = 'Offer Price (£/MWh)', 
            title = 'Average Successful Offer Price over Final 50 Days');
    fig5 = ax5.get_figure()
    
    # rolling 25-day mean of successful offers per fuel type, for periods
    # 0-31 and 42-47
    
    fig6 = plt.figure(6)
    ax6 = rolling_offers_off.loc[:, ['nuclear', 'wind', 'coal', 'ccgt']].plot()
    ax6.set(xlabel = 'Iteration (Day)',
            ylabel = 'Offer Price (£/MWh)',
            title = '25-Day MA for Average Off-Peak Successful Offers')
    fig6 = ax6.get_figure()
    
    # as above, for periods 32-41
    
    fig7 = plt.figure(7)
    ax7 = rolling_offers_peak.loc[:, ['nuclear', 'wind', 'coal', 'ccgt']].plot()
    ax7.set(xlabel = 'Iteration (Day)',
            ylabel = 'Offer Price (£/MWh)',
            title = '25-Day MA for Average Peak Successful Offers')
    fig7 = ax7.get_figure()
    
    # rolling successful volume offers
    
    fig71 = plt.figure(71)
    ax71 = rolling_offers_volume.loc[:, ['nuclear', 'wind', 'coal', 'ccgt']].plot()
    ax71.set(xlabel = 'Iteration (Day)',
            ylabel = 'Offer Volume (MWh)',
            title = '5-Day MA for Average PX Volume Offers by Fuel Type')
    fig71 = ax71.get_figure()
    
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
        fig1.savefig('Results/{0}/PX Profits.jpg'.format(name), dpi = 200)
        fig2.savefig('Results/{0}/Final Dispatch.jpg'.format(name), dpi = 200)
        fig3.savefig('Results/{0}/Average Dispatch.jpg'.format(name), dpi = 200)
        fig4.savefig('Results/{0}/True Dispatch.jpg'.format(name), dpi = 200)
        fig5.savefig('Results/{0}/Final Offer Prices.jpg'.format(name), dpi = 200)
        fig6.savefig('Results/{0}/Off-Peak Rolling Offers.jpg'.format(name), dpi = 200)
        fig7.savefig('Results/{0}/Peak Rolling Offers.jpg'.format(name), dpi = 200)
        fig71.savefig('Results/{0}/Rolling Volume Offers.jpg'.format(name), dpi = 200)
        fig8.savefig('Results/{0}/Average Proportional Dispatch.jpg'.format(name), dpi = 200)
        fig9.savefig('Results/{0}/True Proportional Dispatch.jpg'.format(name), dpi = 200)
        fig10.savefig('Results/{0}/Proportional Dispatch Error.jpg'.format(name), dpi = 200)
        fig11.savefig('Results/{0}/Absolute Dispatch Error.jpg'.format(name), dpi = 200)
        
        params['name'] = name
        params = orderHeaders(params)
        del params['true_wind']
        del params['demand_profile']
        del params['propensities']
        with open('Results/Graph Parameters.csv', 'a+', newline = '') as file:
            w = csv.DictWriter(file, params.keys())
            
            if getsize('Results/Graph Parameters.csv') == 0:
                w.writeheader()    
            else: 
                csv.writer(file).writerow([])

            w.writerow(params)
        


def bmGraphs(results, periods, save_graphs):
    
    avg_bm_marginal_price = results['avg_bm_marginal_price']
    bm_price = results['bm_marginal_price']
    imbal = list(results['imbalance'])
    bm_day_profit = results['bm_day_profit']
    days = len(results['generation'])
    num_agents = len(results['gen_labels'].iloc[-1])
        
    fig1, ax12 = plt.subplots()
    ax13 = ax12.twinx()
    avg_bm_marginal_price['Period'] = np.linspace(0, 47, 48)
    direction = np.array(avg_bm_marginal_price['Direction'])
    color = np.where(direction == 0, 'r', 'b')
    red = mpatches.Patch(color = 'red', label = 'Short')
    blue = mpatches.Patch(color = 'blue', label = 'Long')
    ax12 = avg_bm_marginal_price[['Price']].plot(ax = ax12) 
    ax12 = avg_bm_marginal_price.plot.scatter(x = 'Period', y = 'Price', c = color, ax = ax12)   
    pd.Series([np.mean([imbal[-day][period] for day in range(1, 51)]) for period in range(48)]).plot(ax = ax13,
                                                                                                     secondary_y = True,
                                                                                                     style = '-',
                                                                                                     color = 'orange')
    ax13.right_ax.set(ylabel = '50-Day Average Imbalance at Gate Closure (MWh)')
    ax12.set(xlabel = 'Period',
             xticks = list(range(0, 47, 2)),
             xticklabels = list(range(0, 47, 2)),
             ylabel = 'Price (£/MWh)',
             title = 'Average Balancing Mechanism Marginal Price over Final 50 Days');
    ax12.legend(handles = [red, blue])
    
    
    for period in periods:
        
        if bm_price[days - 1][period][0] == 1:
            
            fig = plt.figure()
            ssp = [bm_price[day][period][1] for day in range(days)]
            plt.title('25-Day MA SSP for Period {0}'.format(period))
            plt.xlabel('Iteration (Day)')
            plt.ylabel('Price (£/MWh)')
            plt.plot(movingAverage(ssp, 25))
            
            if save_graphs == True:
        
                fig.savefig('Results/{0}/SSP for Period {1}.jpg'.format(results['name'], period), dpi = 200)
        
        else:
            
            fig = plt.figure()
            sbp = [bm_price[day][period][1] for day in range(days)]
            plt.title('25-Day MA SBP for Period {0}'.format(period))
            plt.xlabel('Iteration (Day)')
            plt.ylabel('Price (£/MWh)')
            plt.plot(movingAverage(sbp, 25))
            
            if save_graphs == True:
        
                fig.savefig('Results/{0}/SBP for Period {1}.jpg'.format(results['name'], period), dpi = 200)

    
    fig2 = plt.figure()
    ax1 = plt.subplot()
    for i in range(2,num_agents):
        ax1.plot(range(days - 50), movingAverage(bm_day_profit[i::num_agents], 50),
                color = results['gen_labels'][days-1][i][2],
                label = [results['gen_labels'][days-1][i][0], i])
    ax1.set(title = '50-Day MA BM Profits',
            xlabel = 'Iteration (Day)',
            ylabel = 'Profit (£)')
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    fig2 = ax1.get_figure()
    
    if save_graphs == True:
        
        fig1.savefig('Results/{0}/BM Marginal Price.jpg'.format(results['name']), dpi = 200)
        fig2.savefig('Results/{0}/BM Profits.jpg'.format(results['name']), dpi = 200)
   


def pxStrategyEvolution(results, agent_id, step_size):
    """ Creates a series of bar charts illustrating the evolution of the px
    
    offer with the highest propensity for a given agent every step_size days,
    saving them to a directory ready to be made into a gif. Also the evolution
    of the bm intercept and gradient, if requested.
    """
    
    num_agents = len(results['gen_labels'].iloc[-1])
    days = len(results['generation'])
    fuel = results['gen_labels'].iloc[-1][agent_id][0]
    name = results['name']
    
    px_props = results['px_price_props']
    
    px_props_list = px_props[((step_size - 1) * num_agents + agent_id):((days + step_size - 1) * num_agents + agent_id):(step_size * num_agents)]
    
    px_max_props = pd.DataFrame.from_records(px_props_list)
    
    # converts max_prop argument to price offer. Remember to change with offer_set
    px_max_props_df = px_max_props.applymap(lambda x: x.argmax()).transpose()
    px_max_props_df = px_max_props_df.applymap(lambda x: 25 + 5*x)
    
    makedirs('Strategy Evolutions/{0}/Agent {1} - {2}'.format(name, agent_id, fuel))
    
    for day in range(int(days/step_size)):
        fig = plt.figure(day);
        ax = px_max_props_df[day].plot.bar();
        ax.set(xlabel = 'Period',
               xticks = list(range(0, 47, 2)),
               xticklabels = list(range(0, 47, 2)),
               yticks = list(np.linspace(5, 125, 13)),
               yticklabels = list(np.linspace(5, 125, 13)),
               ylabel = 'PX Price Offer', 
               title = 'Evolution of Preferred PX Offer for Agent {0} - {1}, Day {2}'.format(agent_id, fuel, day*step_size));
        fig = ax.get_figure();
        fig.savefig('Strategy Evolutions/{0}/Agent {1} - {2}/{3}.jpg'.format(name, agent_id, fuel, day), dpi = 200)
        plt.close()
        pass;



def bmLadderEvolution(results, agent_id, period, verbose, save_graphs):
    """ Accepts results, an agent, and a period; returns the evolution of their
    
    bm intercept and gradient choices, as well as an averaged bid ladder over 
    the final 100 days and a real bid/offer ladder for a given day
    """  
        
    # plots a real bid/offer ladder for a given day, including BM volumes
    
    bm_offers = results['bm_offers']
    bm_bids = results['bm_bids']
    gen_labels = results['gen_labels']
    day = len(results['generation']) - 1
    
    fig1 = plt.figure()
    
    bm_offers = list(zip(np.cumsum([bm_offers[day, agent_id][period][i][1] for i in range(5)]), 
                        [bm_offers[day, agent_id][period][i][2] for i in range(5)]))
    
    bm_bids = list(zip(np.cumsum([bm_bids[day, agent_id][period][i][1] for i in range(5)]), 
                        [bm_bids[day, agent_id][period][i][2] for i in range(5)]))
    
    if (bm_bids[0][0] == 0) and (bm_offers[4][0] == 0):
        
        print('Generator not Dispatched in PX for specified Period')
        
    else:
    
        bm_ladder = [bm_bids[-i] for i in range(1, 6)] + [bm_offers[i] for i in range(5)]
        
        plt.plot([bm_ladder[i][0] for i in range(10)],
                 [bm_ladder[i][1] for i in range(10)])
        plt.scatter([bm_ladder[i][0] for i in range(5, 10)],
                    [bm_ladder[i][1] for i in range(5, 10)], color = 'r')
        plt.scatter([bm_ladder[i][0] for i in range(0, 5)],
                    [bm_ladder[i][1] for i in range(0, 5)], color = 'b')
        
        plt.title('BM Bid/Offer Ladder for Agent {0} ({1}), Day {2}, Period {3}'.format(agent_id, gen_labels[0][agent_id][0], day, period))
        plt.xlabel('Bid/Offer Volume (MWh)')
        plt.ylabel('Bid/Offer Price (£/MWh)')
        plt.axvline(x = 0, color = 'r')
        plt.grid()
        
        if save_graphs == True:
            
            fig1.savefig('Results/{0}/Full BM Ladder for Agent {1} ({2}), Day {3}, Period {4}.jpg'.format(results['name'],
                                                                                                     agent_id,
                                                                                                     gen_labels[0][agent_id][0],
                                                                                                     day,
                                                                                                     period), dpi = 200)
    
    if verbose == True:
        
        days = len(results['generation'])
        intercept_choices = [results['bm_intercept_choices'][day][agent_id][period] for day in range(days)]
        gradient_choices = [results['bm_gradient_choices'][day][agent_id][period] for day in range(days)]
        
        # for a given agent and period, tracks their gradient and intercept choices
        # through the simulation
        
        rolling_intercept_choices = movingAverage(intercept_choices, 25)
        fig2 = plt.figure()
        plt.plot(rolling_intercept_choices)
        plt.title('25-Day MA of BM Bid/Offer Ladder Intercept Choices for Agent {0}, Period {1}'.format(agent_id, period))
        plt.xlabel('Iteration (Day)')
        plt.ylabel('Price (£/MWh)')
        plt.ylim([0, 100])
        
        rolling_gradient_choices = movingAverage(gradient_choices, 25)
        fig3 = plt.figure()
        plt.plot(rolling_gradient_choices)
        plt.title('25-Day MA of BM Bid/Offer Ladder Gradient Choices for Agent {0}, Period {1}'.format(agent_id, period))
        plt.xlabel('Iteration (Day)')
        plt.ylabel('Price Gradient (£/MWh^2)')
        plt.ylim([0, 10])
        
        # in order to give some sort of converged ladder for a given period, average
        # out the intercept and gradient choices for the last 25 days and construct
        # it from that
        
        average_intercept = np.mean(np.array([intercept_choices[-day] for day in range(25)]))
        average_gradient = np.mean(np.array([gradient_choices[-day] for day in range(25)]))
        
        bid_offer_ladder = [average_intercept + i * average_gradient for i in range(-5, 6)]
        
        fig4, ax = plt.subplots()
        plt.scatter(y = bid_offer_ladder, x = ['min_gen', '-50%', '-25%', '-10%', '-5%', 'PX Dispatch',
                                               '+5%', '+10%', '+25%', '+50%', 'max_gen'])
        plt.xticks(rotation = 45)
        plt.ylabel('Offer Price (£/MWh)')
        ax.set_axisbelow(True)
        plt.title('Average BM Bid/Offer Ladder for Agent {0}, Period {1}'.format(agent_id, period))
        ax.grid()
        
        if save_graphs == True:
            
            fig2.savefig('Results/{0}/Rolling BM Intercept Choices for Agent {1} ({2}), Period {3}.jpg'.format(results['name'],
                                                                                                               agent_id,
                                                                                                               gen_labels[0][agent_id][0],
                                                                                                               period), dpi = 200)
            fig3.savefig('Results/{0}/Rolling BM Gradient Choices for Agent {1} ({2}), Period {3}.jpg'.format(results['name'],
                                                                                                              agent_id,
                                                                                                              gen_labels[0][agent_id][0],
                                                                                                              period), dpi = 200)
            fig4.savefig('Results/{0}/Average BM Ladder for Agent {1} ({2}), Period {3}.jpg'.format(results['name'],
                                                                                                    agent_id,
                                                                                                    gen_labels[0][agent_id][0],
                                                                                                    period), dpi = 200) 



def individualOffers(results, agent_ids, periods, one_gen_per_graph, ma, save_graphs):
    """ Accepts the results of a simulation run, a list of agent_ids, and a list
    
    of periods over which to compare the evolution of the price and volume offers, 
    over a 50-day moving average. one_gen_per_graph is a Bool that specifies 
    whether to plot one agent over the list of periods per graph, or one period 
    for all agents per graph.
    """
    
    generation = results['generation']
    px_offers = results['px_offers']
    gen_labels = results['gen_labels'][len(generation) - 1]
    
    if one_gen_per_graph == True:
        
        for agent_id in agent_ids:
            
            fig1 = plt.figure()
            
            single_gen = {}
            
            for day in range(len(generation)):
                single_gen[day] = [[generation[day][period][agent_id][1], 
                                   generation[day][period][agent_id][2], 
                                   px_offers[day, agent_id][period][0]/(gen_labels[agent_id][1]/2)] for period in periods]
            
            rolling_offers = {}
            
            for i, period in enumerate(periods):
                
                rolling_offers[period] = movingAverage([single_gen[day][i][1] for day in range(len(generation))], ma)
                plt.plot(rolling_offers[period])
        
            plt.legend(labels = periods)
            plt.title('{3}-Day MA PX Price Offers for Generator {0} ({1}) Across Periods {2}'.format(agent_id, gen_labels[agent_id][0], periods, ma))
            plt.xlabel('Iteration (Day)')
            plt.ylabel('Price (£/MWh)')
            plt.ylim([25, 125])
            
            
            fig2 = plt.figure()
            
            rolling_volumes = {}
            
            for i, period in enumerate(periods):
                
                rolling_volumes[period] = movingAverage([single_gen[day][i][2] for day in range(len(generation))], ma)
                plt.plot(rolling_volumes[period])
            
            plt.legend(labels = periods)
            plt.title('{0}-Day MA PX Volume Offers for Generator {1} ({2}) Across Periods {3}'.format(ma, agent_id, gen_labels[agent_id][0], periods))
            plt.xlabel('Iteration (Day)')
            plt.ylabel('Volume as Proportion of Capacity')
            plt.ylim([0.5, 1])
            
            if save_graphs == True:
                
                fig1.savefig('Results/{0}/PX Price Offers for Generator {0} ({1}) Across Periods {2}.jpg'.format(results['name'],
                                                                                                                 agent_id,
                                                                                                                 gen_labels[0][agent_id][0],
                                                                                                                 periods), dpi = 200)
                fig2.savefig('Results/{0}/PX Volume Offers for Generator {0} ({1}) Across Periods {2}.jpg'.format(results['name'],
                                                                                                                  agent_id,
                                                                                                                  gen_labels[0][agent_id][0],
                                                                                                                  periods), dpi = 200)
            
    else:
        
        for period in periods:
            
            fig1 = plt.figure()
            
            single_period = {}
            
            for day in range(len(generation)):
                single_period[day] = [[generation[day][period][agent_id][1], 
                                      generation[day][period][agent_id][2],
                                      px_offers[day, agent_id][period][0]/(gen_labels[agent_id][1]/2)] for agent_id in agent_ids]
            
            rolling_offers = {}
            
            for i, agent_id in enumerate(agent_ids):
                
                rolling_offers[agent_id] = movingAverage([single_period[day][i][1] for day in range(len(generation))], ma)
                plt.plot(rolling_offers[agent_id])
            
            plt.legend(labels = [[agent_id, gen_labels[agent_id][0]] for agent_id in agent_ids])
            plt.title('{2}-Day MA PX Price Offers for Period {0} For Agents {1}'.format(period, agent_ids, ma))
            plt.xlabel('Iteration (Day)')
            plt.ylabel('Price (£/MWh)')
            plt.ylim([25, 125])
            
            
            fig2 = plt.figure()
            
            rolling_volumes = {}
            
            for i, agent_id in enumerate(agent_ids):
                
                rolling_volumes[agent_id] = movingAverage([single_period[day][i][2] for day in range(len(generation))], ma)
                plt.plot(rolling_volumes[agent_id])
            
            plt.legend(labels = [[agent_id, gen_labels[agent_id][0]] for agent_id in agent_ids])
            plt.title('{2}-Day MA PX Volume Offers for Period {0} For Agents {1}'.format(period, agent_ids, ma))
            plt.xlabel('Iteration (Day)')
            plt.ylabel('Volume as Proportion of Capacity')
            plt.ylim([0.5, 1])
            
            if save_graphs == True:
                
                fig1.savefig('Results/{0}/PX Price Offers for Period {1} For Agents {2}.jpg'.format(results['name'],
                                                                                                    period,
                                                                                                    agent_ids), dpi = 200)
                fig2.savefig('Results/{0}/PX Volume Offers for Period {1} For Agents {2}.jpg'.format(results['name'],
                                                                                                     period,
                                                                                                     agent_ids), dpi = 200)



def systemPrices(results, cost_graphs, periods, save_graphs):
    """ Generates graphs of the system cost and price in various forms.
    
    """
    
    system_costs = results['system_costs']
    demand = results['demand']
    px_day_profit = pd.DataFrame(results['px_day_profit']).unstack().apply(lambda x: sum(x), axis = 1)
    px_marginal_price = results['px_marginal_price']
    avg_rolling_offers = results['avg_rolling_offers']
    
    daily_system_costs = [sum(period_costs) for period_costs in system_costs]
    
    system_costs_MWh = [[system_costs[day][period]/demand[period] for period in range(48)] 
                              for day in range(len(system_costs))]
    
    daily_avg_system_costs_MWh = [sum(period_costs)/48 for period_costs in system_costs_MWh]
    
    
    avg_system_period_cost = [np.mean(np.array([system_costs.iloc[-day][period] for day in range(100)])) 
                              for period in range(48)]
    
    avg_system_period_cost_MWh = np.array(avg_system_period_cost/demand)
    
    avg_marginal_price = []
    for period in range(48):
        
        avg_marginal_price.append(np.mean([px_marginal_price.iloc[-day][period] for day in range(1, 26)]))
     
    
    if cost_graphs == True:
        
        # plots 25-day MA of total daily system costs
        fig1 = plt.figure()
        plt.plot(movingAverage(daily_system_costs, 25))
        plt.title('25-Day MA of Total Daily System Cost')
        plt.xlabel('Iteration (Day)')
        plt.ylabel('Cost (£)')
        
        # plots 25-day MA of the average daily system cost per MWh
        fig2 = plt.figure()
        plt.plot(movingAverage(daily_avg_system_costs_MWh, 25))
        plt.title('25-Day MA of Average Daily System Cost per MWh')
        plt.xlabel('Iteration (Day)')
        plt.ylabel('Cost (£/MWh)')
        
        # plots the average system cost per period over the last 100 days of the sim
        fig3 = plt.figure()
        plt.plot(avg_system_period_cost)
        plt.title('System Cost per Period Averaged over Final 100 Iterations')
        plt.xlabel('Period')
        plt.ylabel('Cost (£)')
        
        # the above, but normalised per MWh
        fig4 = plt.figure()
        plt.plot(avg_system_period_cost_MWh)
        plt.title('System Cost per MWh per Period Averaged over Final 100 Iterations')
        plt.xlabel('Period')
        plt.ylabel('Cost (£/MWh)')
        
        if save_graphs == True:
            
            fig1.savefig('Results/{0}/Total Daily System Cost.jpg.'.format(results['name']), dpi = 200)
            fig2.savefig('Results/{0}/Daily System Cost per MWh.jpg'.format(results['name']), dpi = 200)
            fig3.savefig('Results/{0}/Average Total Period System Cost.jpg'.format(results['name']), dpi = 200)
            fig4.savefig('Results/{0}/Average Period System Cost per MWh.jpg'.format(results['name']), dpi = 200)
    
    # total system profit per day, and averaged per-period over last N days
    fig5 = plt.figure()
    plt.plot(px_day_profit)
    plt.title('Total PX Profit per Day')
    plt.xlabel('Iteration (Day)')
    plt.ylabel('Profit (£)')
    
    # marginal px price averaged over the final 25 days
    fig6 = plt.figure()
    plt.plot(avg_marginal_price)
    plt.title('PX Marginal Price Averaged Over Final 25 Days')
    plt.xlabel('Period')
    plt.ylabel('Price (£)')
    
    # plots the evolution of the average accepted offer price for certain periods
    for period in periods:
        
        fig = plt.figure()
        plt.plot(movingAverage(avg_rolling_offers[period], 10))
        plt.title('PX Average Successful Offer Price for Period {0}'.format(period))
        plt.xlabel('Iteration (Day)')
        plt.ylabel('Price (£)')
        
        if save_graphs == True:
            
            fig.savefig('Results/{0}/PX Average Successful Offer Price for Period {1}'.format(results['name'], period), dpi = 200)
    
    
    if save_graphs == True:
        
        fig5.savefig('Results/{0}/Total PX Profit per Day.jpg'.format(results['name']), dpi = 200)
        fig6.savefig('Results/{0}/PX Marginal Price Averaged Over Final 25 Days.jpg'.format(results['name']), dpi = 200)



def rawDispatch(results, days, save_graphs):
    """ Generates five plots of each of the last five days' raw dispatch schedules
    
    by fuel type, not ordering by offer price or differentiating by agent.
    """
    
    generation = results['generation']
    gen_labels = results['gen_labels']
    demand = results['demand']
        
    for day in range(1, days + 1):
        
        all_gen = pd.DataFrame({agent_id: np.zeros(48) for agent_id in gen_labels.iloc[-1].keys()})
    
        for period in range(48):
                        
            for agent_id in gen_labels.iloc[-1].keys():
                
                all_gen.iloc[period][agent_id] = generation.iloc[-day][period][agent_id][5]*2
            
        all_gen.columns = all_gen.columns.to_series().apply(lambda gen: gen_labels.iloc[-1][gen][0])
        all_gen = all_gen.groupby(all_gen.columns, axis = 1).sum()
        
        fig = plt.figure()
        ax = all_gen[['nuclear', 'wind', 'coal', 'ccgt']].plot.bar(stacked = True)
        demand.plot(linewidth = 2)
        ax.set(xlabel = 'Period',
               xticks = list(range(0, 47, 2)),
               xticklabels = list(range(0, 47, 2)),
               ylabel = 'Generation (MW)', 
               title = 'Dispatch Schedule of Day {0} by Fuel Type'.format(len(generation) - day))
        
        if save_graphs == True:
            
            fig.savefig('Results/{0}/Dispatch for Day {1}'.format(results['name'], len(generation) - day), dpi = 200)    
        
        
def emissions(results, periods, save_graphs):
    """ Plots the average emissions per period over the last 25 days, as well
    
    as the evolution of emissions for the whole day and for select periods.
    """
    
    emissions = results['emissions']
    
    avg_emissions = []
    for period in range(48):
        
        avg_emissions.append(np.mean([emissions.iloc[-day][period] for day in range(1, 26)]))
        
    fig1 = plt.figure()
    plt.plot(movingAverage([sum(emissions[day]) for day in range(len(emissions))], 25))
    plt.title('25-Day MA Daily Total CO2 Emissions')
    plt.xlabel('Iteration (Day)')
    plt.ylabel('CO2 Emissions (Tonnes)')
    
    fig2 = plt.figure()
    plt.plot(avg_emissions)
    plt.title('Average CO2 Emissions per Period Over Final 25 Days')
    plt.xlabel('Period')
    plt.ylabel('CO2 Emissions (Tonnes)')
        
    for period in periods:
        
        fig = plt.figure()
        plt.plot(movingAverage([emissions[day][period] for day in range(len(emissions))], 25))
        plt.title('25-Day MA Total CO2 Emissions for Period {0}'.format(period))
        plt.xlabel('Iteration (Day)')
        plt.ylabel('CO2 Emissions (Tonnes)')
        
        if save_graphs == True:
            
            fig.savefig('Results/{0}/Total CO2 Emissions for Period {1}'.format(results['name'], period), dpi = 200)
    
    if save_graphs == True:
        
            fig1.savefig('Results/{0}/Daily Total CO2 Emissions'.format(results['name']), dpi = 200)        
            fig2.savefig('Results/{0}/Average CO2 Emissions per Period Over Final 25 Days'.format(results['name']), dpi = 200)        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
        