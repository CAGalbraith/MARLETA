import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime

# to deal with the difference in mac and pc of path referencing, need to 
# use pathlib for this one...

from pathlib import Path

# it appears that the generation overshoot is caused by the embedded solar, implying it's already been included in the transmission demand
# calculation... but embedded wind hasn't? 
# can we turn this into a function, that takes two date ranges and outputs the spliced datasets, aggregate subsets, and graphs?

# the following function loads csv files locally, and expects dates formatted like 2013-04-15, includes start but not end.
# also set working directory to folder containing the data subsets folder so path references work as they should

def genAggregator(start, end, graphs):
    
    days = int((datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.datetime.strptime(start, '%Y-%m-%d'))/datetime.timedelta(days = 1))
    
    data_folder = Path('Data')
    
    generation = pd.read_csv(data_folder / r'Total Generation By Type Half-Hourly 2013-2016.csv')
    demand = pd.read_csv(data_folder / r'Clean National Demand 2013-2016.csv')
    day_ahead_prices = pd.read_csv(data_folder / r'N2EX Day-Ahead Hourly Prices.csv', encoding = "ISO-8859-1")
    elexon_prices = pd.read_csv(data_folder / r'Elexon SBP-SSP 2013-2016.csv')
    marginal_costs = pd.read_csv(data_folder / r'Generator Marginal Costs.csv')
    
    generation = generation[(generation['Date'] >= start) & (generation['Date'] < end)]
    demand = demand[(demand['Date'] >= start) & (demand['Date'] < end)]
    day_ahead_prices = day_ahead_prices[(day_ahead_prices['Date'] >= start) & (day_ahead_prices['Date'] < end)]
    elexon_prices = elexon_prices[(elexon_prices['Date'] >= start) & (elexon_prices['Date'] < end)]
    marginal_costs = marginal_costs[(marginal_costs['year'] == pd.Timestamp(start).year) & (marginal_costs['quarter'] == pd.Timestamp(start).quarter)]
    
    ccgt = generation[generation['Type'] == 'ccgt']['Generation'].reset_index(drop = True)
    coal = generation[generation['Type'] == 'coal']['Generation'].reset_index(drop = True)
    hydro = generation[generation['Type'] == 'hydro']['Generation'].reset_index(drop = True)
    nuclear = generation[generation['Type'] == 'nuclear']['Generation'].reset_index(drop = True)
    wind = generation[generation['Type'] == 'wind']['Generation'].reset_index(drop = True)
    
    dispatchable = pd.concat([ccgt, coal, hydro, nuclear], axis = 1)
    dispatchable.columns = ['CCGT', 'Coal', 'Hydro', 'Nuclear']
    dispatchable['Aggregate'] = dispatchable.sum(axis = 1)
    
    # this adjusts for the fact that exports are factored into TD already
    interconnectors = demand[['France Int.', 'BritNed Int.', 'Moyle Int.', 'East-West Int.']].reset_index(drop = True)
    interconnectors = interconnectors.applymap(lambda x: max(0, x)) 
    interconnectors = pd.DataFrame(interconnectors.sum(axis = 1), columns = ['Interconnectors']) 
    
    emb_solar = demand['Emb. Solar'].reset_index(drop = True)
    emb_wind = demand['Emb. Wind'].reset_index(drop = True)
    
    intermittent = pd.concat([emb_wind, wind, emb_solar], axis = 1)
    intermittent.columns = ['Embedded Wind', 'Wind', 'Embedded Solar']
    intermittent['Aggregate'] = intermittent.sum(axis = 1).reset_index(drop = True)
    
    gen_agg = pd.concat([ccgt, coal, wind, nuclear, hydro, interconnectors, emb_wind, emb_solar], axis = 1)
    gen_agg.columns = ['CCGT', 'Coal', 'Wind', 'Nuclear', 'Hydro', 'Interconnectors', 'Embedded Wind', 'Embedded Solar']
    gen_agg['Total Generation'] = gen_agg.sum(axis = 1)
    gen_agg['Transmission Generation'] = gen_agg[['CCGT', 'Coal', 'Wind', 'Nuclear', 'Hydro', 'Interconnectors']].sum(axis = 1)
    
    trans_demand = demand['Transmission Demand'].reset_index(drop = True)
    nat_demand = demand['National Demand'].reset_index(drop = True)
    total_demand = trans_demand + intermittent['Embedded Solar'] + intermittent['Embedded Wind']
    demand = pd.concat([nat_demand, trans_demand, total_demand], axis = 1)
    demand.columns = ['National Demand', 'Transmission Demand', 'Total Demand']
    
    costs = pd.DataFrame()
    costs['Wind'] = (gen_agg['Wind'] + gen_agg['Embedded Wind']) * marginal_costs['wind'].iloc[0]
    costs['CCGT'] = gen_agg['CCGT'] * marginal_costs['ccgt'].iloc[0]
    costs['Coal'] = gen_agg['Coal'] * marginal_costs['coal'].iloc[0]
    costs['Nuclear'] = gen_agg['Nuclear'] * 10
    costs['Hydro'] = gen_agg['Hydro'] * marginal_costs['hydro'].iloc[0]
    costs['Solar'] = gen_agg['Embedded Solar'] * marginal_costs['solar'].iloc[0]
    costs['Total'] = costs.sum(axis = 1)
    
    daily_costs = costs.groupby(costs.index // 48 * 48).sum(axis = 1).reset_index(drop = True)
    
    prop_costs_total = costs[['Nuclear', 'Wind', 'Hydro', 'Solar', 'Coal', 'CCGT']].sum(axis = 1)
    prop_costs = costs[['Nuclear', 'Wind', 'Hydro', 'Solar', 'Coal', 'CCGT']].apply(lambda x: x/prop_costs_total).reset_index(drop = True)
    
    daily_prop_costs_total = daily_costs[['Nuclear', 'Wind', 'Hydro', 'Solar', 'Coal', 'CCGT']].sum(axis = 1).reset_index(drop = True)
    daily_prop_costs = daily_costs[['Nuclear', 'Wind', 'Hydro', 'Solar', 'Coal', 'CCGT']].apply(lambda x: x/daily_prop_costs_total).reset_index(drop = True)
    
    price = pd.concat([elexon_prices['Date'], 
                       elexon_prices['Period'], 
                       elexon_prices['SSP'], 
                       elexon_prices['SBP'], 
                       day_ahead_prices['Day-Ahead Price']], 
                       axis = 1).reset_index(drop = True)
    
    error = demand['Total Demand'] - gen_agg['Total Generation']

    if graphs:
        fig_1 = plt.figure(1)
        plt.plot(gen_agg['Total Generation']*2)
        plt.plot(demand['Total Demand']*2)
        plt.legend(['Total Generation', 'Total Demand'])
        plt.ylabel('MW')
        plt.xlabel('Settlement Period')
        plt.title('Total Generation and Total Demand')
        fig_1.show()
        
        fig_2 = plt.figure(2)
        plt.plot(error*2)
        plt.ylabel('MW')
        plt.xlabel('Settlement Period')
        plt.title('Error between Total Demand and Total Generation')
        fig_2.show()
        
        fig_3 = plt.figure(3)
        plt.plot(costs)
        plt.legend(costs.columns)
        plt.ylabel('Cost (£)')
        plt.xlabel('Settlement Period')
        plt.title('Generation Cost per Settlement Period')
        fig_3.show()
        
        fig_4 = plt.figure(4)
        plt.plot(daily_costs)
        plt.legend(daily_costs.columns)
        plt.ylabel('Cost (£)')
        plt.xlabel('Day')
        plt.title('Generation Cost per Day')
        fig_4.show()
        
        fig_5 = plt.figure(5)
        ax5 = daily_prop_costs[['Nuclear', 'Wind', 'Hydro', 'Solar', 'Coal', 'CCGT']].plot.bar(stacked = True) 
        ax5.set(xlabel = 'Day',
                xticks = list(range(0, days, days//5)),
                xticklabels = list(range(0, days, days//5)),
                ylabel = 'Proportion of Costs', 
                title = 'Proportional Generation Costs per Day');
        fig_5 = ax5.get_figure()
        fig_5.show()
        
    results = dict()
    results['Intermittent'] = intermittent
    results['Interconnectors'] = interconnectors
    results['Dispatchable'] = dispatchable
    results['Supply'] = gen_agg
    results['Demand'] = demand
    results['Price'] = price
    results['Error'] = error
    results['Costs'] = costs
    results['Daily Costs'] = daily_costs
    results['Proportional Costs'] = prop_costs
    results['Export'] = pd.concat([demand, gen_agg, costs, price, prop_costs], axis = 1)
    
    return results




