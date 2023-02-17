import numpy as np
import copy, random  

import DataModule as dm
import learning_test3 as lt3 
import VirtualMarketModule as vm 
import VisualizeModule as vis 

action_threshold = .75

def feedback(pred, current_price):
    market.price_log(current_price) 
    if len(market.price_history) < 2:
        return np.array([0, 0]) 
    

    #print(market.price_history)
    if np.max(pred) == pred[0] and pred[0] >= action_threshold:
        market.buy() 
        market.stat()

    elif np.max(pred) == pred[1] and pred[1] >= action_threshold:
        market.sell()
        market.stat() 

    # else:
    #     price_final = market.price_history[-1]
    #     price_initial = market.price_history[-2] 
    #     return market.assets * (price_final-price_initial)/price_initial

    price_final = market.price_history[-1]
    price_initial = market.price_history[-2] 
    diff = price_final-price_initial
   
    if diff > 5:
        return np.array([1, 0])
    elif diff < -5:
        return np.array([0, 1])
    else:
        return np.array([0, 0]) 
    return market.assets * (price_final-price_initial)/price_initial

    

def opportunity_cognizance(last_price, current_price):
    return current_price - last_price   #simple

database = dm.DataBase('BTCUSD', "tick_data_BTCUSD.txt")

market = vm.Market() 

trainer = lt3.Trainer(250, 2, feedback_function=feedback) 

desired_keys =  ["highPrice", "lowPrice", "priceChangePercent", "volume", 'lastPrice']

#create initial training set
database.Load() 
#database.create_training_set() 
#database.create_validation_set()

#train
tick_data = copy.deepcopy(database.tick_data)
DEPTH = 50 
EPOCH_SIZE = 100 
DATA_SIZE = len(tick_data) 
epochs = int(DATA_SIZE / EPOCH_SIZE) 

epoch_data = {}


for i in range(epochs):
    data = {}
    for j in range(i*EPOCH_SIZE, (i+1)*EPOCH_SIZE):
        data[j] = tick_data[j+1] 
    epoch_data[i] = data 


samples = [] 
training_prices = []
data = [[database.tick_data[i+1] for i in range(begin, begin+DEPTH)] for begin in range(DATA_SIZE-DEPTH)] 

for d in data:
    #result = [database.strip(j, desired_keys) for j in data]#database.strip(d, desired_keys)
    price = database.raw_strip(d, ['lastPrice']) 
    result = database.strip(d, desired_keys) 
    training_prices.append(price[-1]) 
    samples.append(result.flatten()) 


validation_prices = []
database.load_validation_data() 

validation_prices = database.strip(database.validation_data, ['lastPrice']) 



#market.price_history = training_prices
trainer.current_state_list = training_prices

#randomize samples
random.shuffle(samples)

trainer.set_sample_set(samples[0:20])

for i in range(1):
    trainer.reinforcement_train(8, iter=8) 



decisions = trainer.run_validation_set(validation_prices) 
#print(market.decisions)

#convert trainer decisions to 1, -1, 0 sequence
decisions_to_plot = []
for d in decisions: 
    if max(d)==d[0] and d[0] >= action_threshold:
        decisions_to_plot.append(1)
    elif max(d)==d[1] and d[1] >= action_threshold:
        decisions_to_plot.append(-1)
    else:
        decisions_to_plot.append(0) 

vis.decision_graph(validation_prices, decisions_to_plot) 

print('assets: ', market.assets) 
print('done') 


        
        




