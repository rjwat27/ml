import numpy as np 


from DataModule import DataBase as dm 

import fanout_layer as fl 


#ui.no_growth_test(30, layers=1, fanout=0) 
#ui.legacy_no_growth_test(30, fanout=0) 

database = dm('BTCUSD', "tick_data_BTCUSD.txt")


database.Load() 

samples, prices = database.list_of_samples(len_of_samples=20, tick_epoch=2) 


#make answer set 

answers = []

for i in range(len(samples)-1):
    answers.append(np.average(prices[i+1])-np.average(prices[i])) 

samples.pop(0) #reduce sample set size to match answer set 

net = fl.fanout_network(len(samples[0]), 1, 4, [len(samples[0]), 3, 3, 3, 1], 
                        fanout=0, growth=True) 

#train 

t = 0

e = 20

while e > .5 and t < 100:

    e, iter = fl.run_learn_cycle(net, samples, answers, 1, random=True, num_iter=10)
    print('epoch: ', t) 
    print('iter: ', iter) 
    print('layer sizes: %d %d %d', len(net.hidden_layers[0].biases1), len(net.hidden_layers[1].biases1), len(net.hidden_layers[2].biases1))

    t += 1


weights = np.array([net.hidden_layers[n].w1 for n in range(net.layers)])
biases = np.array([net.hidden_layers[n].biases1 for n in range(net.layers)])

np.save('net_weights', allow_pickle=True) 
np.save('net_biases', allow_pickle=True) 

print("Done") 


#test network on test data for sanity

# predicted_prices = [prices[0]]
# prices.pop(0) 

# for i in range(len(samples)):
#     predicted_prices.append(net.activate(samples[i])) 

# from matplotlib import pyplot as plt

# x = range(len(prices))

# plt.plot(x, prices, 'blue') 

# plt.plot(x, predicted_prices, color='green') 







