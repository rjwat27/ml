import numpy as np 

from matplotlib import pyplot as plt

from DataModule import DataBase as dm 

import mlp 

similarity_threshold = .9

desired_keys =  ["highPrice", "lowPrice", "priceChangePercent", "volume", 'lastPrice']


#ui.no_growth_test(30, layers=1, fanout=0) 
#ui.legacy_no_growth_test(30, fanout=0) 

database = dm('BTCUSD', "../tick data/tick_data_BTCUSD.txt")


database.Load(file_dir='../tick data') 

sample_width = 50
foresight = 20

samples, prices = database.list_of_samples(len_of_samples=sample_width, tick_epoch=2) 

data = samples 
max = np.max(data)
min = np.min(data) 

#normalize samples globally
for i in range(len(desired_keys)):
    # max = np.max([j[i] for j in data]) 
    # min = np.min([j[i] for j in data]) 
    if max==min and max > 0:
        data[:,i] = .5*np.ones(len(data)) 
    elif max==min and max < 0:
        data[:,i] = -.5*np.ones(len(data)) 
    else:
        data[:,i] = (np.array(data[:,i]) - min*np.ones(len(data))) * (2/(max - min)) - np.ones(len(data)) #test

samples = data 

test_samples, test_prices = database.list_of_samples(len_of_samples=sample_width, tick_epoch=2) 
# test_samples = test_samples[0:50]
# test_prices = test_prices[0:50] 



#make answer set 

answers = []

for i in range(len(samples)-1):
    answers.append(np.average(prices[i+1])-np.average(prices[i])) 

samples = samples[0:len(answers)] #reduce sample set size to match answer set 

#clean
std = np.std(answers)
mean = np.mean(answers)
new_a = []
new_s = []
for i in range(len(answers)):
    if abs(answers[i]-mean) > 2*std:
        new_a.append(answers[i])
        new_s.append(samples[i]) 

samples = new_s
answers = new_a 


for i in range(len(samples)):
    s = samples[i]
    n = np.array(s) + 2*i*np.ones(len(s))
    plt.plot(range(len(s)), s) 
plt.show()
plt.clf()

print(len(samples))

net = mlp.mlp(len(samples[0]), 1, 2, [len(samples[0]), 20, 1], 
                        fanout=0, growth=True, learning_rate=.1) 



#train 

t = 0

e = 20

mlp.run_learn_cycle(net, samples, answers, 1, cohort=360, random=True, num_iter=100)

#test net
predictions = []
start = test_prices[0]
predictions = [start]
for s in test_samples:
    p = net.activate(s)
    predictions.append(predictions[-1]+p) 

x = range(len(test_prices))
y = range(len(predictions))
plt.plot(x, np.average(test_prices, axis=0))
plt.plot(y, predictions, color='green') 
plt.show()

# input(len(samples))

# while e > .5 and t < 100:

#     e, iter = mlp.run_learn_cycle(net, samples, answers, 1, cohort=20, random=True, num_iter=10)
#     print('epoch: ', t) 
#     print('iter: ', iter) 
#     print('layer sizes: %d %d %d', len(net.hidden_layers[0].biases1), len(net.hidden_layers[1].biases1), len(net.hidden_layers[2].biases1))

#     t += 1


# weights = np.array([net.hidden_layers[n].w1 for n in range(net.layers)])
# biases = np.array([net.hidden_layers[n].biases1 for n in range(net.layers)])

# np.save('net_weights', allow_pickle=True) 
# np.save('net_biases', allow_pickle=True) 

print("Done") 


#test network on test data for sanity

# predicted_prices = [prices[0]]
# prices.pop(0) 

# for i in range(len(samples)):
#     predicted_prices.append(net.activate(samples[i])) 



# x = range(len(prices))

# plt.plot(x, prices, 'blue') 

# plt.plot(x, predicted_prices, color='green') 







