import numpy as np
import copy, joblib, pickle, time
from time import sleep 
from sklearn import svm 
from sklearn.neural_network import MLPRegressor as mlp
from torch import convolution

import DataModule as dm 

import VirtualMarketModule as vm

import VisualizeModule as vis

action_threshold = .5

delay=10 

database = dm.DataBase('BTCUSD', "tick data/tick_data_BTCUSD")

Watcher = dm.Watcher() 

market = vm.Market(max=10000) 
market.decisions = [0] 

desired_keys =  ["highPrice", "lowPrice", "priceChangePercent", "volume", 'lastPrice']

STEP = 1e-5

DEPTH = 20
FORESIGHT = 10 
OVERLAP_FACTOR = 4
EPOCH_SIZE = 100 

"""label mode: 0 means discrete labeling i.e. 1 for buy scenario and -1 for sell; 1 means price change projection as ratio of projected average price over current average price"""

LABEL_MODE = 0

#database.create_training_set()
#database.Load()  

#create initial training set
#database.Load(fname='training_data.txt')  
#database.create_training_set() 
#database.create_validation_set()
# print('Creating new data set')
# database.create_training_set(1000)
# database.Save() 
def prep(iter=DEPTH):
    print('Prepping...') 
    for i in range(iter):
        data = Watcher.full_data('BTCUSD') 
        #Market.price_log() 
        database.log(data) 
        sleep(delay)
    print('Finished Prepping') 
#prep(4000) 
database.Load() 
#database.Save() 


#train
tick_data = copy.deepcopy(database.tick_data)

DATA_SIZE = len(tick_data[1]) 

epochs = int(DATA_SIZE / EPOCH_SIZE) 

epoch_data = {}
training_prices = []

'''create training data'''


brain = svm.SVR() 
#brain = mlp(hidden_layer_sizes=1000)
#brain.fit(validation_samples, validation_answers) 

def get_tick_vectors(all_data, depth=DEPTH):
    samples = []
 
    '''this need to be streamlined instead of loading all data every recalibration'''
    for batch in all_data:
        data = all_data[batch]
        size = len(data) 
        data = [[data[i+1] for i in range(begin, begin+depth)] for begin in range(size-depth)] 
       
        for i in range(len(data)-1):
            #result = [database.strip(j, desired_keys) for j in data]#database.strip(d, desired_keys)
            d = data[i] 
            result = database.strip(d, desired_keys) 
            samples.append(result.flatten())

    return samples

def learn(sv, all_data, sample, labeling_mode = LABEL_MODE, overlap=OVERLAP_FACTOR, step=STEP, depth=DEPTH, foresight=FORESIGHT):
    '''data understood to be a dictionary of dictionaries of dictionaries'''
    samples = []
    training_prices = []
    answers = []#[0] 
    '''this need to be streamlined instead of loading all data every recalibration'''
    for batch in all_data:
        data = all_data[batch]
        size = len(data) 
        data = [[data[i+1] for i in range(begin, begin+depth)] for begin in range(size-depth)] 
        #overlap should be less than the depth

        #convolutional layer
        #weights = convolutional_layer(data, sample, 50) 
    
        #from data extract every 25 or DEPTH/2 for 50% overlap between samples
        if depth > overlap:
            data = data[0::int(depth/overlap)] 
        else:
            data = data[0::1] 
       
        for i in range(len(data)-1):
            #result = [database.strip(j, desired_keys) for j in data]#database.strip(d, desired_keys)
            d = data[i] 
            price = database.raw_strip(d, ['lastPrice']) 
            result = database.strip(d, desired_keys) 
            training_prices.append(price[-1]) 
            samples.append(result.flatten())

        #create answer set
        
        for i in range(len(data)-1):
            d_initial = data[i]
            d_final = data[i+1] 
            avg_price_initial = np.average(database.raw_strip(d_initial, ['lastPrice'])[-foresight:-1])  
            avg_price_final = np.average(database.raw_strip(d_final, ['lastPrice'])[0:foresight]) 
            training_prices.append(avg_price_final) 
            '''still have reservations about the switch here but still working fine'''
            if labeling_mode==0:
                if avg_price_final/avg_price_initial >= 1+step:
                    answers.append(-1)
                elif avg_price_final/avg_price_initial <= 1-step:
                    answers.append(1)
                else:
                    answers.append(0)
            else:
                answers.append(avg_price_final/avg_price_initial)  
    print('len samples: ', len(samples), 'len answers: ', len(answers)) 

    '''this is a test in reversing order of input data'''
    samples.reverse()
    answers.reverse() 

    '''lightweight convolution on only the sparse samples'''
    #test = get_tick_vectors(all_data=all_data) 
    #print('num of tick vectors: ', len(test)) 
    weights = convolutional_layer(samples, sample.T, 50)
    #print('num of weights: ', len(weights)) 
    # print(weights)
    # input()
     
    sv.fit(samples, answers, sample_weight=weights[0::int(depth/overlap)] ) #FIXME 
    #sv.fit(samples, answers, sample_weight=None)

    return sv

def convolutional_layer(data, sample, half_width):
    '''TODO remove reduntant/repetitive sample extraction that adds latency'''
    #vectors = snag_samples(data)[0] 
    vectors = data 
    center = np.argmax(np.dot(vectors, sample))
 
    x = range(0, len(data)) 
    sigma = half_width**2 / np.log(.5)
 
    y = np.exp((x-center)**2/sigma) 

    
    return y
    
def snag_samples(data):
    samples = []
    for d in data:
        #result = [database.strip(j, desired_keys) for j in data]#database.strip(d, desired_keys)
        price = database.raw_strip(d, ['lastPrice']) 
        result = database.strip(d, desired_keys) 
        training_prices.append(price[-1]) 
        samples.append(result.flatten())
    return samples, training_prices 

def decision(prediction, threshold=action_threshold):
    #svm returns scalar here
    # if prediction >= action_threshold:
    #     market.buy() 
    # elif prediction <= -action_threshold:
    #     market.sell() 
    market.trigger_decide(prediction, threshold) 

def run_cycle(iter=10, learn_set=database.live_data, depth=DEPTH, threshold=action_threshold):    #good ol' fetch data from internet, feed in last *DEPTH* samples, feed into svm for prediction, execute decision based on prediction
    
    size = len(learn_set)#database.size
    for j in range(iter):
        #snag samples 
        if size < depth:
            prep(depth) 
        # print(size, size-DEPTH)
        # print(database.tick_data) 
        data = [database.live_data[i+1] for i in range(size-depth, size)] 
        print(np.shape(data)) 
        input()
        samples = snag_samples([data])[0] #data singular list

        prediction = brain.predict(samples) #samples should be singular list
        decision(prediction, threshold) 
        print(prediction) 

        #grab more data!
        d = database.full_data()
        size = database.log(d) 
        price = Watcher.price() 
        market.price_log(price) 

        market.stat() 
        vis.decision_graph(market.price_history, market.decisions) 
        #wait 
        sleep(delay) #timing here is imprecise, as does not consider latency 

def test_cycle(sv, validation_data, depth, threshold):
    '''this really just sees how much money sv returned'''
    size = len(validation_data)
    data = [[validation_data[i+1] for i in range(begin, begin+depth)] for begin in range(size-depth)] 

    market.price_history = []
    #from data extract every 25 or DEPTH/2 for 50% overlap between samples
    #data = data[0::int(depth/overlap)] 

    samples = []
    training_prices = []
  
    for d in data:
        #result = [database.strip(j, desired_keys) for j in data]#database.strip(d, desired_keys)
        price = database.raw_strip(d, ['lastPrice']) 
        result = database.strip(d, desired_keys) 
        training_prices.append(price[-1]) 
        samples.append(result.flatten())

    for i in range(len(samples)):
        pred = sv.predict([samples[i]]) 

        decision(pred, threshold)

        market.price_log(training_prices[i]) 
        market.stat()

    return market.assets 

def test_hyperparameters(overlap, depth, threshold, step, learn_set, validation_set):
    permutations = {(i, j, k, l):0 for i in overlap for j in depth for k in threshold for l in step} 

    iter = len(overlap)*len(depth)*len(threshold)*len(step) 
    i=1
    for p in permutations:
        print('Evaluating combination ', i, ' out of ', iter)
        o = p[0]
        d = p[1]
        t = p[2]
        s = p[3]

        brain = svm.SVR()

        learn(brain, learn_set, o, s, d) 

        result = test_cycle(brain, validation_set, d, t) 

        permutations[p] = result 

        i += 1

    return permutations 

def hyperparameter_statistics_and_adjustment(overlap=OVERLAP_FACTOR, depth=DEPTH, threshold=action_threshold, 
                                             step=STEP, prices = market.price_history):
    price_avg = np.average(prices) 
    price_movement_avg = np.std(prices) 

    avg_percent_diff = price_movement_avg/price_avg 

    STEP = abs(avg_percent_diff - 1)



    pass 


#brain = joblib.load('svm_data.joblib') 


print('Running...')
#print(database.size)
prep()
#weights = convolutional_layer(database.tick_data, database.live_data, 50)
size = len(database.live_data)
data = [database.live_data[i+1] for i in range(size-DEPTH, size)] 

s= np.array(snag_samples([data])[0]) #data singular list


brain = learn(brain, database.tick_data, s) 

print('database size: ', database.size) 
run_cycle(iter=100) 

print('size of the live data: ', len(database.live_data)) 
database.Save() 
print('Assets: ', market.assets) 
joblib.dump(brain, 'svm_data.joblib') 
print('Saved')  

#print(market.decisions) 

#vis.decision_graph(validation_prices, market.decisions, plot=True) 


'''brute force optimization of hyperparameters'''
# overlap = [1, 2, 4, 5, 10, 25, 50]
# depth = [5, 10, 25, 50, 75, 100]
# step = [.1, .01, .001, .0001, .00001] 
# threshold = [.25, .5, .75, .9] 
# learn_set = database.tick_data
# database.create_validation_set(size=360) 
# database.load_validation_data() 

# validation_set = database.validation_data #validation and train set same


# results = test_hyperparameters(overlap, depth, threshold, step, learn_set, validation_set) 

# #print(results) 
# results = joblib.load('hyperparameter_test.joblib') 

# max_result = [0, 0]
# for r in results:
#     if results[r]>max_result[1]:
#         max_result = [r, results[r]] 

# print('best parameter set: ', max_result[1], max_result[0]) 

# joblib.dump(results, 'hyperparameter_test.joblib') 

print('done') 





