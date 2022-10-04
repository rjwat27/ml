#cryptoVision Top
import numpy as np
import datetime, time, requests, random 
from time import sleep 
import json, copy 
import argparse, os, sys

#import NEATModule as neat, NNModule as nn, TrainingModule as tm
import TrainingModule as tm, NNModule as nn 
import DataModule as dm, ArchetypeModule as am#, PartitionModule as pm 
import VirtualMarketModule as vm, VisualizeModule as vis  

class Connection:
    def __init__(self, key, weight, enabled):
        self.key = key
        self.weight = weight
        self.partial = 1 
        self.enabled = enabled

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="Train from stored tick data", default=False, type=bool) 
parser.add_argument("-r", "--run", help="Normal operation mode; training and trading", default=False, type=bool) 

Archetype_Manager = am.Archetype_Manager("archetype_data.txt") 
Watcher = dm.Watcher() 
DataBase = dm.DataBase('BTCUSD', "tick_data_BTCUSD.txt") 
Market = vm.Market() 

desired_keys =  ['lastPrice', "highPrice", "lowPrice", "priceChangePercent", "volume"]
# DataBase.Load() 

# up, down = Archetype_Manager.partition(DataBase.tick_data) 

# Archetype_Manager.cluster(up, 1)
# Archetype_Manager.cluster(down, -1)  
#Archetype_Manager.Load() 

#begin with preliminary training for all archetypes
# print("\033[H\033[2J", end="")
# print("Trained ", 0, " out of ", Archetype_Manager.archetype_count) 

#     print("\033[H\033[3J", end="")
#     print("Trained ", a, " out of ", Archetype_Manager.archetype_count) 

# buy_threshold = .75
# sell_threshold = -.75 
action_threshold = .75
depth = 20
foresight = 20 
def prep(depth=20):
    for i in range(depth):
        data = Watcher.full_data('BTCUSD') 
        #Market.price_log() 
        DataBase.log(data) 
        sleep(15) 
def eval(depth=20): #database needs sufficient data for depth and of appropriate size
    
    data = [DataBase.tick_data[str(i)] for i in range(DataBase.size - depth, DataBase.size)]
    
    data = DataBase.strip(data, desired_keys) #glean info from tick data we want and normalize
    
   
    archetype = Archetype_Manager.match(data)[0] 
    Market.arch_log(archetype)
    Market.data_log(data) 
    net = 0

    if archetype.expert is list:
        a_dict = copy.deepcopy(archetype.expert)
    

        a_dict.pop('key') 
        a_dict.pop('fitness') 
        
        
        for i in archetype.expert["connections"]:
            
            a_dict['connections'][tuple(i)] = Connection(tuple(i), float(archetype.expert['connections'][i]['weight']),  archetype.expert['connections'][i]['enabled'])
            a_dict['connections'].pop(i) 
   
        net = nn.FNN(**a_dict)
    elif archetype.expert is nn.FNN:
        net = archetype.expert 
    else:
        archetype.expert = nn.FNN(archetype.size, 3, 30, 10, [], [])
        net = archetype.expert
    archetype.log(data) 
    
    return net.activate(data), archetype.accuracy
def run(iter=60, depth=20, wait=15):
    for i in range(iter):
        data = Watcher.full_data('BTCUSD') 
        DataBase.log(data)
        
        Market.price_log(float(data['lastPrice'])) 
        
        prediction, accuracy = eval(depth)
        
 
        if max(prediction) == prediction[0] and prediction[0]*accuracy >= action_threshold:
            print('Buy')
            Market.decide(1) 
        elif max(prediction) == prediction[1] and prediction[1]*accuracy >= action_threshold:
            print('Sell') 
            Market.decide(-1) 
        else:
            Market.decide(0)
       

        Market.stat() 
        time4 = time.time()
        vis.graph(Market.price_history, Market.decisions) 
        time5 = time.time()
        graph_time = time5-time4
        print('graph time: ', graph_time, '\n') 
        #print("\033[H\033[3J", end="")
        print("Assets: ", Market.assets) #lets um add more huh? 
        time3 = time.time() 
        if time3-time0 < 15:
            delay = 15 - (time3-time0) 
            sleep(delay) 
        #sleep(wait) 

def train_all():
    for a in Archetype_Manager.archetypes:
        tm.train(Archetype_Manager.archetypes[a]) 
        #print("\033[H\033[3J", end="")
        print('Training...') 
        print("Trained ", a, " out of ", Archetype_Manager.archetype_count) 
def train(set): #input list of archetypes to train 
    for a in set:
        tm.train(Archetype_Manager.archetypes[a]) 
        print("\033[H\033[3J", end="")
        print("Trained ", a, " out of ", len(set))  

def train_2(arch):       #backpropogation without neat algo 
    a = Archetype_Manager.archetypes[arch]
    sample_set = [a.training_data[i][0] for i in range(len(a.training_data))]  #my sketchy way to improve intermitent learning fer now 
    answer_set = [a.training_data[i][1] for i in range(len(a.training_data))] 
    net = 0

    if a.expert is list:
        a_dict = copy.deepcopy(a.expert)
    

        a_dict.pop('key') 
        a_dict.pop('fitness') 
        
        
        for i in a.expert["connections"]:
            
            a_dict['connections'][tuple(i)] = Connection(tuple(i), float(a.expert['connections'][i]['weight']),  a.expert['connections'][i]['enabled'])
            a_dict['connections'].pop(i) 
   
        net = nn.FNN(**a_dict)
    elif a.expert is nn.FNN:
        net = a.expert 
    else:
        a.expert = nn.FNN(a.size, 3, 30, 10, [], [])
        net = a.expert
    if sample_set:
        result = nn.GD(net, sample_set[-2:-1], answer_set[-2:-1])  
        a.expert = result[0]
    #print("\033[H\033[3J", end="")
    # print("Trained ", a.key, " out of ", Archetype_Manager.archetype_count) 
    # print("best error", result[1]) 


for a in Archetype_Manager.archetypes:
    Archetype_Manager.archetypes[a].expert = nn.FNN(Archetype_Manager.archetypes[a].size, 3, 30, 10, [], []) 

print('Running...') 
prep() 
time0 = time.time()
while True:
    time9 = time.time()
    run(iter=1)
    time0 = time.time() 
    time6 = time.time()
    training_data = Market.training_data_prep() 
    time7 = time.time()
    print('time make training data: ', time7-time6, '\n') 
    for i in training_data: #(archetype, data, change)      dont worry...this works for any input set
        i[0].train_log((i[1], i[2])) 
    if DataBase.size > depth + foresight:
        time1 = time.time()
        for a in Archetype_Manager.archetypes:
            Archetype_Manager.archetypes[a].stat(depth=20)
            #time1 = time.time()  
            print('data size: ', len(Archetype_Manager.archetypes[a].training_data)) 
            train_2(a) 
            #time2 = time.time()
           
            #print('time: ', time2 - time1) 
        time2 = time.time()
        print('train time: ', time2-time1, '\n') 
    time10 = time.time()
    print('total time: ', time10-time9) 
    print('num archetypes: ', Archetype_Manager.archetype_count) 
  
    

print("done.") 


