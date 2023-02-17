#partition

import numpy as np
import datetime, time, requests, random 
from time import sleep 
import json 
import ArchetypeModule as AM 

file = open("training_data3.txt", 'r') 
data = json.load(file) 

for entry in data:
    data[entry] = json.loads(data[entry]) 

up = []
down = []

breadth = 20
foresight = 20 

def partition(data, context, foresight, jump=.001):  #context = how far in the past to consider; foresight = how far ahead to calculate net change; 
                                                #jump = decimal percent change whether up or dow
    up = []
    down = []
    for i in range(context, len(data)-foresight):
        x = float(data[str(i)]['lastPrice'])
        y = float(data[str(i+foresight-1)]['lastPrice']) 
        if y/x >= 1+jump:
            up.append([data[str(j)] for j in range(i-context-1, i+1)]) 
        if y/x <= 1-jump:
            down.append([data[str(j)] for j in range(i-context-1, i+1)])        #test 
    return up, down 
        
similarity_threshold = .9 

#clustering 

    # change = float(tick['priceChangePercent']) 
    # top = float(tick['highPrice']) 
    # low = float(tick['lowPrice'])
    # volume = float(tick['volume']) 
    # count =  float(tick['count'])

up_vectors = []

for i in up:
    vector = []
    for j in i:
         vector += [float(j['lastPrice'])]#float(j['priceChangePercent']), float(j['highPrice']) , float(j['lowPrice'])]#, float(j['volume']) , float(j['count'])]

    vector = np.array(vector)# / np.linalg.norm(vector) 
    up_vectors.append(vector) 

down_vectors = [] 

for i in down:
    vector = []
    for j in i:
         vector += [float(j['lastPrice'])]#[float(j['priceChangePercent']), float(j['highPrice']) , float(j['lowPrice'])]#, float(j['volume']) , float(j['count'])]

    vector = np.array(vector)# / np.linalg.norm(vector) 
    down_vectors.append(vector) 
    
up_mean = np.average(down_vectors, axis=0) 


archetype_list_up = AM.cluster(up_vectors, similarity_threshold) 

archetype_list_down = AM.cluster(down_vectors, similarity_threshold) 


