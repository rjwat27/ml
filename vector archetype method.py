import math, datetime, random
from os import times
import os 
import numpy as np
from matplotlib import pyplot as plt 
from time import sleep, time
import time  
from urllib import request 
import requests 
import warnings 
import hmac 
import hashlib 
warnings.filterwarnings("ignore") 

archetypes = [] 
History = []
Prices = [] 
Predictions = []
Actuals = [] 
base_endpoint = "https://api.binance.us" 
batchsize = 10


accuracy = 1.0 

class Archetype(): 
    def __init__(self, initialization_vector, initial_bias):
        self.vector = initialization_vector 
        self.forward_vector = initialization_vector

        self.average_error = 0.0
        self.learningrate = 0.1 
        self.bias = initial_bias 
        self.forward_bias = initial_bias 

        self.sample_space = [] 
        self.sample_space_predictions = []
        self.sample_space_actuals = [] 

        self.sample_count = 0 

    def forward_vector_create(self):
        if np.size(self.sample_space) > 1: 
            sum = np.sum(self.sample_space, axis=0)
            self.forward_vector = sum / np.size(self.sample_space, axis=0) 
            self.forward_bias = 0
            for i in range(len(self.sample_space)):
                sample = self.sample_space[i]
                bias = self.sample_space_actuals[i] 
                self.forward_bias += (distance(sample, self.forward_vector)**2 * bias)
            self.forward_bias /= len(self.sample_space) 

    def Update_Archetype(self): 
        self.forward_vector_create() 
        direction = self.forward_vector - self.vector 
        self.vector += self.learningrate * self.average_error * direction 
        norm = np.linalg.norm(self.vector)
        self.vector = self.vector / norm 

        self.bias += self.learningrate * self.average_error * (self.bias - self.forward_bias) 
        if self.bias > 1:
            self.bias = 1
        elif self.bias < -1:
            self.bias = -1 

        if len(self.sample_space) > 100:
            self.sample_space.pop(0) 
        if len(self.sample_space_predictions) > 100:
            self.sample_space_predictions.pop(0) 
        if len(self.sample_space_actuals) > 100:
            self.sample_space_actuals.pop(0) 

        #new archetype creation
        if abs(self.bias) > .5:
            new_set = [] 
            bias_set = [] 
            for i in range(len(self.sample_space_actuals)):
                if abs(self.sample_space_actuals[i] - self.bias) > .5:
                    new_set.append(self.sample_space[i]) 
                    bias_set.append(self.sample_space_actuals[i]) 
            new_bias = np.average(bias_set) 
            if abs(self.bias - new_bias) > 1 and len(archetypes) < 100:
                New_Archetype(new_set, new_bias) 

    def error(self):
        if len(self.sample_space) == 0:
            return 0 
        error = 0
        for i in range(len(self.sample_space_actuals)):
            error += abs(self.sample_space_actuals[i] - self.sample_space_predictions[i])
        self.average_error = error / batchsize 

def distance(vect1, vect2):
    return np.dot(vect1, vect2) 

def bias_influence(vect1, vect2):
    return np.exp(distance(vect1, vect2) - 1)**2

def Predict(input):
    bias1 = 0
    bias2 = 0
    factor1 = 0
    factor2 = 0 
    z = archetypes[0].vector 
    for vector in archetypes: #find closest archetype 
        x = distance(vector.vector, input)
        if x > factor1:
           factor1 = x
           bias1 = vector.bias 
           z = vector.vector 
    for vector in archetypes: #find second closest archetype 
        x = distance(vector.vector, input)
        if x > factor2 and np.dot(z, vector.vector) < 1:
            bias2 = vector.bias 
            factor2 = x
       
    return (bias1*factor1 + bias2*factor2) / 2

def GetPrices():
  try:
    params = {'symbol':'BTCUSD'} 
    Data1 = requests.get(base_endpoint+"/api/v3/ticker/24hr", params=params)  
    #print(Data1.status_code)
    Data = Data1.json() 
   
  except:
    
      print(Data1.status_code)
      print("Error ocurred in fetching price data" )
    
      Data = "error" 
      return Data 

  return Data['lastPrice']

def Get_Data():
    try:
        params = {'symbol':'BTCUSD', 'interval':'1m', 'limit':'2'} 
        Data1 = requests.get(base_endpoint+"/api/v3/klines", params=params)  
        # print(Data1.status_code)
        Data = Data1.json() 

        kline_final = np.array(Data[1])
        kline_initial = np.array(Data[0])
        kline_diff = [] 
        for i in range(len(kline_final)):
            kline_diff.append(float(kline_final[i]) - float(kline_initial[i])) 
        
    
    except:
        #   errornum += 1
        print(Data1.status_code)
        print("Error ocurred in fetching price data" )
        #   print("Error #: ", errornum ) 
        Data = "error" 
        return Data 

    norm = np.linalg.norm(np.array([kline_diff[1], kline_diff[2], kline_diff[3], kline_diff[4], kline_diff[5], kline_diff[8]]))

    return np.array([kline_diff[1], kline_diff[2], kline_diff[3], kline_diff[4], kline_diff[5], kline_diff[8]]) / norm

def Sort(input, prediction, actual):
    max = 0  
    archetype = archetypes[0] 
    for vector in archetypes:
        if np.dot(input, vector.vector) > max:
            max = np.dot(input, vector.vector)
            archetype = vector 

    archetype.sample_space.append(input) 
    archetype.sample_space_predictions.append(prediction)
    archetype.sample_space_actuals.append(actual)  

def Archetype_Initialize(dim):
    x = np.zeros(2**dim) 
    for i in range(2**dim):
        x[i] = bin(i)[2:]
    vector_list = []
    for y in x:
        bits = str(int(y)).zfill(dim)
        p = np.zeros(dim) 
        for i in range(dim):
            p[i] = (int(bits[i])*2) - 1
        norm = np.linalg.norm(p) 
        vector_list.append(p / norm) 
    return vector_list

def Save_Data():
    packet = [] 
    
    for archetype in archetypes:
        data = []
        data.append(len(archetype.sample_space)) 
        data.append(archetype.bias) 
        data = data + np.ndarray.tolist(archetype.vector)
        
        for i in range(len(archetype.sample_space)):
            data = data + np.ndarray.tolist(archetype.sample_space[i])
            #data = data + [np.ndarray.tolist(np.array(archetype.sample_space_predictions[i]))]
            try:
                data = data + np.ndarray.tolist(np.array(archetype.sample_space_actuals[i]))
            except:
                data.append(np.ndarray.tolist(np.array(archetype.sample_space_actuals[i])))
        packet = packet + data 
    
    np.savetxt('archetype data.csv', np.array(packet), delimiter=', ') 

def Load_Data(dim): 
    data = np.ndarray.tolist(np.loadtxt('archetype data.csv', delimiter=', '))
    position = 0 
    for i in range(len(archetypes)):
        samples = int(data[position]) 
        archetypes[i].bias = data[position+1] 
        archetypes[i].vector = data[position+2:position+2+dim]
        position = position+2+dim
        breadth = (position+samples*(dim+1))
        
        sample_packet = data[position:breadth:1+dim]
        position = position + breadth 
        for j in range(samples):
            print(len(sample_packet), samples) 
            start = j*dim+1
            end = start + dim 
            archetypes[i].sample_space.append(sample_packet[start:end]) 
            #archetypes[i].sample_space_prediction.append(sample_packet[i][dim])
            archetypes[i].sample_space_actuals.append(sample_packet[end]) 

def New_Archetype(samples, bias):
    sum = np.sum(samples, axis=0) 
    vector = sum / len(samples) 
    new_archetype = Archetype(vector, bias) 
    archetypes.append(new_archetype)  

def Apoptosis(archetypes):
    for i in range(len(archetypes)):
        archetype = archetypes[i]
        for j in range(len(archetypes)):
            others = archetypes[j] 
            if archetype.vector.any() != others.vector.any():
                if distance(archetype.vector, others.vector) > .9:
                    if (archetype.bias >= 0 and others.bias >= 0) or (archetype.bias <= 0 and others.bias <= 0):
                        if abs(archetype.bias - others.bias) > 0:
                            archetypes.pop(j) 
                        else:
                            archetypes.pop(i) 
#initialize archetypes
input2 = Get_Data() 

dim = len(input2) 
test = 0
vectors = Archetype_Initialize(dim) 

for vector in vectors:
    new_bias = np.random.rand()*2 - 1 
    new_archetype = Archetype(vector, 0) 
    archetypes.append(new_archetype)

load_data = False 

if load_data:
    Load_Data(dim) 

price = GetPrices() 

prediction = Predict(input2)

History.append(input2)
Prices.append(price) 
Predictions.append(prediction) 

counter = 0
fail_counter = 0 
sleep(60)  
print("Running")
while True:
    #os.system('cls||clear')
    try:
        input1 = Get_Data() 

        price = GetPrices()
    except:
        print("connection error") 
        input1 = "error" 
        continue 

    # if input1=='error' or price=='error':
    #     print("Network error occurred.") 
    #     response = input("Continue? Y or N:")
    #     if response == 'Y':
    #         continue 
    #     else:
    #         quit 

    prediction = Predict(input1)

    History.append(input1)
    Prices.append(price) 
    Predictions.append(prediction) 

    #if len(Prices) >= 2:
    if Prices[-1] > Prices[-2]:
        Actuals.append(1) 
    if Prices[-1] < Prices[-2]: 
        Actuals.append(-1) 
    else:
        Actuals.append(0) 
    Sort(History[0], Predictions[0], Actuals[0]) 
    print("prediction: ", Predictions[0])
    print("actual: ", Actuals[0]) 
    History.pop(0) 
    Predictions.pop(0)
    test = Actuals[0]
    Actuals.pop(0) 
    counter += 1
    fail_counter += 1
    print(fail_counter) 
    if counter >= batchsize:
        for archetype in archetypes:
            if len(archetype.sample_space) > 10:
                archetype.forward_vector_create() 
                archetype.Update_Archetype()  
                archetype.error() 
                print("Archetype Update Successfull") 
        
        #error / effective archetypes 
            error = 0
            effective_archetypes = 0
            for archetype in archetypes:
                error += archetype.average_error 
                if archetype.bias != 0:
                    effective_archetypes += 1
            if effective_archetypes > 0:
                error = error / effective_archetypes 
        Apoptosis(archetypes) 
        print("Effective Archetypes: ", effective_archetypes)
        counter = 0
        # print("Average error: ", error) 
        
        # print("Effective Archetypes: ", effective_archetypes)
        History = [History[-1]] 
        Prices = [Prices[-1]] 
        Predictions = [Predictions[-1]]
        Actuals = []
        Save_Data() 
    
    sleep(60) 




