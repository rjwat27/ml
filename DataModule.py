#crypto data module
import numpy as np
import datetime, time, requests, random 
from datetime import datetime 
from time import sleep 
import json, os

class Watcher():
    def __init__(self):
        self.nothing = 0 
        self.base_endpoint = "https://api.binance.us" 

    def full_data(self, symbol='BTCUSD'):
        Data1 = requests.get(self.base_endpoint+"/api/v3/ticker/24hr?symbol=" + symbol) 
        Data = Data1.json() 
        return Data 

    def price(self, symbol='BTCUSD'):
        Data1 = requests.get(self.base_endpoint + "/api/v3/ticker/price?symbol=" + symbol)  
        Data = Data1.json()
        return float(Data['price']) 



class DataBase():
    def __init__(self, symbol, filename):
        self.symbol = symbol
        self.tick_data = {} 
        self.live_data = {} 
        self.validation_data = {}
        self.filename = filename 
        self.base_endpoint = "https://api.binance.us" 
        self.size = 0 
        self.validation_size = 0

        self.desired_keys =  ["highPrice", "lowPrice", "priceChangePercent", "volume", 'lastPrice']

    def Save(self, data=None, fname=None):
        if fname is None:
            now = datetime.now()
            date_time = now.strftime("%m_%d_%Y_%H%M") 
            fname =  self.filename+"_"+date_time
   
        if data is None:
            data = self.live_data
        file = open(fname, 'w') 
        json.dump(data, file)
        print("Saved to file: ", fname) 

    def Load(self, fname=None):     #raw data
        # if fname is None:
        #     fname = self.filename
        '''TODO select files based on security type, for now all bitcoin'''
        local_dir = os.path.dirname(__file__)
        file_dir = local_dir+"/tick data"#os.path.join(local_dir, self.filename)
        # self.size = 0 
        # file = open(file_path, 'r') 
        # data = json.load(file) 
        files = os.listdir(file_dir)#local_dir+"/tick data") 
        tick_size = 0
        for file in files:
            print(file) 
            f = open(file_dir+"/"+file, 'r')
            data = json.load(f) 
            sub_tick = {} 
            num = 0
            for entry in data:  #this should render all previous data replaced
                # if type(entry)==str:
                #     entry = 
                sub_tick[num+1] = data[entry] 
                num += 1
                #self.tick_data[self.size+1] = data[entry]
                # self.tick_data[entry]['tag'] = self.size + 1 
                
            self.tick_data[tick_size+1] = sub_tick
            tick_size += 1  
            #self.size = len(self.tick_data) 
        
    def full_data(self, symbol='BTCUSD'):
        Data1 = requests.get(self.base_endpoint+"/api/v3/ticker/24hr?symbol=" + symbol) 
        Data = Data1.json() 
        return Data    

    def log(self, data):
        data['tag'] = self.size + 1
        self.live_data[int(self.size+1)] = data 
        self.size += 1 
        

        if self.size > 10000:
            now = datetime.now() 
            date_time = now.strftime("%m_%d_%Y_%H:%M")
            self.Save(fname = "tick_data_BTCUSD_"+date_time) 
            print("Saved to file: tick_data_BTCUSD_"+date_time)
            self.live_data = {}
            self.size = 0 
            print("Backed up data; prepping new data") 
            for i in range(50):
                data = self.full_data()
                self.log(data) 
            return self.size 
        else:
            return self.size 

    def raw_strip(self, data, keys):        #same as strip function but without normalizing/fixing up
        if type(data)==dict:
            data = np.array([[float(data[d][key]) for key in keys] for d in data]) 
        if type(data)==list:
            data = np.array([[float(d[key]) for key in keys] for d in data])
        
        return data#data.flatten()
    def strip(self, data, keys):
        if type(data)==dict:
            data = np.array([[float(data[d][key]) for key in keys] for d in data]) 
        if type(data)==list:
            data = np.array([[float(d[key]) for key in keys] for d in data])

        for i in range(len(keys)):
            max = np.max([j[i] for j in data]) 
            min = np.min([j[i] for j in data]) 
            if max==min and max > 0:
                data[:,i] = .5*np.ones(len(data)) 
            elif max==min and max < 0:
                data[:,i] = -.5*np.ones(len(data)) 
            else:
                data[:,i] = (np.array(data[:,i]) - min*np.ones(len(data))) * (2/(max - min)) - np.ones(len(data)) #test
        return data#data.flatten()

        
    def partition(self, data, depth=20, foresight=20, jump=.001):  #depth = how far in the past to consider; foresight = how far ahead to calculate net change; 
                                                #jump = decimal percent change whether up or dow
        up = []
        down = []
        for i in range(depth, len(data)-foresight):
            x = float(data[str(i)]['lastPrice'])
            y = float(data[str(i+foresight-1)]['lastPrice']) 
            if y/x >= 1+jump:
                up.append([data[str(j)] for j in range(i-depth, i+1)]) 
            if y/x <= 1-jump:
                down.append([data[str(j)] for j in range(i-depth, i+1)])        #test 
        return up, down 

    def create_training_set(self, size=600, wait=10):
        for i in range(size):
            print(i) 
            self.log(self.full_data()) 
            sleep(wait) 
        self.Save(data=self.live_data, fname='training_data.txt') 

    def create_validation_set(self, size = 200, wait=10):
        self.validation_size = 0
        for i in range(size):
            print(i)
            data = self.full_data()
            self.validation_data[self.validation_size+1] = data 
            self.validation_size += 1  
            sleep(wait)
        
        self.Save(self.validation_data, 'validation_data.txt') 

    def load_validation_data(self, fname='validation_data.txt'):
        local_dir = os.path.dirname(__file__)
        file_path = os.path.join(local_dir, fname)
        #self.size = 0 
        file = open(file_path, 'r') 
        data = json.load(file) 


        for entry in data:
            self.validation_data[int(entry)] = data[entry]

    def list_of_samples(self, num_of_samples=100, len_of_samples=50):
        desired_keys =  self.desired_keys
        samples = [] 
        training_prices = []
        data = [[self.tick_data[i+1] for i in range(begin, begin+len_of_samples)] for begin in range(len(self.tick_data) - len_of_samples)] 

        for d in data:
            #result = [database.strip(j, desired_keys) for j in data]#database.strip(d, desired_keys)
            price = self.raw_strip(d, ['lastPrice']) 
            result = self.strip(d, desired_keys) 
            training_prices.append(price[-1]) 
            samples.append(result.flatten()) 

        pass 



