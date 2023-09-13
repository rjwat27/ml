from typing import Iterable
import numpy as np 
import json, copy 


class archetype():
    def __init__(self, key=0, archetype=0, bias=0, size=0, data_samples=[], training_data=[], sample_standard_deviation=0, 
                bias_standard_deviation=0, expert=0, accuracy=1, data_size=0, max = 1000): 
        self.key = key       
        self.archetype = archetype  #pass a normalized set 
        self.bias = 0 
        self.size = len(archetype) 
        self.max = max 

        self.data_samples = data_samples        
        self.training_data = []                 #list of tuples as ([data set], outcome)
        self.decisions = [] 
        self.data_size = 0 
        self.sample_standard_deviation = 0 
        self.bias_standard_deviation = 0 

        self.expert = expert    #FNN object  as dict 
        self.accuracy = 1

        self.age = 0 
        self.mistaken_data = [] 

    def set_expert(self, net):
        expert = copy.deepcopy(net.__dict__)  
        for i in net.connections:
            expert['connections'][str(i)] = net.connections[i].__dict__  
            expert["connections"].pop(i) 
        for i in net.nodes:
            expert["nodes"][i] = net.nodes[i].__dict__
        expert['ninputs'] = self.size 
        expert['noutputs'] = 1 #for now
        expert['hidden_layer1'] = 0
        expert['hidden_layer2'] = 0 
        
        self.expert = expert   #FNN object  as dict 

    def log(self, data):
        for item in [data]:
            self.data_samples.append(item) 
            self.data_size += 1 

    def train_log(self, data):
        self.training_data.append(data) 

        if len(self.training_data) > self.max:
            self.training_data.pop(0) 

    def stat(self, depth, jump=.001):  #depth in number of 15sec intervals
        #update archetype 
        data_matrix = []
        biases = []  
        #create training data from data_samples
        # for i in range(depth, len(self.data_samples)-depth):
        #     x = self.data_samples[i]["lastPrice"]
        #     y = self.data_samples[i+depth]["lastPrice"]
        #     if y/x > 1+jump:
        #         change = 1
        #     elif y / x < 1-jump:
        #         change = -1
        #     else:
        #         change = 0 
        #     self.training_data.append(([self.data_samples[j] for j in range(i-depth, i)], change)) 
        for item in self.data_samples:
            data_matrix.append(item[0])             #test 
            biases.append(item[1]) 
       
        # if len(data_matrix) > 1:
        #     self.archetype = np.average(data_matrix, axis=0)
        self.bias = np.average(biases)
        self.size = len(self.archetype)  

        #archetype statistics
        # sample_deviations = [] 
        # bias_deviations = [] 
        # for item in self.data_samples:      #ensure that everything is numpy array 
        #     sample_deviations.append(self.archetype - item[0]) 
        #     bias_deviations.append(self.bias - item[1]) 
        # self.sample_standard_deviation = np.std(sample_deviations) 
        # self.bias_standard_deviation = np.std(bias_deviations) 
        2-2

def make_arch(data):
    return archetype(archetype=data) 

class Archetype_Manager():
    def __init__(self, filename):
        self.archetypes = {}
        self.archetype_count = 0 
        self.max_radius = 0     #radians 
        self.filename = filename
    

    def cluster(self, data, bias, similarity_threshold=.5):
        #find average vector from samples
        #initiate archetype for average vector
        #repeat for all samples outside similarity threshold
        #until all samples are assigned an archetype...samples assumed to be normalized 
        archetype_list = [] 
        count = 0
        while data: #iterate while data samples are yet to be grouped 
            vectors = [] 
            for u in data:
                avg = np.average(u, axis=0) 
                max = np.max(u) 
                min = np.min(u) 
                vector = (np.array(u) - min*np.ones(len(u))) * (2/(max - min)) - np.ones(len(u)) 
                vectors.append(vector) 
            
            mean = np.average(vectors, axis=0) 
            #arch = archetype(count, mean, [], 0) 
            surrogate = archetype(key=count, archetype=list(vectors[0])) 
           
            count += 1 
            left_over = [] 
        
            if True:#len(arch.data_samples)==0:
                left_over = [] 
                for vector in vectors:
                    if self.archetypes:
                        for a in self.archetypes:
                            if np.dot(self.archetypes[a].archetype, vector) >= similarity_threshold:
                                self.archetypes[a].log(list(vector))
                                self.archetypes[a].training_data.append((list(vector), bias)) 
                    elif np.dot(surrogate.archetype, vector) >= similarity_threshold:
                        surrogate.log(list(vector))
                        surrogate.training_data.append((list(vector), bias)) 
                    else:
                        left_over.append(vector) 
                if surrogate.training_data:
                    archetype_list.append(surrogate) 
            else:
                archetype_list.append(arch) 
            
            data = left_over 

        for i in range(len(archetype_list)):
            self.add(archetype_list[i])
        
        return archetype_list

    def partition(self, data, context=20, foresight=20, jump=.001):  #context = how far in the past to consider; foresight = how far ahead to calculate net change; 
                                                    #jump = decimal percent change whether up or dow
        up = []
        down = []
        for i in range(context+1, len(data)-foresight):
            x = float(data[str(i)]['lastPrice'])
            y = float(data[str(i+foresight-1)]['lastPrice']) 
            if y/x >= 1+jump:
                up.append([data[str(j)] for j in range(i-context+1, i+1)]) 
            if y/x <= 1-jump:
                down.append([data[str(j)] for j in range(i-context+1, i+1)])        #test 
        up_vectors = []

        for i in up:
            vector = []
            for j in i:
                vector += [float(j['lastPrice'])]
            #print(vector) 
            vector = np.array(vector)
            up_vectors.append(vector) 

        down_vectors = [] 

        for i in down:
            vector = []
            for j in i:
                vector += [float(j['lastPrice'])]

            vector = np.array(vector) 
            down_vectors.append(vector) 
        return up_vectors, down_vectors 

    def match(self, data):      #pass normalized data 
        #given data sample, return corresponding archetype key
        #for now just iterate through archetypes for O(n) 
        #later implement tree for O(logn) 
        
        max = None
        match = 0
        if not self.archetypes:
            arch =  make_arch(data)
            self.add(arch) 
        for archetype in self.archetypes:
            distance = np.dot(self.archetypes[archetype].archetype, data) 
            if max==None or distance > max:
                max = distance
                match = self.archetypes[archetype] 
        match.log(data) 
        return match, max 
        
    def evaluate(self, data):   #normalized data
        expert = self.match(data)[0].expert 
        #on expert call neural net evaluate function 

    def stat(self, depth=20):
    
        num = 0
        arch_list = [] 
        for archetype in self.archetypes:
            archetype = self.archetypes[archetype] 
            #archetype.stat(depth) 
            arch_list.append(archetype.archetype) 
            num += 1
        self.archetype_count = num 
        avg = np.average(arch_list, axis=0) 
        max = 0 
        far = 0
        for a in self.archetypes:
            a = self.archetypes[a]
            #apop and mitosis
            if a.age > 100 and a.accuracy < .75: #accuracy threshold 
                self.mitosis(a) 
            distance = np.dot(a.archetype, avg) 
            if distance < max:
                max = distance
                far = a 
        self.max_radius = np.arccos(max / (np.linalg.norm(far.archetype)*np.linalg.norm(avg)))

    def add(self, archetype):
        archetype.key = self.archetype_count + 1
        self.archetypes[archetype.key] = archetype 
        self.archetype_count += 1 

    def Save(self):
        load_data = {}
        for a in self.archetypes:
            a_data = {}
            for i in self.archetypes[a].__dict__:
                a_data[i] = self.archetypes[a].__dict__[i] 
            load_data[a] = a_data 
        file = open(self.filename, 'w') 
        json.dump(load_data, file) 

    def Load(self): 
        self.size = 0 
        file = open(self.filename, 'r') 
        data = json.load(file) 
        load_data = {} 
        for entry in data:      #iterate over archetype data entries 
            load_data[entry] = data[entry]#json.loads(data[entry]) 
        for a in load_data:
            a = load_data[a] 
            #par = a.__dict__ 
            arch = archetype(**a)#a['key'], a['archetype'], a['bias'], a['size'], a['data_samples'], a['training_data'], a['sample_standard_deviation'], 
                            #a['bias_standard_deviation'], a['expert'], a['accuracy'])
            self.archetypes[self.archetype_count+1] = arch 
            self.archetype_count += 1 
            
    #mitosis and apoptosis 
    def mitosis(self, arch):
        data = arch.mistaken_data
        avg = np.average(data) 
        u = make_arch(avg) 
        max = np.max(u) 
        min = np.min(u) 
        a = (np.array(u) - min*np.ones(len(u))) * (2/(max - min)) - np.ones(len(u)) 
        archetype = make_arch(a)
        self.add(archetype)  

        arch.mistaken_data = [] 

    def apoptosis(self, arch): 
        self.archetypes.pop(arch.key)





