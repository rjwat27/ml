import numpy as np 



class archetype():
    def __init__(self, archetype, data_samples, expert):
        self.archetype = archetype 
        self.bias = 0 
        self.size = len(archetype) 

        self.data_samples = data_samples        #list of tuples as ([data set], outcome) 
        self.sample_standard_deviation = 0 
        self.bias_standard_deviation = 0 

        self.expert = expert 

    def set_expert(self, expert):
        self.expert = expert    #FNN object  

    def log(self, data):
        for item in data:
            self.data_samples.append(item) 

    def stat(self):
        #update archetype 
        data_matrix = []
        biases = []  
        for item in self.data_samples:
            data_matrix.append(item[0]) 
            biases.append(item[1]) 
        self.archetype = np.average(data_matrix, axis=0)
        self.bias = np.average(biases)
        self.size = len(self.archetype)  

        #archetype statistics
        sample_deviations = [] 
        bias_deviations = [] 
        for item in self.data_samples:      #ensure that everything is numpy array 
            sample_deviations.append(self.archetype - item[0]) 
            bias_deviations.append(self.bias - item[1]) 
        self.sample_standard_deviation = np.std(sample_deviations) 
        self.bias_standard_deviation = np.std(bias_deviations) 

        

