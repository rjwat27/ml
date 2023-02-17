import numpy as np
#API interface
class Master():
    def __init__(self):
        self.nothing = None 
        self.decision_threshold = .8

    def decide(self, confidence, accuracy):
        if confidence * accuracy > self.decision_threshold:
            2-2

class Decision_History():

    def __init__(self, max):
        self.max = max
        self.arch_history = [] 
        self.result_history = []
        self.price_history = [] 
        self.data_history = [] 

    def result_log(self, result):
        self.result_history.append(result) 
    def arch_log(self, archetype):
        self.arch_history.append(archetype) 
    def price_log(self, price):
        self.price_history.append(price) 
    def data_log(self, prediction):
        self.data_history.append(prediction) 

    def training_data_prep(self, depth, foresight, jump=.001):
        training_data = []
        for i in range(depth, len(self.price_history)-foresight):
            up = False
            down = False
            up_jump = 0
            down_jump = 0
            change = 0
            for j in range(i, i+foresight):
                x = self.price_history[i]
                y = self.price_history[j]
                if y/x >= 1+jump:
                    up = True
                    up_jump = y/x - 1
                if y/x <= 1-jump:
                    down = True
                    down_jump = 1 - y/x  
                
            if up and down:
                if up_jump==down_jump:
                    change = 0
                elif up_jump>down_jump:
                    change = 1
                else:
                    change = -1 
            elif up:
                change = 1
            elif down:
                change = -1 
            entry = (self.arch_history[i], self.data_history[i], change) 
            training_data.append(entry) 
            


