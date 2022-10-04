#feed forward backpropogation NN
import ArchetypeModule
from decimal import Overflow
from email.policy import Compat32
from http.client import NOT_MODIFIED
from operator import xor
from os import error
from typing import List, final
import numpy as np
import math
import datetime, copy 
from time import sleep, time  
import urllib.request 
import sys, random 
from matplotlib import pyplot as plt
import matplotlib as mpl
from numpy.core.function_base import linspace
from numpy.lib import histograms
from numpy.lib.function_base import delete
from numpy.typing import _128Bit 
import requests, json
mpl.use('agg') 
 

# # 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   1.0,     0.0,     0.0,     0.0]
# # 3-input XOR inputs and expected outputs 
# xor_inputs = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]
# xor_outputs = [   0.0,     1.0,     1.0,     0.0,   1.0,    0.0,  0.0,   1.0]

batchsize = 5


class FNN():
    def __init__(self, ninputs, noutputs, hidden_layer1, hidden_layer2, nodes, connections):
        self.inputs = ninputs
        self.outputs = noutputs 
        self.hidden1 = hidden_layer1
        self.hidden2 = hidden_layer2 
        self.connections = {} 
        self.nodes = {}
        self.sources = {} 
        self.learningrate = 2

        self.layer1 = []
        self.layer2 = [] 


        class Node:
            def __init__(self, key, bias):
                self.key = key
                self.bias = bias 
                self.input = 0
                self.output = 0 
                self.partial = 1 
                self.sources = 0 
                self.delta = -1 

        class Connection:
            def __init__(self, key, weight, enabled):
                self.key = key
                self.weight = weight
                self.partial = 1 
                self.enabled = enabled 
        if not nodes:
            node_list = [Node(key, np.random.rand()*2-1) for key in range(-self.inputs, self.hidden2 + self.hidden1 + self.outputs)] #biases initialized at 0 
            for node in node_list:
                self.nodes[node.key] = node 
        else:
            for n in range(-self.inputs, 0):
                self.nodes[n] = Node(n, 0) 
    
            for n in nodes:
                if hasattr(nodes[n], 'bias'): 
                    self.nodes[n] = Node(n, nodes[n].bias)
                     
                else:
                    self.nodes[n] = Node(n, 0) 
        if len(connections)==0:     #modified for recurrent networks 
            count = 0 
            for i in range(-self.inputs, 0):
                for j in range(self.outputs, self.outputs + self.hidden1):
                    self.connections[(i, j)] = (Connection([i, j], np.random.rand()*2 - 1, True if i!=j else False))
                    count += 1
            if self.hidden2 != 0:
                for j in range(self.outputs, self.outputs + self.hidden1):
                    for k in range(self.outputs + self.hidden1, self.outputs+self.hidden1+self.hidden2):#self.inputs+self.hidden, self.inputs+self.hidden+self.outputs):
                        self.connections[(j, k)] = (Connection((j, k), np.random.rand()*2 - 1, True if j!=k else False)) 
                for k in range(self.outputs + self.hidden1, self.outputs+self.hidden1+self.hidden2):
                    for l in range(self.outputs): 
                        self.connections[(k, l)] = (Connection((k, l), np.random.rand()*2 - 1, True if k!=l else False))
            elif self.hidden1 != 0:
                for j in range(self.outputs, self.outputs + self.hidden1):
                    for k in range(self.outputs):#self.inputs+self.hidden, self.inputs+self.hidden+self.outputs):
                        self.connections[(j, k)] = (Connection((j, k), np.random.rand()*2 - 1, True if j!=k else False))
            else:
                for j in range(-self.inputs, 0):
                    for k in range(0, self.outputs):
                        self.connections[(j, k)] = (Connection((j, k), np.random.rand()*2 - 1, True if j!=k else False))
        else:
            for key in list(connections): 
                enabled = True 
                if key[0]==key[1]:
                    enabled=False 
                self.connections[key] = Connection(key, connections[key].weight, enabled) 

        self.bias_limit()
        self.efficient() 

    def bias_limit(self):
        for node in self.nodes:
            for connection in self.connections:
                if self.connections[connection].key[1] == node:
                    self.nodes[node].sources += 1 

    def sigmoid(self, x):
        return (1 / (1+np.exp(-x)))

    def sig_der(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def activation_function(self, x, bias):
        #return self.sigmoid(x + bias) 
        if x >= bias: 
            return x#self.sigmoid(x) * 2 - 1
        else:
            return 0
        #return (1 / (1+np.exp(-x + bias)))#*2 - 1 

    def activate_der(self, x, bias):
        #result = self.sig_der(x + bias) 
        if x >= bias:
            return 1 
        else:
            return .1
        # result = self.activation_function(x, bias)*(1 - self.activation_function(x, bias)) 
        if result <.5:
            return .5
        else:
            return result 


    def node_output(self, node, input):
        input_nodes = range(-1*self.inputs, 0) 
        if node in input_nodes:
   
            return input[node+self.inputs]
            self.nodes[node].input = input[node+self.inputs]
            result =  self.activation_function(input[node+self.inputs], self.nodes[node].bias) 
            #print("node output: ", node, self.nodes[node].input) 
            return result
        else:
            x = 0 
            # sources = self.sources[node] 
            # for source in sources:
            #     x += self.node_output(source, input) 
            for connection in self.connections:
                connection = self.connections[connection] 
                if connection.key[1] == node  and connection.enabled==True:
                    if connection.key[0]==connection.key[1]:
                        x += self.nodes[node].output 
                    else:
                        x += self.node_output(connection.key[0], input)*connection.weight 
            
            self.nodes[node].input = x

            result = self.activation_function(x, self.nodes[node].bias) 
            self.nodes[node].output = result 
            return result 
            # print("node: ", node)
            # print("input: ", self.nodes[node].input)
            
    


    def activate(self, input):      
        return [self.node_output(i, input) for i in range(self.outputs)] 

    def delta(self, node, error):   #error a 1d array of errors
        if self.nodes[node].key in range(self.outputs):       #i think this now works for multiple outputs 
            self.nodes[node].delta = error[node] #* self.activate_der(self.nodes[node].input, self.nodes[node].bias)  
            return error[node] #* self.activate_der(self.nodes[node].input, self.nodes[node].bias)  
        else:
            net_delta = 0
            for connection in self.connections:
                connection = self.connections[connection] 
                if connection.enabled == False:
                    continue 
                if connection.key[0] == self.nodes[node].key:
                    next = self.nodes[connection.key[1]]  
                    if False:#next.delta != -1 :
                        addend = connection.weight*next.delta*self.activation_function(next.input, next.bias)#self.activate_der(next.input, next.bias) 
                        net_delta += addend 
                        # if addend==0:
                        #     print(connection.weight,self.activation_function(next.input),(1-self.activation_function(next.input)),next.delta) 
                    else:
                        net_delta += connection.weight*self.delta(next.key, error)*self.activation_function(next.input, next.bias)#self.activate_der(next.input, next.bias) 

            self.nodes[node].delta = net_delta#*self.activate_der(self.nodes[node].input, self.nodes[node].bias) 
            
            return net_delta#*self.activate_der(self.nodes[node].input, self.nodes[node].bias)  

    

    def backpropogate(self, error):
        #reset deltas
        for node in self.nodes:
            self.nodes[node].delta = -1
            
        #calculate deltas:
        for node in self.nodes:
            self.delta(node, error) 
        #calculate partials:
        #self.partials() 
     
        #update weights 
        for connection in self.connections:
            connection = self.connections[connection] 
            i = connection.key[0]
            j = connection.key[1] 
            addend = self.learningrate*self.nodes[j].delta*self.activation_function(self.nodes[i].input, self.nodes[i].bias) 
          
            if addend == 0:     #seems to work now but may need revision for more complex problems 
                delta = self.delta(j, error) 
                connection.weight += self.learningrate*delta 
            else:
                connection.weight += addend
      
            # if connection.weight > 1:
            #     connection.weight = 1
            # elif connection.weight < -1:
            #     connection.weight = -1 
            
        
        #update biases
        # for node in range(len(self.nodes)-self.inputs):#self.nodes:
        #     node_output = (self.activation_function(self.nodes[node].input, self.nodes[node].bias))
        #     addend = self.learningrate*self.nodes[node].delta * node_output# - self.nodes[node].input)#self.nodes[node].delta#*self.activate_der(self.nodes[j].input)#self.learningrate*self.nodes[i].delta*self.activation_function(self.nodes[i].input)
        #     if True:#addend == 0:# and error != 0:
        #         #delta = self.delta(node, error) 
        #         self.nodes[node].bias += -1*self.learningrate*self.nodes[node].delta*self.activate_der(self.nodes[node].input, self.nodes[node].bias) 
          
        #         # if node_output==0 and self.nodes[node].input > 0:
        #         #     self.nodes[node].bias -= self.learningrate*delta 
        #         # elif node_output==0 and self.nodes[node].input < 0:
        #         #     self.nodes[node].bias += self.learningrate*delta 
        #         #self.nodes[node].bias *= -1 
        #     elif addend < 0:
        #         self.nodes[node].bias += addend 
           

            # if self.nodes[node].bias < -self.nodes[node].sources:
            #     self.nodes[node].bias = -self.nodes[node].sources 
            # elif self.nodes[node].bias > self.nodes[node].sources:
            #     self.nodes[node].bias = self.nodes[node].sources  

        #reset deltas
        for node in self.nodes:
            self.nodes[node].delta = -1

        #print('(-1, 0): ', self.connections[(-1, 0)].weight) 

    def efficient(self):      #for more efficient calculations 
        layer1 = []
        layer2 = []
       
        for c in self.connections:
            if c[0] in range(-self.inputs, 0):
                layer1.append(c[1]) 
            elif c[1] in range(self.outputs):
                layer2.append(c[0]) 

        self.layer1 = layer1
        self.layer2 = layer2 


def train(net, sample_set, answer_set):
    net_start = copy.deepcopy(net) 
    best_net = copy.deepcopy(net)
    counter = 0
    error = 1
    total_error = 1  
    error_best = None
    while abs(total_error) > .2 and counter < 10:#for i in range(1000):
        predictions = []
        actuals = [] 
        total_error = 0 
        size = len(sample_set)
        if size==0:
            continue  
        for i in range(size): 
        
            size = len(sample_set) 
            input1 = i

            output = net.activate(sample_set[input1])    #for uni-output nets 

            #output_best = best_net.node_output(0, sample_set[input1]) 
      

            predictions.append(output) 
            actuals.append(answer_set[input1])
            if answer_set[input1] == 1:
                correct = np.array([1, 0, 0])
            elif  answer_set[input1] == -1:
                correct = np.array([0, 1, 0]) 
            else:
                correct = np.array([0, 0, 1]) 

            error = correct - output 
            
            net.backpropogate(error)
            
            e = np.sum(error) 
            total_error += e 
        total_error = total_error / size 
        #print(total_error) 
        if error_best is None:
            error_best = total_error 
        if total_error < error_best:
            best_net = copy.deepcopy(net) 
            error_best = total_error 
       
        counter += 1
       
  
    return net, error_best #before returned best net

def train2(net, sample_set, answer_set, batchsize=10):
    net_start = copy.deepcopy(net) 
    best_net = copy.deepcopy(net)
    counter = 0
    error = 1
    avg_error = np.array([1, 1])   
    error_best = None 
    while all(np.absolute(avg_error) > .2) and counter < len(sample_set):#for i in range(1000):
        predictions = []
        actuals = [] 
        total_error = []
        samples = sample_set[counter:counter+batchsize]
        answers = answer_set[counter:counter+batchsize] 
        size = len(samples)
        if size==0:
            continue 
        e = []
        for i in range(size): 
        
            input1 = i
            
            output = net.activate(samples[input1])    #for uni-output nets 

            #output_best = best_net.node_output(0, sample_set[input1]) 
            

            predictions.append(output) 
            actuals.append(answers[input1])
            # if answers[input1] == 1:
            #     correct = np.array([1, 0])

            # else:
            #     correct = np.array([0, 1]) 

            error = answers[input1] - output#correct - output 
         
            total_error.append(np.absolute(error)) 
            net.backpropogate(error)
            
            e.append(error) 
            #total_error += e 
        e = np.array(e) 
        total_error = np.array(total_error) 
        avg_total = np.average(total_error, axis=0) 
        avg_error = np.average(e, axis=0) 
       
        avg_input = np.average(samples, axis=0) 
        net.activate(avg_input) 
        #net.backpropogate(avg_error) 
        
        # total_error = np.square(e) 
        # total_error = np.sum(total_error) / 2
        #print(avg_total) 
        if error_best is None:
            error_best = avg_total 
        if np.linalg.norm(total_error) < np.linalg.norm(error_best):
            best_net = copy.deepcopy(net) 
            error_best = avg_total 
       
        counter += batchsize
       
  
    return best_net, avg_total  #before returned best net


def GD(net, sample_set, answer_set):
    #net = FNN(inputs, outputs, hidden1, hidden2, nodes, connections)
    #print(net.connections[(0,0)].__dict__) 
    best_net, error_best = train(net, sample_set, answer_set) 
    return best_net, error_best  

def GD2(net, sample_set, answer_set, batchsize):
    #net = FNN(inputs, outputs, hidden1, hidden2, nodes, connections)
    #print(net.connections[(0,0)].__dict__) 
    best_net, error_best = train2(net, sample_set, answer_set, batchsize) 
    return best_net, error_best  








