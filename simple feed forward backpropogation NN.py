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

# file = open("training_data2.txt", 'r') 
# data = json.load(file) 



# data_list = [] 
# for i in range(batchsize):
#     data_list.append(json.loads(data[str(i)]))
#     wallet = 100 
# prices_high = []
# prices_low = [] 
# volumes = []
# counts = [] 
# for tick in data_list:
#     change = float(tick['priceChangePercent']) 
#     top = float(tick['highPrice']) 
#     low = float(tick['lowPrice'])
#     volume = float(tick['volume']) 
#     count =  float(tick['count'])
#     prices_high.append(top)
#     prices_low.append(low)
#     volumes.append(volume)
#     counts.append(count) 
#     if change > 0:
#         wallet = wallet*((change/100)+1)
# max = wallet 
# #normalize 
# prices_high_bounds = (np.max(prices_high), np.min(prices_high))
# prices_low_bounds = (np.max(prices_low), np.min(prices_low))  
# volumes_bounds = (np.max(volumes), np.min(volumes))
# counts_bounds = (np.max(counts), np.min(counts)) 
# for tick in data_list:
#     top = float(tick['highPrice']) 
#     low = float(tick['lowPrice'])
#     volume = float(tick['volume']) 
#     count =  float(tick['count'])

#     tick['highPrice'] = (top - prices_high_bounds[1])/(prices_high_bounds[0] - prices_high_bounds[1]) 
#     tick['lowPrice'] = (low - prices_low_bounds[1])/(prices_low_bounds[0] - prices_low_bounds[1])
#     tick['volume'] = (volume - volumes_bounds[1])/(volumes_bounds[0] - volumes_bounds[1]) 
#     tick['count'] = (count - counts_bounds[1])/(counts_bounds[0] - counts_bounds[1])  

class FNN():
    def __init__(self, ninputs, noutputs, hidden_layer1, hidden_layer2, connections):
        self.inputs = ninputs
        self.outputs = noutputs 
        self.hidden1 = hidden_layer1
        self.hidden2 = hidden_layer2 
        self.connections = {} 
        self.nodes = {}
        self.sources = {} 
        self.learningrate = 2

        class Node:
            def __init__(self, key, bias):
                self.key = key
                self.bias = bias 
                self.input = 0
                self.partial = 1 
                self.sources = 0 
                self.delta = -1 

        class Connection:
            def __init__(self, key, weight, enabled):
                self.key = key
                self.weight = weight
                self.partial = 1 
                self.enabled = enabled 
        
        node_list = [Node(key, 0) for key in range(-self.inputs, self.hidden2 + self.hidden1 + self.outputs)] #biases initialized at 0 
        for node in node_list:
            self.nodes[node.key] = node 
        if len(connections)==0:
            count = 0 
            for i in range(-self.inputs, 0):
                for j in range(self.outputs, self.outputs + self.hidden1):
                    self.connections[(i, j)] = (Connection((i, j), np.random.rand()*2 - 1, True if i!=j else False))
                    count += 1
            if self.hidden2 != 0:
                for j in range(self.outputs, self.outputs + self.hidden1):
                    for k in range(self.outputs + self.hidden1, self.outputs+self.hidden1+self.hidden2):#self.inputs+self.hidden, self.inputs+self.hidden+self.outputs):
                        self.connections[(j, k)] = (Connection((j, k), np.random.rand()*2 - 1, True if j!=k else False)) 
                for k in range(self.outputs + self.hidden1, self.outputs+self.hidden1+self.hidden2):
                    for l in range(self.outputs): 
                        self.connections[(k, l)] = (Connection((k, l), np.random.rand()*2 - 1, True if k!=l else False))
            else:
                for j in range(self.outputs, self.outputs + self.hidden1):
                    for k in range(self.outputs):#self.inputs+self.hidden, self.inputs+self.hidden+self.outputs):
                        self.connections[(j, k)] = (Connection((j, k), np.random.rand()*2 - 1, True if j!=k else False))
        else:
            self.connections = connections 

        self.bias_limit()

    def bias_limit(self):
        for node in self.nodes:
            for connection in self.connections:
                if self.connections[connection].key[1] == node:
                    self.nodes[node].sources += 1 

    def sigmoid(x):
        return (1 / (1+np.exp(-x)))

    def sig_der(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def activation_function(self, x, bias):
        if x >= bias: 
            return abs(x)
        else:
            return 0 
        #return (1 / (1+np.exp(-x + bias)))#*2 - 1 

    def activate_der(self, x, bias):
        if x >= bias:
            return 1 
        else:
            return 1
        # result = self.activation_function(x, bias)*(1 - self.activation_function(x, bias)) 
        # if result <.1:
        #     return .1
        # else:
        #     return result 


    def node_output(self, node, input):
        input_nodes = range(-1*self.inputs, 0) 
        if node in input_nodes:
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
                    
                    x += self.node_output(connection.key[0], input)*connection.weight 
            
            self.nodes[node].input = x
            
            result = self.activation_function(x, self.nodes[node].bias) 
            return result 
            # print("node: ", node)
            # print("input: ", self.nodes[node].input)
            
    


    def activate(self, input):      #for now this only accommadates nets with 1 output node 
        return self.node_output(0, input) 

    def delta(self, node, error):
        sources = 0 
        if self.nodes[node].key == 0:       #only fitted for single output
            self.nodes[node].delta = error * self.activate_der(self.nodes[0].input, self.nodes[0].bias)  
            return error * self.activate_der(self.nodes[0].input, self.nodes[0].bias)  
        else:
            net_delta = 0
            for connection in self.connections:
                connection = self.connections[connection] 
                if connection.enabled == False:
                    continue 
                if connection.key[0] == self.nodes[node].key:
                    next = self.nodes[connection.key[1]]  
                    if next.delta != -1 :
                        addend = connection.weight*next.delta
                        net_delta += addend 
                        # if addend==0:
                        #     print(connection.weight,self.activation_function(next.input),(1-self.activation_function(next.input)),next.delta) 
                    else:
                        net_delta += connection.weight*self.delta(next.key, error)

            self.nodes[node].delta = net_delta*self.activate_der(self.nodes[node].input, self.nodes[node].bias) 
            
            return net_delta#*self.activate_der(self.nodes[node].input, self.nodes[node].bias)  

    def partials(self):
        for node in self.nodes:
            self.nodes[node].partial = 0 
        for connection in self.connections:
            connection = self.connections[connection] 
            product = self.activate_der(self.nodes[0].input) 
            start_node = 0  #adjust for multiple hidden layers 
            stop = False 
            next_layer = [0] 
            for node in next_layer:
                for conn in self.connections:
                    conn = self.connections[conn] 
                    if conn.key[1] == node:
                        next_layer.append(conn.key[0])#self.activate_der(self.nodes[conn[0]].input)) 
                    if conn.key[0] == connection.key[0]:
                        next_layer.clear() 
                        stop = True 
                        break 
                if not stop:
                    for node in next_layer:
                        product *= self.activate_der(self.nodes[node].input)*self.connections[(node, start_node)].weight 
                else:
                    break 
            bias_product = product        
            product *= self.activation_function(self.nodes[connection.key[0]].input) 
            bias_product *= self.activate_der(self.nodes[connection.key[0]].input)*connection.weight 

            self.nodes[connection.key[0]].partial += bias_product 
            connection.partial = product 

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
            addend= -self.learningrate*self.nodes[j].delta*self.activation_function(self.nodes[i].input, self.nodes[i].bias) 
            if addend == 0:     #seems to work now but may need revision for more complex problems 
                delta = self.delta(j, error) 
                connection.weight += -self.learningrate*delta 
            else:
                connection.weight += -self.learningrate*self.nodes[j].delta*self.activation_function(self.nodes[i].input, self.nodes[i].bias) 
            if connection.weight > 1:
                connection.weight = 1
            elif connection.weight < -1:
                connection.weight = -1 
            
        
        #update biases
        for node in self.nodes:
            addend = self.learningrate*self.nodes[node].delta * (self.activation_function(self.nodes[node].input, self.nodes[node].bias))# - self.nodes[node].input)#self.nodes[node].delta#*self.activate_der(self.nodes[j].input)#self.learningrate*self.nodes[i].delta*self.activation_function(self.nodes[i].input)
            if addend == 0:# and error != 0:
                delta = self.delta(node, error) 
                if True:#self.nodes[node].bias >= 0:
                    self.nodes[node].bias -= self.learningrate*delta 
                else:
                    self.nodes[node].bias += self.learningrate*delta 
                #self.nodes[node].bias *= -1 
            elif addend < 0:
                self.nodes[node].bias += addend 

            if self.nodes[node].bias < -self.nodes[node].sources:
                self.nodes[node].bias = -self.nodes[node].sources 
            elif self.nodes[node].bias > self.nodes[node].sources:
                self.nodes[node].bias = self.nodes[node].sources  

        #reset deltas
        for node in self.nodes:
            self.nodes[node].delta = -1

# net = FNN(2, 1, 2, 2, []) 
# net.nodes[-1].bias = 0
# net.nodes[-2].bias = 0 
# net.nodes[1].bias = 0.5
# net.nodes[0].bias = -1.5
# net.connections[(-1, 1)].weight = 1
# net.connections[(-2, 1)].weight = 1
# net.connections[(1, 0)].weight = -1
# net_nodes_minus1_biasstart = net.nodes[-1].bias
# net_nodes_minus2_biasstart = net.nodes[-2].bias 
# net_nodes_1biasstart = net.nodes[1].bias
# net_nodes_0biasstart = net.nodes[0].bias
# net_connections_minus1_1_start = 1
# net_connections[(-2, 1)].weight = 1
# net_connections[(1, 0)].weight = -1
# average_error = 1
# error_list = [] 
# counter = 0
# error = 1 
# net_start = copy.deepcopy(net) 
# best_net = copy.deepcopy(net) 
# net.nodes[1].bias = 0
# net.nodes[2].bias = 1
# net.nodes[3].bias = 1
# net.nodes[4].bias = 2 
def train(net, sample_set, answer_set):
    net_start = copy.deepcopy(net) 
    best_net = copy.deepcopy(net)
    counter = 0
    error = 1  
    while abs(error) > .1 and counter < 1000:#for i in range(1000):
        predictions = []
        actuals = [] 
        raw_error = 0 
        raw_error_best = 0
        error = 0 
        error_best = 0 
        size = len(sample_set) 
        for i in range(size): 
        
            size = len(xor_inputs) 
            input1 = i#random.choice(range(size)) 
            #print(xor_inputs[input1]) 

            output = net.node_output(0, sample_set[input1])
            output_best = best_net.node_output(0, sample_set[input1]) 
            
            # if output > 0:
            #     input() 

            predictions.append(output) 
            actuals.append(answer_set[input1]) 
            raw_error += answer_set[input1] - output
            raw_error_best += answer_set[input1] - output_best 
            error += (raw_error)**2 / 2
            error_best += (raw_error_best)**2 / 2 
            #error_list.append(abs(error)) 
            # print("node 0 bias: ", net.nodes[0].bias) 
            # print("output: ", output) 
            # print("error: ", error) 
        if error < error_best:
            best_net = copy.deepcopy(net) 
            error_best = error 
        net.backpropogate(raw_error)
        # if len(error_list) > 20:
        #     average_error = np.average(error_list) 
        #     error_list.clear() 
        counter += 1
    #     print("error: ", error) 
    #     print("counter: ", counter) 
        

    # print("\n")
    # print("output: ", output)
    # print("error: ", error) 
    # print("\n\nBest Net:\n") 
    # print(" best error: ", error_best) 
    # print("\nstart: \n") 

    # for connection in net_start.connections:
    #         print("connection: ", net_start.connections[connection].key, net_start.connections[connection].weight) 
    # # print("\n")
    # for node in net_start.nodes:
    #     print("node bias: ", node, net_start.nodes[node].bias) 
    # print("\nafter: \n") 
    # for connection in net.connections:
    #         print("connection: ", net.connections[connection].key, net.connections[connection].weight) 
    # # print("\n")
    # for node in net.nodes:
    #     print("node bias: ", node, net.nodes[node].bias) 
    # print("done")
    return error_best, best_net  


trials = 100 
successes = 0 
for i in range(trials):
    net = FNN(2, 1, 1, 0, []) 
    error_best, net = train(net, xor_inputs, xor_outputs) 
    if error_best < .1:
        successes += 1 

print("Solution Rate:\n", successes / trials) 
for i in range(len(xor_inputs)):
    print(xor_inputs[i])
    print(net.activate(xor_inputs[i]))
    print("\n") 
for connection in net.connections:
    print("connection: ", connection, net.connections[connection].weight) 
for node in net.nodes:
    print("node: ", node, net.nodes[node].bias) 
print("Done.") 



