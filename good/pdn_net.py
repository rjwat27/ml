import numpy as np

import PhaseDomainNeuron as pdn







class PDN_Network():
    def __init__(self):
        self.ninputs = 0
        self.noutputs = 0

        self.input_layer = []

        self.hidden = []

        self.output_layer = []

        self.fanout = 0 
        self.learning_rate = .01 

        '''experimental PID controller parameters for variable learning rate'''
        self.E = 1
        self.I = .1
        self.D = .1

        self.connections = {} 
        self.neurons = {} 


        #plotting tools
        self.STREAM_SIZE = int(10e3) 
        self.input_stream = [0 for i in range(self.STREAM_SIZE)] 
        self.output_stream = [0 for i in range(self.STREAM_SIZE)] 

    '''sets up all layers with neuron objects
       and initializes random biases and weights
       between layers'''

    def configure(self, params):
        '''pass dict of params for network'''
        self.ninputs = params['ninputs']
        self.noutputs = params['noutputs']
        self.fanout = params['fanout'] 

        key = 1

        #create layers

        self.input_layer = [pdn.PDN(vref = 150, key=key+i) for i in range(self.ninputs)] 

        key += self.ninputs 
        
        tmp = params['hidden'] 
        
        for t in tmp:
            self.hidden.append([pdn.PDN(vref = np.random.rand()*340, key=key+i) for i in range(t)])
            key += t  

        self.output_layer = [pdn.PDN(vref = np.random.rand()*340, key=key+i) for i in range(self.noutputs)] 

        #aggregate all neurons in all layers
        for n in self.input_layer + self.output_layer:
            self.neurons[n.key] = n 
        for h in self.hidden:
            for n in h:
                self.neurons[n.key] = n 

        #create network connections between layers
        #does not consider fanout limitations yet
        if len(self.hidden) > 0:
            for i in self.input_layer:
                for h in self.hidden[0]:
                    self.connections[(i.key, h.key)] = 1*np.random.rand()

            for i in range(len(self.hidden)-1):
                h1 = self.hidden[i]
                h2 = self.hidden[i+1]
                for i in h1:
                    for j in h2:
                        self.connections[(i.key, j.key)] = 1*np.random.rand()

            for h in self.hidden[-1]:
                for j in self.output_layer:
                    self.connections[(h.key, j.key)] = 1*np.random.rand()
        else:
            for i in self.input_layer:
                for j in self.output_layer:
                    self.connections[(i.key, j.key)] = 1*np.random.rand()

    def import_weights(self, weights, fanout_codes):
        '''i really, really hope this works'''
        temp = {}
        layer_sizes = [self.ninputs] + self.hidden + [self.noutputs] 
        for i in range(self.ninputs):
            for j in fanout_codes[0][i]:
                n1 = self.input_layer[i]
                n2 = self.hidden[0][j]
                temp[n1.key, n2.key] = weights[0][i][j]
        for i in range(len(self.hidden)-1):
            for j in range(len(self.hidden[i])):
                for k in fanout_codes[i][j]:
                    n1 = self.hidden[i][j]
                    n2 = self.hidden[i+1][k]
                    temp[n1.key, n2.key] = weights[i][j][k] 

        for i in range(len(self.hidden[-1])):
            for j in fanout_codes[-1][i]:
                n1 = self.hidden[-1][i]
                n2 = self.output_layer[j]
                temp[n1.key, n2.key] = weights[-1][i][j] 
          
        self.connections = temp 

    '''pushes inputs onto input neurons and
       pulls output from output neurons'''

    def forward(self, input_vector):
        self.input_stream.append(input_vector)
        self.input_stream.pop(0) 
        for i in range(self.ninputs):
            self.input_layer[i].forward(input_vector[i])

        for c in self.connections:
            n1 = self.neurons[c[0]]
            n2 = self.neurons[c[1]] 
            o = (n1.output()) * self.connections[c] 
            n2.forward(o) 

        output = [n.output() for n in self.output_layer] 

        self.output_stream.append(output)
        self.output_stream.pop(0) 

        return output


    '''pushes feedback onto the output neurons'''

    def backward(self, feedback_vector):
        for i in range(self.noutputs):
            self.output_layer[i].backward(feedback_vector[i]) 

        for c in self.connections:
            n1 = self.neurons[c[0]]
            n2 = self.neurons[c[1]] 
            o = abs(n2.backpropagate())*self.connections[c] 
            #print(feedback_vector[0], o)
            n1.backward(o) 

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x)) 

    def tweak_learning_rate(self):
        error = np.array([n.feedback_stream[-100:-1] for n in self.output_layer])

        error = np.sum(error, axis=0)
        total = abs(np.sum(error)) 

        diff = error[-98:-1] - error[-99:-2] 
  
        dev = np.std(diff) 

        self.learning_rate += -1*self.sigmoid(dev)*self.learning_rate + total*self.I 


    '''Tick all neurons in the network'''

    def tick_network(self):
        for n in self.neurons:
            self.neurons[n].tick() 

        '''weights not updated since imported externally'''
        # for c in self.connections:
        #     n1 = self.neurons[c[0]]
        #     n2 = self.neurons[c[1]]

        #     self.connections[c] += n1.output() * n2.backpropagate() * self.learning_rate  






























