import numpy as np
#import PhaseDomainNeuron as pdn 
xor_inputs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]] 
xor_outputs = [0, 1, 1, 0, 1, 0, 0, 1]  

def relu(x):
    return np.maximum(0, np.arctan(x)) 
def sig(x):
    return relu(x) 
    return 1/(1+np.exp(-x)) 
def sig_der(x):
    return 1 if x > 0 else .1#(1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x))) 
    return sig(x)*(1-sig(x)) 


  


class actual_fanout_layer():
    '''simple one hidden layer network'''
    def __init__(self, inputs, outputs, learning_rate=.1, fanout=0, activation_function=sig):
        self.ninputs = inputs
        self.noutputs = outputs
        
        self.fanout = fanout 

        self.activation_function = activation_function

        self.input = 0
        self.input1 = 0
        

        self.biases1 = np.array([-inputs/2 + i*inputs/outputs for i in range(self.noutputs)])



        self.delta1 = np.zeros(self.noutputs)

        self.max_change = 0 
        #TODO for many layers may be wise and even very effective to test convergence on individual 
        #layers and update them individually on a rollover signal from the following layer-
        #for now just a single max_change for global convergence, though this may drastically increase learning times as global convergence
        #on large mult-layer networks will take a while

        self.learning_rate = learning_rate

        self.w1 = np.array([[np.random.rand()*2-1 for i in range(self.noutputs)] for j in range(self.ninputs)])
       


        '''set up fanout weight connections'''
        '''for now just a random wiring, TODO option to link manually or with patterned automation instead'''

        layer1 = range(self.noutputs) 
        if self.fanout < self.noutputs and self.fanout!=0:
            self.fanout_encoding1 = [np.random.choice(layer1, size=self.fanout, replace=False) for i in range(self.ninputs)]
        else:
            self.fanout_encoding1 = [layer1 for j in range(self.ninputs)] 

        self.feature_similarity_threshold = .9 #arbitario

        self.back_signal = [0 for i in range(self.ninputs)] 


    def activate(self, input):
        self.input = input
        self.input1 = np.zeros(self.noutputs) 
        for i in range(self.ninputs):
            for j in self.fanout_encoding1[i]:
                self.input1[j] += self.w1[i][j]*input[i]
        
        output= self.activation_function(self.input1 + self.biases1)
        self.output = output 
        return output 

        
    def delta(self, feedback):
        '''update learning rate''' 
        self.learning_rate = .01
        bias_changes = []

        for i in range(self.noutputs):
            self.delta1[i] = feedback[i]
            bias_changes.append(self.learning_rate*feedback[i]*self.output)

        '''how to reconcile custom activation function with sig_der?'''
        ders = np.array([sig_der(self.output[i])*feedback[i] for i in range(self.noutputs)])
        
        self.back_signal = [np.sum([ders[j]*self.w1[i][j] for j in self.fanout_encoding1[i]]) for i in range(self.ninputs)]  


        return max(bias_changes[0])  
        
    def backpropogate(self):
        return self.back_signal 

    def adjust(self):
        weight_changes = []
    
        for i in range(self.ninputs):
            for j in range(self.noutputs):
                weight_changes.append(self.learning_rate*self.delta1[j]*self.input[i])             
                self.w1[i][j] += self.learning_rate*self.delta1[j]*self.input[i] 

        self.max_change = max(weight_changes)
    
        return self.max_change

    def adjust_biases(self):
        for i in range(self.noutputs):
            self.biases1[i] += -self.learning_rate*self.delta1[i]*self.output[i]


    def add_hidden_node(self, num=1):
        if self.noutputs > 100: #remove?
            return 
        for g in range(num):
            average_bias_space = np.random.rand()*2*self.ninputs - self.ninputs
            new_weights = np.array([np.random.rand()*2 - 1  for i in range(self.ninputs)]) 

            self.w1 = np.append(self.w1.T, np.array([new_weights]), axis=0).T 
            self.biases1 = np.append(self.biases1, np.array([average_bias_space]), axis=0)

            self.delta1 = np.append(self.delta1, np.array([0]), axis=0) 


    def prune_worst(self, external_weights, num=1):
        m = None 
        if num < 1:
            print("a;lsdkjf;asldjldjf")
            input()
        if True:#for j in range(num):
            relevance_scores = []
            for i in range(self.noutputs):
                temp = self.w1.T 

                if np.max(np.absolute(external_weights[i])) < .01 / (.01 + relu(np.sum(np.absolute(temp[i]))+self.biases1[i])):
                    relevance_scores.append(np.max(np.absolute(external_weights[i])))

            if not relevance_scores:
                return None 
    
            m = np.argmin(relevance_scores) 

            self.w1 = np.delete(self.w1.T, m, axis=0).T 
            self.biases1 = np.delete(self.biases1, m, axis=0)

        '''this really only makes sense with one replacement at a time'''
        return m #other external layers need this information   

    def prune_weights(self, m):
        self.w1 = np.delete(self.w1, m, axis=0)    

    def add_weights(self):
        new_weights = np.array([np.random.rand()*2 for i in range(self.noutputs)])

        self.w1 = np.append(self.w1, np.array([new_weights]), axis=0)

    def vectorize_node_weights(self, node):
        vector = np.array([self.w1[i, node] for i in range(self.ninputs)]) 
        return (vector / np.linalg.norm(vector)) #normalize 

    def is_converged(self):
        return (self.max_change < .01) 


class fanout_network():
    '''layer sizes includes the input size and output size'''
    '''layers denotes the number of input-set weights into a neuron vector, which equals (num_hidden + output_layer)'''
    '''therefore, the layer_sizes will be one greater than layers'''
    def __init__(self, inputs, outputs, layers, layer_sizes=[], fanout=0, learning_rate=.1):
        self.ninputs = inputs
        self.noutputs = outputs 
        self.layers = layers 
        self.layer_sizes = layer_sizes
        self.fanout = fanout 
        self.learning_rate = learning_rate

        self.max_change = 0


        if not layer_sizes:
            print("ya yer gonna need some data on the layer sizes") 
            return 
        else:
            self.hidden_layers = [actual_fanout_layer(self.layer_sizes[i], self.layer_sizes[i+1], self.learning_rate, self.fanout) for i in range(self.layers)]

    def import_weights(self, weights, fanouts):
        for i in range(len(weights)):
            self.hidden_layers[i].w1 = weights[i]
            self.hidden_layers[i].fanout_encoding1 = fanouts[i] 

    def import_biases(self, biases):
        for i in range(len(biases)):
            self.hidden_layers[i].biases1 = biases[i] 
   

    def activate(self, input1):
        for l in self.hidden_layers:
            input1 = l.activate(input1)

        return input1 
        
    def delta(self, feedback):
        for l in range(1, self.layers+1):
            self.hidden_layers[-l].delta(feedback)
            feedback = self.hidden_layers[-l].backpropogate() 

    def adjust(self):
        for l in range(1, self.layers):
            self.hidden_layers[l].adjust()

    def adjust_bias(self):
        for l in range(self.layers):
            self.hidden_layers[l].adjust_biases() 

    def evolve(self, force=False):
        for l in range(1, self.layers):
            if self.hidden_layers[l].is_converged() or force:
                external_weights = self.hidden_layers[l].w1
                # if max(1, int(np.log10(self.hidden_layers[l-1].noutputs))) != 1:
                #     print("invalid number of prune neurons entered to prune_worst")
                #     print(max(1, int(np.log10(self.hidden_layers[l-1].noutputs))), l)
                #     input()
                n = self.hidden_layers[l-1].prune_worst(external_weights, num=1)  #arbitrario
                # if (self.hidden_layers[l-1].noutputs - len(self.hidden_layers[l-1].biases1)) <= 0:
                #     print('ERROR REPLACING NODE, NON-POSITIVE INPUT GIVEN')
                #     input() 
                
                if n!=None:
                    self.hidden_layers[l].prune_weights(n) 
                
                    self.hidden_layers[l-1].add_hidden_node(self.hidden_layers[l-1].noutputs - len(self.hidden_layers[l-1].biases1))
                    self.hidden_layers[l].add_weights() 


class actual_fanout_layer_with_neuron_structures():
    '''simple one hidden layer network'''
    def __init__(self, inputs, outputs, learning_rate=.1, fanout=0, neural_object=None):
        if neural_object==None:
            print("what are you thinking, if no provide object, why not use the simpler version?")
            return
        self.ninputs = inputs
        self.noutputs = outputs
        
        self.fanout = fanout 

        self.neural_object = neural_object

        self.input = 0
        self.input1 = 0
        

        #self.biases1 = np.array([-inputs/2 + i*inputs/outputs for i in range(self.noutputs)])
        '''neural object should accept a 'bias' parameter'''
        self.nodes = np.array([self.neural_object(-inputs/2 + i*inputs/outputs) for i in range(self.noutputs)])


        self.delta1 = np.zeros(self.noutputs)

        self.max_change = 0 
        #TODO for many layers may be wise and even very effective to test convergence on individual 
        #layers and update them individually on a rollover signal from the following layer-
        #for now just a single max_change for global convergence, though this may drastically increase learning times as global convergence
        #on large mult-layer networks will take a while

        self.learning_rate = learning_rate

        self.w1 = np.array([[np.random.rand()*2-1 for i in range(self.noutputs)] for j in range(self.ninputs)])
       


        '''set up fanout weight connections'''
        '''for now just a random wiring, TODO option to link manually or with patterned automation instead'''

        layer1 = range(self.noutputs) 
        if self.fanout < self.noutputs and self.fanout!=0:
            self.fanout_encoding1 = [np.random.choice(layer1, size=self.fanout, replace=False) for i in range(self.ninputs)]
        else:
            self.fanout_encoding1 = [layer1 for j in range(self.ninputs)] 

        self.feature_similarity_threshold = .9 #arbitario

        self.back_signal = [0 for i in range(self.ninputs)] 


    def activate(self, input):
        self.input = input
        self.input1 = np.zeros(self.noutputs) 
        for i in range(self.ninputs):
            for j in self.fanout_encoding1[i]:
                self.input1[j] += self.w1[i][j]*input[i]
        
        '''neuron structure needs a 'forward' method'''
        output= [self.nodes[n].forward(self.input1[n]) for n in range(self.noutputs)] 
        self.output = output 
        return output 

        
    def delta(self, feedback):

        bias_changes = []

        for i in range(self.noutputs):
            self.delta1[i] = feedback[i]
            bias_changes.append(self.learning_rate*feedback[i]*self.output[i])

        for n in range(self.noutputs):
            '''nodes structure need 'backward' method'''
            self.nodes[n].backward(feedback[n]) 

        '''how to reconcile custom activation function with derivative of activation when hard to define or non-existant?'''
        ders = np.array([sig_der(self.output[i])*feedback[i] for i in range(self.noutputs)])
       

        self.back_signal = [np.sum([ders[j]*self.w1[i][j] for j in self.fanout_encoding1[i]]) for i in range(self.ninputs)]  


        return max(bias_changes)  
        
    def backpropogate(self):
        return self.back_signal 

    def adjust(self):
        weight_changes = []
    
        for i in range(self.ninputs):
            for j in range(self.noutputs):
                weight_changes.append(self.learning_rate*self.delta1[j]*self.input[i])             
                self.w1[i][j] += self.learning_rate*self.delta1[j]*self.input[i] 

        self.max_change = max(weight_changes)
    
        return self.max_change



    def add_hidden_node(self, num=1):
        if self.noutputs > 100: #remove?
            return 
        for g in range(num):
            average_bias_space = np.random.rand()*2*self.ninputs - self.ninputs
            new_weights = np.array([np.random.rand()*2 - 1  for i in range(self.ninputs)]) 

            self.w1 = np.append(self.w1.T, np.array([new_weights]), axis=0).T 
            #self.biases1 = np.append(self.biases1, np.array([average_bias_space]), axis=0)
            self.nodes = np.append(self.nodes, self.neural_object(average_bias_space), axis=0)

            self.delta1 = np.append(self.delta1, np.array([0]), axis=0) 


    def prune_worst(self, external_weights, num=1):
        m = None 
        if num < 1:
            print("a;lsdkjf;asldjldjf")
            input()
        if True:#for j in range(num):
            relevance_scores = []
            temp = self.w1.T 
            for i in range(self.noutputs):
                
                if np.max(np.absolute(external_weights[i])) < .01 / (.01 + self.noutputs):
                    relevance_scores.append(np.max(np.absolute(external_weights[i])))

            if not relevance_scores:
                return None 
    
            m = np.argmin(relevance_scores) 

            self.w1 = np.delete(self.w1.T, m, axis=0).T 
            self.nodes = np.delete(self.nodes, m, axis=0)

        '''this really only makes sense with one replacement at a time'''
        return m #other external layers need this information   

    def prune_weights(self, m):
        self.w1 = np.delete(self.w1, m, axis=0)    

    def add_weights(self):
        new_weights = np.array([np.random.rand()*2 for i in range(self.noutputs)])

        self.w1 = np.append(self.w1, np.array([new_weights]), axis=0)

    def vectorize_node_weights(self, node):
        vector = np.array([self.w1[i, node] for i in range(self.ninputs)]) 
        return (vector / np.linalg.norm(vector)) #normalize 

    def tick_neurons(self):
        for n in self.nodes:
            n.tick() 

    def is_converged(self):
        return (self.max_change < .01) 


class fanout_network_with_neuron_structures():
    '''layer sizes includes the input size and output size'''
    '''layers denotes the number of input-set weights into a neuron vector, which equals (num_hidden + output_layer)'''
    '''therefore, the layer_sizes will be one greater than layers'''
    def __init__(self, inputs, outputs, layers, layer_sizes=[], fanout=0, learning_rate=.1):
        self.ninputs = inputs
        self.noutputs = outputs 
        self.layers = layers 
        self.layer_sizes = layer_sizes
        self.fanout = fanout 
        self.learning_rate = learning_rate

        self.max_change = 0


        if not layer_sizes:
            print("ya yer gonna need some data on the layer sizes") 
            return 
        else:
            self.hidden_layers = [actual_fanout_layer_with_neuron_structures(self.layer_sizes[i], self.layer_sizes[i+1], self.learning_rate, self.fanout, neural_object=pdn.PDN) for i in range(self.layers)]

    def activate(self, input1):
        for l in self.hidden_layers:
            input1 = l.activate(input1)

        return input1 
        
    def delta(self, feedback):
        for l in range(1, self.layers+1):
            self.hidden_layers[-l].delta(feedback)
            feedback = self.hidden_layers[-l].backpropogate() 

    def adjust(self):
        for l in range(1, self.layers):
            self.hidden_layers[l].adjust()

    def evolve(self, force=False):
        for l in range(1, self.layers):
            if self.hidden_layers[l].is_converged() or force:
                external_weights = self.hidden_layers[l].w1
                # if max(1, int(np.log10(self.hidden_layers[l-1].noutputs))) != 1:
                #     print("invalid number of prune neurons entered to prune_worst")
                #     print(max(1, int(np.log10(self.hidden_layers[l-1].noutputs))), l)
                #     input()
                n = self.hidden_layers[l-1].prune_worst(external_weights, num=1)  #arbitrario
                # if (self.hidden_layers[l-1].noutputs - len(self.hidden_layers[l-1].biases1)) <= 0:
                #     print('ERROR REPLACING NODE, NON-POSITIVE INPUT GIVEN')
                #     input() 
                
                if n!=None:
                    self.hidden_layers[l].prune_weights(n) 
                
                    self.hidden_layers[l-1].add_hidden_node(self.hidden_layers[l-1].noutputs - len(self.hidden_layers[l-1].biases1))
                    self.hidden_layers[l].add_weights() 

    def tick_neurons(self):
        for l in self.hidden_layers:
            l.tick_neurons() 

