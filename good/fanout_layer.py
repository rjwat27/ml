import numpy as np 
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


class fanout_layer():
    '''simple one hidden layer network'''
    def __init__(self, inputs, outputs, hidden, learning_rate=.1, fanout=0):
        self.ninputs = inputs
        self.noutputs = outputs
        self.hidden = hidden 
        self.fanout = fanout 

        self.input = 0
        self.input1 = 0
        

        self.biases1 = np.array([-inputs/2 + i*inputs/hidden for i in range(self.hidden)])
        self.biases2 = np.array([-hidden/2 + i*hidden/outputs for i in range(self.noutputs)]) 


        self.delta1 = np.zeros(self.hidden)

        self.max_change = 0 
        #TODO for many layers may be wise and even very effective to test convergence on individual 
        #layers and update them individually on a rollover signal from the following layer-
        #for now just a single max_change for global convergence, though this may drastically increase learning times as global convergence
        #on large mult-layer networks will take a while

    

        self.learning_rate = learning_rate

        self.w1 = np.array([[np.random.rand()*2-1 for i in range(hidden)] for j in range(self.ninputs)])
        self.w2 = np.array([[np.random.rand()*2-1 for i in range(self.noutputs)] for j in range(self.hidden)])


        '''set up fanout weight connections'''
        '''for now just a random wiring, TODO option to link manually or with patterned automation instead'''

        layer1 = range(self.hidden) 
        if self.fanout < self.hidden and self.fanout!=0:
            self.fanout_encoding1 = [np.random.choice(layer1, size=self.fanout, replace=False) for i in range(self.ninputs)]
        else:
            self.fanout_encoding1 = [layer1 for j in range(self.ninputs)] 
        if self.fanout < self.noutputs and self.fanout!=0:

            self.fanout_encoding2 = [np.random.choice(range(self.noutputs)) for i in range(self.hidden)] 
        else:
            self.fanout_encoding2 = [range(self.noutputs) for j in range(self.hidden)] 

        self.feature_similarity_threshold = .9 #arbitario

        self.back_signal = [0 for i in range(self.ninputs)] 


    def activate(self, input):
        self.input = input
        self.input1 = np.zeros(self.hidden) 
        for i in range(self.ninputs):
            for j in self.fanout_encoding1[i]:
                self.input1[j] += self.w1[i][j]*input[i]
        
        self.input2 = sig(self.input1 + self.biases1)
        
        self.input3 = np.zeros(self.noutputs)
        for i in range(self.hidden):
            for j in self.fanout_encoding2[i]:
                self.input3[j] += self.w2[i][j]*self.input2[i] 

        output = sig(self.input3 + self.biases2)
        self.output = output
        return output 
        pass 

    def layer_activate(self, input):
        self.input = input
        self.input1 = np.zeros(self.hidden) 
        for i in range(self.ninputs):
            for j in self.fanout_encoding1[i]:
                self.input1[j] += self.w1[i][j]*input[i]
        
        self.input2 = sig(self.input1 + self.biases1)
        self.output = self.input2
        return self.output

        
    def delta(self, feedback):
        '''update learning rate''' 
        self.learning_rate = .01
        bias_changes = []
        for i in range(self.noutputs):
            bias_changes.append(self.learning_rate*feedback[i]*sig_der(self.input3[i]))

        ders = np.array([sig_der(self.input3[i])*feedback[i] for i in range(self.noutputs)])
        


        feedback2 = [np.sum([ders[j]*self.w2[i][j] for j in self.fanout_encoding2[i]]) for i in range(self.hidden)]  


        if len(feedback2)!=self.hidden:
            print('feedback2 size incorrect') 

        for i in range(self.hidden):
            self.delta1[i] = feedback2[i] 
            bias_changes.append(self.learning_rate*feedback2[i]*sig_der(self.input1[i]))


        return max(bias_changes)  

    def layer_delta(self, feedback):
        if len(feedback)!=len(self.input2):
            print("feedback size does not match output layer. Verify use of uni-layer functionality")
        bias_changes = []
        for i in range(self.hidden):
            bias_changes.append(self.learning_rate*feedback[i]*sig_der(self.input2[i]))

        ders = np.array([sig_der(self.input2[i])*feedback[i] for i in range(self.hidden)])
        


        self.back_signal = [np.sum([ders[j]*self.w1[i][j] for j in self.fanout_encoding1[i]]) for i in range(self.ninputs)]  

        return max(bias_changes)  
        
    def backpropogate(self):
        return self.back_signal 

    def adjust(self, feedback):
        weight_changes = []
        for i in range(self.hidden):
            for j in range(self.noutputs):
                weight_changes.append(self.learning_rate*feedback[j]*self.input2[i]) 
                #print(self.learning_rate*feedback[j]*self.input2[i])             
                self.w2[i][j] += self.learning_rate*feedback[j]*self.input2[i]  
    
        for i in range(self.ninputs):
            for j in range(self.hidden):
                weight_changes.append(self.learning_rate*self.delta1[j]*self.input[i]) 
                #print(self.learning_rate*self.delta1[j]*self.input[i])              
                self.w1[i][j] += self.learning_rate*self.delta1[j]*self.input[i] 
                pass 
    
        return max(weight_changes)

    def layer_adjust(self, feedback):
        weight_changes = []
        for i in range(self.ninputs):
            for j in range(self.hidden):
                weight_changes.append(self.learning_rate*feedback[j]*self.input[i]) 
                #print(self.learning_rate*feedback[j]*self.input2[i])             
                self.w1[i][j] += self.learning_rate*feedback[j]*self.input[i]  
            
        return max(weight_changes)



    def add_hidden_node(self, num=1, use_intelligent_search=False):
        if self.hidden > 100:
            return 
        for g in range(num):
            average_bias_space = np.random.rand()*2*self.ninputs - self.ninputs
            new_weights = np.array([np.random.rand()*2 - 1  for i in range(self.ninputs)]) 

            self.w1 = np.append(self.w1.T, np.array([new_weights]), axis=0).T 
            self.biases1 = np.append(self.biases1, np.array([average_bias_space]), axis=0)
     

            new_weights = np.array([np.random.rand()*2 for i in range(self.noutputs)])

            self.w2 = np.append(self.w2, np.array([new_weights]), axis=0)

            self.delta1 = np.append(self.delta1, np.array([0]), axis=0) 


            '''warnings'''
            if np.shape(self.w1)!=(self.ninputs, self.hidden):
                print('w1 error: ', np.shape(self.w1), ', should be: ', (self.ninputs, self.hidden))
            if np.shape(self.w2)!=(self.hidden, self.noutputs):
                print('w2 error: ', np.shape(self.w2), ', should be: ', (self.hidden, self.noutputs))
   
    def layer_add_hidden_node(self, num=1):
        if self.hidden > 100:
            return 
        for g in range(num):
            average_bias_space = np.random.rand()*2*self.ninputs - self.ninputs
            new_weights = np.array([np.random.rand()*2 - 1  for i in range(self.ninputs)]) 

            self.w1 = np.append(self.w1.T, np.array([new_weights]), axis=0).T 
            self.biases1 = np.append(self.biases1, np.array([average_bias_space]), axis=0)  

        self.delta1 = np.append(self.delta1, np.array([0]), axis=0)     


 
    def prune_worst(self, num=1):
        for j in range(num):
            relevance_scores = []
            for i in range(self.hidden):
                temp = self.w1.T 

                if np.max(np.absolute(self.w2[i])) < .01 / (.01 + relu(np.sum(np.absolute(temp[i]))+self.biases1[i])):
                    relevance_scores.append(np.max(np.absolute(self.w2[i])))

            if not relevance_scores:
                continue 
    
            m = np.argmin(relevance_scores) 

            self.w1 = np.delete(self.w1.T, m, axis=0).T 
            self.biases1 = np.delete(self.biases1, m, axis=0)
            self.w2 = np.delete(self.w2, m, axis=0) 
            # self.hidden -= 1
            # self.delta1 -= 1

    def layer_prune_worst(self, external_weights, num=1):
        for j in range(num):
            relevance_scores = []
            for i in range(self.hidden):
                temp = self.w1.T 

                if np.max(np.absolute(external_weights[i])) < .01 / (.01 + relu(np.sum(np.absolute(temp[i]))+self.biases1[i])):
                    relevance_scores.append(np.max(np.absolute(external_weights[i])))

            if not relevance_scores:
                continue 
    
            m = np.argmin(relevance_scores) 

            self.w1 = np.delete(self.w1.T, m, axis=0).T 
            self.biases1 = np.delete(self.biases1, m, axis=0)

        return m #other external layers need this information   

    def layer_prune_weights(self, m):
        self.w1 = np.delete(self.w1, m, axis=0)    

    def layer_add_weights(self):
        new_weights = np.array([np.random.rand()*2 for i in range(self.hidden)])

        self.w1 = np.append(self.w1, np.array([new_weights]), axis=0)

    def vectorize_node_weights(self, node):
        vector = np.array([self.w1[i, node] for i in range(self.ninputs)]) 
        return (vector / np.linalg.norm(vector)) #normalize 

  
class actual_fanout_layer():
    '''simple one hidden layer network'''
    def __init__(self, inputs, outputs, learning_rate=.1, fanout=0):
        self.ninputs = inputs
        self.noutputs = outputs
        
        self.fanout = fanout 

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
        
        output= sig(self.input1 + self.biases1)
        self.output = output 
        return output 

        
    def delta(self, feedback):
        '''update learning rate''' 
        self.learning_rate = .01
        bias_changes = []

        for i in range(self.noutputs):
            self.delta1[i] = feedback[i]
            bias_changes.append(self.learning_rate*feedback[i]*self.output)

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









