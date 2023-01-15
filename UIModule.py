import numpy as np 
#import seaborn as sns 
from matplotlib import pyplot as plt 

xor_inputs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]] 
xor_outputs = [0, 1, 1, 0, 1, 0, 0, 1]  

def create_features_pca(n, samples, answers):
    '''extract n most prominent features from samples'''
    #convolve
    prominent_features = np.multiply(samples, answers)

    cov = np.cov(prominent_features.T)

    values, vectors = np.linalg.eig(cov) 

    eig_dict = {values[i]:vectors[i] for i in range(len(values))} 

    n_values = np.sort(values)[-n:-1] 

    n_vectors = [eig_dict[v] for v in n_values] 

    return n_vectors  

     

    pass 

def relu(x):
    return np.maximum(0, np.arctan(x)) 
def sig(x):
    return relu(x) 
    return 1/(1+np.exp(-x)) 
def sig_der(x):
    return 1 if x > 0 else .1#(1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x))) 
    return sig(x)*(1-sig(x)) 


class simple_learn():
    '''simple one hidden layer network'''
    def __init__(self, inputs, outputs, hidden, layers=1, layer_sizes=[], learning_rate=.1, fanout=0):
        self.ninputs = inputs
        self.noutputs = outputs
        self.hidden = hidden 
        self.fanout = fanout 

        
        self.layer_sizes = layer_sizes if layer_sizes else [self.hidden for i in range(layers)] 
        self.layers = layers + 2
        self.layer_sizes = [self.ninputs] + self.layer_sizes + [self.noutputs] 

        self.input = 0
        self.input1 = 0

        self.dataflow_matrix = [[0 for i in range(s)] for s in self.layer_sizes] 
        

        self.biases = []

        for i in range(1, self.layers):
            b = np.array([-self.layer_sizes[i-1]/2 + j*self.layer_sizes[i-1]/self.layer_sizes[i] for j in range(self.layer_sizes[i])])
            self.biases.append(b) 

        self.biases1 = np.array([-inputs/2 + i*inputs/hidden for i in range(self.hidden)])
        self.biases2 = np.array([-hidden/2 + i*hidden/outputs for i in range(self.noutputs)]) 


        self.deltas = [np.zeros(self.layer_sizes[i]) for i in range(1, self.layers)] 
        self.delta1 = np.zeros(self.hidden)

        self.max_change = 0 
        #TODO for many layers may be wise and even very effective to test convergence on individual 
        #layers and update them individually on a rollover signal from the following layer-
        #for now just a single max_change for global convergence, though this may drastically increase learning times as global convergence
        #on large mult-layer networks will take a while

        
        

        self.learning_rate = learning_rate


        self.W = []

        for i in range(1, self.layers):
            w = np.array([[np.random.rand()*2-1 for p in range(self.layer_sizes[i])] for j in range(self.layer_sizes[i-1])])
            self.W.append(w) 

        self.w1 = np.array([[np.random.rand()*2-1 for i in range(hidden)] for j in range(self.ninputs)])
        self.w2 = np.array([[np.random.rand()*2-1 for i in range(self.noutputs)] for j in range(self.hidden)])


        '''set up fanout weight connections'''
        '''for now just a random wiring, TODO option to link manually or with patterned automation instead'''

        self.fanout_codes = []

        temp = self.layer_sizes
        for s in range(1, len(temp)):
            l = range(temp[s])
            if self.fanout < temp[s] and self.fanout!=0:
                f = [np.random.choice(l, size=self.fanout, replace=False) for i in range(temp[s-1])]
                self.fanout_codes.append(f)
            else:
                self.fanout_codes.append([l for j in range(temp[s-1])]) 

        # if self.fanout < temp[-1] and self.fanout!=0:
        #     f = [np.random.choice(range(self.noutputs), size=self.fanout, replace=False) for i in range(temp[-1])]
        #     self.fanout_codes.append(f)
        # else:
        #     self.fanout_codes.append([range(self.noutputs) for j in range(temp[-1])]) 


        # layer1 = range(self.hidden) 
        # if self.fanout < self.hidden and self.fanout!=0:
        #     #self.fanout_encoding1 = [[np.random.choice(layer1, replace=False) for i in range(self.fanout)] for j in range(self.ninputs)]
        #     #self.fanout_encoding1 = np.random.choice(layer1, size=(self.ninputs, self.fanout), replace=False)
        #     self.fanout_encoding1 = [np.random.choice(layer1, size=self.fanout, replace=False) for i in range(self.ninputs)]
        # else:
        #     self.fanout_encoding1 = [layer1 for j in range(self.ninputs)] 
        # if self.fanout < self.noutputs and self.fanout!=0:
        #     #self.fanout_encoding2 = [[np.random.choice(range(self.noutputs, replace=False)) for i in range(self.fanout)] for j in range(self.hidden)]
        #     #self.fanout_encoding2 = np.random.choice(range(self.noutputs), size=(self.hidden, self.noutptus), replace=False)
        #     self.fanout_encoding2 = [np.random.choice(range(self.noutputs)) for i in range(self.hidden)] 
        # else:
        #     self.fanout_encoding2 = [range(self.noutputs) for j in range(self.hidden)] 

        self.feature_similarity_threshold = .9 #arbitario
    


    def activate(self, input):
        self.input = input 
        self.input1 = np.dot(self.input, self.w1) 
        if len(self.input1)!=self.hidden:
            print('incorrect input1 size') 
        
        self.input2 = sig(self.input1+self.biases1)
        if len(self.input2)!=self.hidden:
            print('incorrect input1 size', len(self.input2))  
        self.input3 = np.dot(self.input2, self.w2) 
        if len(self.input3)!=self.noutputs:
            print('incorrect input3 size', len(self.input3)) 
  
        output = sig(self.input3 + self.biases2) 

        self.output = output 
        return output 

    #test
    def layer_activate(self, input1):
        self.input = input1
        self.dataflow_matrix[0] = self.input 
        temp = self.input
  
        for k in range(self.layers-1):
            #print('go')
            for i in range(self.layer_sizes[k]):
                #print(k, len(self.fanout_codes)) 

                for j in self.fanout_codes[k][i]:
                    #print(k, i, j)
                    #print(k, j, np.shape(self.fanout_codes))
                    self.dataflow_matrix[k+1][j] += self.W[k][i][j]*temp[i]
            #print(k)
            temp = sig(self.dataflow_matrix[k+1] + self.biases[k])
        self.output = temp
        
        return self.output 
        print('a;sdf: ', len(temp))
           
        #input()
        # '''and the output layer'''
        #o = np.zeros(self.noutputs)
        for i in range(self.layer_sizes[-2]):
            for j in range(self.noutputs):
                print(i, len(temp))
                self.dataflow_matrix[-1][j] += self.W[-1][i][j]*temp[i] 

        self.output = sig(self.dataflow_matrix[-1] + self.biases[-1])
   
        return self.output 
        


        pass 







    def fanout_activate(self, input):
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

    def learn(self, feedback):
        self.delta(feedback)
        change = self.adjust(feedback) 
        self.age += 1
        if self.age > self.lifespan:
            self.evolve()
            self.age = 0
            #adjust lifespan? 

        #return change


    def evolve(self):
        #self.prune() 
        self.add_hidden_node(num=3) 

    def delta(self, feedback):
        '''update learning rate''' 
        self.learning_rate = .01#/(np.std(self.error_history) + .01)
        bias_changes = []
        '''backpropagate through final layer'''
        # for i in range(self.noutputs):
        #     bias_changes.append(self.learning_rate*feedback[i]*sig_der(self.input3[i]))

        #ders = np.array([sig_der(self.input3[i])*feedback[i] for i in range(self.noutputs)])
       
        #feedback2 = np.dot(ders, self.w2.T)
        #feedback2 = [np.sum([ders[j]*self.w2[i][j] for j in self.fanout_encoding2[i]]) for i in range(self.hidden)]  


        '''update deltas in all hidden layers'''
        for l in range(1, self.layers):
            for i in range(self.layer_sizes[-l]):
                self.deltas[-l][i] = feedback[i]
                bias_changes.append(self.learning_rate*feedback[i]*sig_der(self.dataflow_matrix[-l][i]))
            # print(feedback, self.deltas[-1]) 
            # input()
            ders = np.array([sig_der(self.dataflow_matrix[-l][i])*self.deltas[-l][i] for i in range(self.layer_sizes[-l])])
            # print(len(self.layer_sizes), len(self.W), len(self.fanout_codes))
            # print(l)
            feedback = [np.sum([ders[j]*self.W[-l][i][j] for j in self.fanout_codes[-l][i]]) for i in range(self.layer_sizes[-(l+1)])]
                


        # for i in range(self.hidden):
        #     self.delta1[i] = feedback2[i] 
        #     bias_changes.append(self.learning_rate*feedback2[i]*sig_der(self.input1[i]))
            #self.biases1 += self.learning_rate*feedback2[i]*sig_der(self.input1[i]) 

            # if self.biases1[i] > self.ninputs:
            #     self.biases1[i] = self.ninputs
            # elif self.biases1[i] < -self.ninputs:
            #     self.biases1[i] = -self.ninputs 
        #print(self.delta1) 

        return max(bias_changes)
        
    def adjust(self, feedback):

        weight_changes = []
        for l in range(self.layers-1):
            for i in range(self.layer_sizes[l]):
                for j in range(self.layer_sizes[l+1]):
                    # print(self.layer_sizes) 
                    # print(self.deltas)
                    d = self.learning_rate*self.deltas[l][j]*sig(self.dataflow_matrix[l][i])
                    weight_changes.append(d) 
                    #print(self.learning_rate*feedback[j]*self.input2[i]) 
                    #print(d)            
                    self.W[l][i][j] += d     

        return max(weight_changes)                
        

    def adjust_layer(self, feedback, layer):
        weight_changes = []
        for i in range(self.layer_sizes[layer]):
            for j in range(self.layer_sizes[layer+1]):
                d = self.learning_rate*self.deltas[layer+1][j]*sig(self.dataflow_matrix[layer][i]) 
                weight_changes.append(d)
                self.W[layer][i][j] += d 


        # weight_changes = []
        # for i in range(self.hidden):
        #     for j in range(self.noutputs):
        #         weight_changes.append(self.learning_rate*feedback[j]*self.input2[i]) 
        #         #print(self.learning_rate*feedback[j]*self.input2[i])             
        #         self.w2[i][j] += self.learning_rate*feedback[j]*self.input2[i]  
    
        # # input()
        # # print(self.w2)
        # # input() 
        # for i in range(self.ninputs):
        #     for j in range(self.hidden):
        #         weight_changes.append(self.learning_rate*self.delta1[j]*self.input[i]) 
        #         #print(self.learning_rate*self.delta1[j]*self.input[i])              
        #         self.w1[i][j] += self.learning_rate*self.delta1[j]*self.input[i] 
        #         pass 

        '''adjust weights on historical inputs'''
        # print('shape memory: ', np.shape(self.memory_register))
        # print('shape memory weight: ', np.shape(self.w_memory))
        # print('feedback shape: ', np.shape(feedback)) 
        # for i in range(self.hidden):
        #     for j in range(self.noutputs):
        #         for k in range(self.memory_depth):
        #             #print('k i j: ', k, i, j) 
        #             self.w_memory[k][i][j] += self.learning_rate*feedback[j]*self.memory_register[k][i] 
    
        return max(weight_changes)

    #test layer functionality
    def add_hidden_node(self, layer_num, num=1, use_intelligent_search=False):
        if self.hidden > 100:
            return 
        if use_intelligent_search:
            suggested_weights = self.intelligent_feature_searching() 
            num = len(suggested_weights) 
       
        for g in range(num):

            if use_intelligent_search:
                average_bias_space = 0
                new_weights = np.flip(suggested_weights[g]).flatten()  

            else:
                average_bias_space = np.random.rand()*2*self.layer_sizes[layer_num] - self.layer_sizes[layer_num]#abs(np.average(self.biases1) + (np.random.rand()*2-1)) 
                #print('new bias: ', average_bias_space)
                new_weights = np.array([np.random.rand()*2 - 1  for i in range(self.layer_sizes[layer_num-1])]) 
                #new_weights = np.array([1 for i in range(self.ninputs)]) 
        
            #self.w1 = np.append(self.w1.T, np.array([new_weights]), axis=0).T 
            self.W[layer_num-1] = np.append(self.W[layer_num-1].T, np.array([new_weights]), axis=0).T

            #self.biases1 = np.append(self.biases1, np.array([average_bias_space]), axis=0)
            #print(self.biases[layer_num-1])
            self.biases[layer_num-1] = np.append(self.biases[layer_num-1], np.array([average_bias_space]), axis=0)
            #print(self.biases[layer_num-1])
            #self.layer_sizes[layer_num] += 1 

            new_weights = np.array([np.random.rand()*2 for i in range(self.layer_sizes[layer_num+1])])
            #new_weights = np.array([1 for i in range(self.noutputs)])

            self.W[layer_num] = np.append(self.W[layer_num], np.array([new_weights]), axis=0)

            self.deltas[layer_num-1] = np.append(self.deltas[layer_num-1], np.array([0]), axis=0) 

            # '''expand memory register/weights'''
            # # for m in self.memory_register:
                
                
            # #     m = self.add_row(m, 0) 
            # zeros = np.zeros((self.memory_depth)) 
            # self.memory_register = self.add_row(self.memory_register.T, zeros).T 
            
      
            # for i in range(len(self.w_memory)):      
            #     new_weights = np.array([np.random.rand()*2-1 for i in range(self.noutputs)]) 
            #     self.w_memory[i] = self.add_row(self.w_memory[i], new_weights)

            # '''warnings'''
            # if np.shape(self.w1)!=(self.ninputs, self.hidden):
            #     print('w1 error: ', np.shape(self.w1), ', should be: ', (self.ninputs, self.hidden))
            # if np.shape(self.w2)!=(self.hidden, self.noutputs):
            #     print('w2 error: ', np.shape(self.w2), ', should be: ', (self.hidden, self.noutputs))

    '''only prune_worst is layer compatibile at this time'''
    def prune(self):
      
        nodes_to_remove = []
        for i in range(self.hidden):
            temp = self.w1.T 

            if False:#(abs(self.biases1[i]) > self.ninputs) and np.max(np.absolute(self.w2[i])) < .01 / (.01 + relu(np.sum(np.absolute(temp[i]))+self.biases1[i])): 
                nodes_to_remove.append(i)
            elif np.max(np.absolute(self.w2[i])) < .01 / (.01 + relu(np.sum(np.absolute(temp[i]))+self.biases1[i])):
                nodes_to_remove.append(i) 

        '''now, to remove the nodes'''
        if self.hidden < len(nodes_to_remove)+3:
            return 
        self.w1 = np.delete(self.w1.T, nodes_to_remove, axis=0).T 
        self.biases1 = np.delete(self.biases1, nodes_to_remove, axis=0)
        self.w2 = np.delete(self.w2, nodes_to_remove, axis=0) 
        self.hidden -= len(nodes_to_remove) 
        self.delta1 -= len(nodes_to_remove) 
        return
        # '''historical adjustment'''
        for n in nodes_to_remove:
            self.memory_register = self.remove_nth_row(self.memory_register.T, n).T 

           
            self.w_memory = self.remove_nth_row(self.w_memory, n, ax=1) 
   
    def prune_similar(self):
        nodes_to_remove, nodes_to_combine = self.find_common_nodes()
    
        if self.hidden < len(nodes_to_remove)+3:
            return 
        self.w1 = np.delete(self.w1.T, nodes_to_remove, axis=0).T 
        self.biases1 = np.delete(self.biases1, nodes_to_remove, axis=0)
        self.w2 = np.delete(self.w2, nodes_to_remove, axis=0) 
        self.hidden -= len(nodes_to_remove) 
        self.delta1 -= len(nodes_to_remove) 
        return 

    #test layer compatibility
    def prune_worst(self, layer_num, num=1):
        # print('test: ', self.layer_sizes[layer_num])
        # input()
        for j in range(num):
            relevance_scores = []
            for i in range(self.layer_sizes[layer_num]):
                temp = self.W[layer_num].T 

                #print(layer_num, self.biases[layer_num-1])
                if np.max(np.absolute(self.W[layer_num][i])) < .01 / (.01 + relu(np.sum(np.absolute(self.W[layer_num][i]))+self.biases[layer_num-1][i])):
                    relevance_scores.append(np.max(np.absolute(self.W[layer_num][i])))

            if not relevance_scores:
                continue 
    
            m = np.argmin(relevance_scores) 

            self.W[layer_num-1] = np.delete(self.W[layer_num-1].T, m, axis=0).T 
            self.biases[layer_num-1] = np.delete(self.biases[layer_num-1], m, axis=0)
            self.W[layer_num] = np.delete(self.W[layer_num], m, axis=0) 
            #self.layer_sizes[layer_num] -= 1
            self.deltas[layer_num-1] = np.delete(self.deltas[layer_num-1], m, axis=0)
            #self.deltas[layer_num] -= 1
 

    def delete_node(self, node):
        self.w1 = np.delete(self.w1.T, node, axis=0).T 
        self.biases1 = np.delete(self.biases1, node, axis=0)
        self.w2 = np.delete(self.w2, node, axis=0) 
        self.hidden -= len(node) 
        self.delta1 -= len(node) 

    def vectorize_node_weights(self, node):
        vector = np.array([self.w1[i, node] for i in range(self.ninputs)]) 
        return (vector / np.linalg.norm(vector)) #normalize 

    def find_common_nodes(self):
        unique_nodes = []
        redundant_nodes = []
        for i in range(self.hidden):
            if not unique_nodes:
                unique_nodes.append(i)
                continue 
            else:
                present = False
                for e in unique_nodes:
                    v1 = self.vectorize_node_weights(i)
                    v2 = self.vectorize_node_weights(e) 
                    
                    if np.dot(v1, v2) > self.feature_similarity_threshold:                      
                        present=True
                        break

                if not present:
                    unique_nodes.append(i)
                else:
                    redundant_nodes.append(i) 

        return redundant_nodes, unique_nodes

    def unique(self):
        nodes_to_delete = []
        w = np.zeros(self.noutputs) 
        for i in range(self.hidden):
            for j in range(i, self.hidden-1):
                v1 = self.vectorize_node_weights(i)
                v2 = self.vectorize_node_weights(j)
                if i!=j and np.dot(v1, v2) < self.feature_similarity_threshold:
                    nodes_to_delete.append(i) 
                    break


        #condense weight information from deleted nodes
        for n in nodes_to_delete:
            w += self.w2[n][:] 

        self.delete_node(nodes_to_delete) 

    '''some helper functions'''
    def remove_nth_row(self, matrix, row_num, ax=0):
        matrix = np.delete(matrix, row_num, axis=ax) 
        return matrix

    def add_row(self, matrix, row, ax=0):
        matrix = np.append(matrix, np.array([row]), axis=ax)
        return matrix


def create_heatmap(nn, xlim=1, ylim=1):
    print('creating heatmap...') 
    xrange = np.linspace(0, xlim, 100)
    yrange = np.linspace(0, ylim, 100) 
    uniform_data = np.array([[nn.activate([x, y]) for x in xrange] for y in yrange]).reshape((100, 100)) 
    ax = sns.heatmap(uniform_data, linewidth=0.5)
    plt.show()

def no_growth_test(num_nodes=3, layers=1, fanout=0):
    test = simple_learn(len(xor_inputs[0]), 1, num_nodes, layers=layers, fanout=fanout)
  
    iter = 0
    e = 2
    errors = [1, 1] 
    converged = False

    test_inputs = [[0], [1]]
    test_outputs = [1, 0] 

    def o(a, b, c, d, x):
        return sig(d+c*sig(b+a*x)) 

    def run_learn_cycle():
        errors = 5
        iter = 0
        max_change = 10
        while np.sum(np.abs(errors)) >= .2 and converged==False and iter<10000 and max_change > .01:
            e = 0
            errors = []
            changes = []
            for i in range(len(xor_inputs)):
                # result = test.fanout_activate(xor_inputs[i])
                result = test.layer_activate(xor_inputs[i])

                answer = xor_outputs[i]

                error = answer - result
            
            
                errors.append(error) 

                for j in range(1):
                    max_bias_change = test.delta(error)    
                    max_weight_change = test.adjust(error) 

                changes.append(max_weight_change) 

            max_change = max(changes) 

            iter += 1
        print('iter: ', iter) 
        return np.sum(np.abs(errors)), iter  

    error, iter = run_learn_cycle() 
 
    t = 0
    while error > .5 and t < 100:
        #print(test.layers)
        for i in range(1, test.layers-1):
            #print(i)
            test.prune_worst(layer_num=i, num=1) 
            test.add_hidden_node(layer_num=i, num=(test.layer_sizes[i]-len(test.biases[i-1])), use_intelligent_search=False) 
            #print(len(test.biases[i-1]))
   
        error, iter = run_learn_cycle()
        t += 1
        print(t)
 
    

    for i in range(len(xor_inputs)):
        result = test.layer_activate(xor_inputs[i]) 
        answer = xor_outputs[i]
        e = answer - result 
        print(xor_inputs[i], ':, ', result, answer, e) 

    pass 



def test_run(inputs=xor_inputs, outputs=xor_outputs):
    test = simple_learn(len(inputs[0]), 1, 3) 
    t = 0
    error = 3
    
    while error > 2:
        
        e = []
        test.max_change = 0
        for i in range(len(inputs)):
            result = test.activate(inputs[i]) 
            # print('test: ', result) 
            # input()
            loss = outputs[i] - result
            test.learn(loss) 

            e.append(abs(loss)) 
        #test.prune()
        error = np.sum(e) 
        t += 1
        #print(t, error)

    print('Num of nodes: ', test.hidden)
    test.prune() 
    for i in range(len(inputs)):
        print(inputs[i], ':, ', test.activate(inputs[i]), outputs[i])
    # print('second layer max weights: \n\n')
    # for i in range(test.hidden):
    #     w = np.absolute(test.w2[:][i])
    #     print(max(w)) 
    #test.unique()
    # print('Num of nodes after similarity prune: ', test.hidden)
    # for i in range(len(inputs)):
    #     print(inputs[i], ':, ', test.activate(inputs[i]), outputs[i]) 

    
    #test.find_common_nodes() 


    return test 




def main(inputs=xor_inputs, outputs=xor_outputs):

    test = simple_learn(len(inputs[0]), 1, len(inputs[0]))

    

    iter = 0
    e = 2
    errors = [1, 1] 
    converged = False

    test_inputs = [[0], [1]]
    test_outputs = [1, 0] 

    def o(a, b, c, d, x):
        return sig(d+c*sig(b+a*x)) 

    def run_learn_cycle():
        errors = 5
        iter = 0
        max_change = 10
        while np.sum(np.abs(errors)) >= .2 and converged==False and iter<10000 and max_change > .01:
            e = 0
            errors = []
            changes = []
            for i in range(len(inputs)):
                result = test.activate(inputs[i])

                answer = outputs[i]

                error = answer - result
            
                #print(xor_inputs[i], ': ', result, answer, error) 
            
                errors.append(error) 

                for j in range(1):
                    max_bias_change = test.delta(error)    
                    max_weight_change = test.adjust(error) 

                #changes.append(max_bias_change)
                changes.append(max_weight_change) 

            max_change = max(changes) 

                
            # input()
            # print('\n') 
            iter += 1
        print('iter: ', iter) 
        return np.sum(np.abs(errors)), iter  

    error, iter = run_learn_cycle() 
    #test.prune() 
    #test.add_hidden_node(num=3, use_intelligent_search=False) 
    t = 0
    while error > .5 and t < 100:
      
        #test.prune() 
        test.add_hidden_node(num=3, use_intelligent_search=False) 
        #test.biases2 = [np.random.rand() for i in range(test.noutputs)] 
        error, iter = run_learn_cycle()
        t += 1
        print(t)

    #create_heatmap(test) 

    print('\n') 


    
    #input()
    # print('num of hidden nodes: ', test.hidden) 
    # print('iter: ', t) 
    print('biases before: ', len(test.biases1)) 
    test.prune_similar() 
    test.prune() 
    print('biases after: ', len(test.biases1)) 

    for i in range(len(inputs)):
        result = test.activate(inputs[i]) 
        answer = outputs[i]
        e = answer - result 
        print(xor_inputs[i], ':, ', result, answer, e) 

    input('Continue...')

    t = 0
    error = 1
    while error > .5 and t < 100:
      
        #test.prune() 
        #test.add_hidden_node(num=3, use_intelligent_search=False) 
        #test.biases2 = [np.random.rand() for i in range(test.noutputs)] 
        error, iter = run_learn_cycle()
        t += 1
        print(t)

    test.prune() 

    for i in range(len(inputs)):
        result = test.activate(inputs[i]) 
        answer = outputs[i]
        e = answer - result 
        print(xor_inputs[i], ':, ', result, answer, e) 

    print('biases after: ', len(test.biases1)) 

    #print(test.find_common_nodes())
    # print('done.') 
    #return t, test.biases1, error, test 



if __name__=='main':
    main() 


#main()
            










