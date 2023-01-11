import numpy as np 
import seaborn as sns 
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

        self.error_history = [] 
        self.memory_depth = 2
        self.memory_register = np.array([[0 for i in range(hidden)] for j in range(self.memory_depth)])
          
        

        self.age = 0 
        self.lifespan = 100 #change me 
        self.max_change = 0 
        

        self.learning_rate = learning_rate


        self.w1 = np.array([[np.random.rand()*2-1 for i in range(hidden)] for j in range(self.ninputs)])
        self.w2 = np.array([[np.random.rand()*2-1 for i in range(self.noutputs)] for j in range(self.hidden)])


        '''set up fanout weight connections'''
        '''faulty, could possibly be linked twice to the same node'''
        layer1 = range(self.hidden) 
        if self.fanout < self.hidden and self.fanout!=0:
            #self.fanout_encoding1 = [[np.random.choice(layer1, replace=False) for i in range(self.fanout)] for j in range(self.ninputs)]
            #self.fanout_encoding1 = np.random.choice(layer1, size=(self.ninputs, self.fanout), replace=False)
            self.fanout_encoding1 = [np.random.choice(layer1, size=self.fanout, replace=False) for i in range(self.ninputs)]
        else:
            self.fanout_encoding1 = [layer1 for j in range(self.ninputs)] 
        if self.fanout < self.noutputs and self.fanout!=0:
            #self.fanout_encoding2 = [[np.random.choice(range(self.noutputs, replace=False)) for i in range(self.fanout)] for j in range(self.hidden)]
            #self.fanout_encoding2 = np.random.choice(range(self.noutputs), size=(self.hidden, self.noutptus), replace=False)
            self.fanout_encoding2 = [np.random.choice(range(self.noutputs)) for i in range(self.hidden)] 
        else:
            self.fanout_encoding2 = [range(self.noutputs) for j in range(self.hidden)] 

        '''invert'''

      

        self.w_memory = [np.array([[np.random.rand()*2-1 for i in range(self.noutputs)] for j in range(self.hidden)]) for k in range(self.memory_depth)]
        '''memory weights form 3d array'''

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
        '''update error history'''
        self.error_history.append(feedback) 
        if len(self.error_history) > 20:
            self.error_history.pop() 
        '''update learning rate''' 
        self.learning_rate = .01#/(np.std(self.error_history) + .01)
        bias_changes = []
        for i in range(self.noutputs):
            bias_changes.append(self.learning_rate*feedback[i]*sig_der(self.input3[i]))
            #self.biases2[i] += self.learning_rate*feedback[i]*sig_der(self.input3[i]) 

            # if self.biases2[i] > self.hidden:
            #     self.bises2[i] = self.hidden
            # elif self.biases2[i] < -self.hidden:
            #     self.biases2[i] = -self.hidden  


        ders = np.array([sig_der(self.input3[i])*feedback[i] for i in range(self.noutputs)])
       

        #feedback2 = np.dot(ders, self.w2.T)
        feedback2 = [np.sum([ders[j]*self.w2[i][j] for j in self.fanout_encoding2[i]]) for i in range(self.hidden)]  


        if len(feedback2)!=self.hidden:
            print('feedback2 size incorrect') 

        for i in range(self.hidden):
            self.delta1[i] = feedback2[i] 
            bias_changes.append(self.learning_rate*feedback2[i]*sig_der(self.input1[i]))
            #self.biases1 += self.learning_rate*feedback2[i]*sig_der(self.input1[i]) 

            # if self.biases1[i] > self.ninputs:
            #     self.biases1[i] = self.ninputs
            # elif self.biases1[i] < -self.ninputs:
            #     self.biases1[i] = -self.ninputs 
        #print(self.delta1) 

        return max(bias_changes)
        
    def adjust(self, feedback):
        weight_changes = []
        for i in range(self.hidden):
            for j in range(self.noutputs):
                weight_changes.append(self.learning_rate*feedback[j]*self.input2[i]) 
                #print(self.learning_rate*feedback[j]*self.input2[i])             
                self.w2[i][j] += self.learning_rate*feedback[j]*self.input2[i]  
    
        # input()
        # print(self.w2)
        # input() 
        for i in range(self.ninputs):
            for j in range(self.hidden):
                weight_changes.append(self.learning_rate*self.delta1[j]*self.input[i]) 
                #print(self.learning_rate*self.delta1[j]*self.input[i])              
                self.w1[i][j] += self.learning_rate*self.delta1[j]*self.input[i] 
                pass 

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

    def add_hidden_node(self, num=1, use_intelligent_search=False):
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
                average_bias_space = np.random.rand()*2*self.ninputs - self.ninputs#abs(np.average(self.biases1) + (np.random.rand()*2-1)) 
                #print('new bias: ', average_bias_space)
                new_weights = np.array([np.random.rand()*2 - 1  for i in range(self.ninputs)]) 
                #new_weights = np.array([1 for i in range(self.ninputs)]) 
        
            self.w1 = np.append(self.w1.T, np.array([new_weights]), axis=0).T 
            self.biases1 = np.append(self.biases1, np.array([average_bias_space]), axis=0)
            self.hidden += 1 

            new_weights = np.array([np.random.rand()*2 for i in range(self.noutputs)])
            #new_weights = np.array([1 for i in range(self.noutputs)])

            self.w2 = np.append(self.w2, np.array([new_weights]), axis=0)

            self.delta1 = np.append(self.delta1, np.array([0]), axis=0) 

            # '''expand memory register/weights'''
            # # for m in self.memory_register:
                
                
            # #     m = self.add_row(m, 0) 
            # zeros = np.zeros((self.memory_depth)) 
            # self.memory_register = self.add_row(self.memory_register.T, zeros).T 
            
      
            # for i in range(len(self.w_memory)):      
            #     new_weights = np.array([np.random.rand()*2-1 for i in range(self.noutputs)]) 
            #     self.w_memory[i] = self.add_row(self.w_memory[i], new_weights)

            '''warnings'''
            if np.shape(self.w1)!=(self.ninputs, self.hidden):
                print('w1 error: ', np.shape(self.w1), ', should be: ', (self.ninputs, self.hidden))
            if np.shape(self.w2)!=(self.hidden, self.noutputs):
                print('w2 error: ', np.shape(self.w2), ', should be: ', (self.hidden, self.noutputs))

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
            self.hidden -= 1
            self.delta1 -= 1
 

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

    def intelligent_feature_searching(self):
        '''find unique features among nodes'''
        features = []
        weight_vectors = []
        for i in range(len(self.biases1)):
            f = np.argsort(self.w1.T[i]) 
            if (f.tolist() in np.array(features).tolist()): 
                pass
            else:
                features.append(np.argsort(self.w1.T[i])) 
                weight_vectors.append(self.w1.T[i])
        #print(weight_vectors[0]) 
        return weight_vectors  
        pass 



    '''some helper functions'''
    def remove_nth_row(self, matrix, row_num, ax=0):
        matrix = np.delete(matrix, row_num, axis=ax) 
        return matrix

    def add_row(self, matrix, row, ax=0):
        matrix = np.append(matrix, np.array([row]), axis=ax)
        return matrix



def ryan_test(inputs=xor_inputs, outputs=xor_outputs):
    '''only update weights...essentially extracting any possible information from a given feature'''
    '''then begin adjusting biases /w weights'''
    test = simple_learn(len(inputs[0]), 1, len(inputs[0])*2)

    def run_learn_cycle_weights():
        errors = 5
        iter = 0
        max_change = 10
        while np.sum(np.abs(errors)) >= .2 and iter<10000 and max_change > .01:
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

    def run_learn_cycle_biases():
        errors = 5
        iter = 0
        max_change = 10
        while np.sum(np.abs(errors)) >= .2 and iter<10000 and max_change > .01:
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
                    max_bias_change = test.bias_tuning(error)    
                    max_weight_change = test.adjust(error) 

                changes.append(max_bias_change)
                changes.append(max_weight_change) 

            max_change = max(changes) 

                    
            # input()
            # print('\n') 
            iter += 1
        print('iter: ', iter) 
        return np.sum(np.abs(errors)), iter 
        
    run_learn_cycle_weights() 
    #run_learn_cycle_biases() 

    for i in range(len(inputs)):
        result = test.activate(inputs[i]) 
        answer = outputs[i]
        e = answer - result 
        print(xor_inputs[i], ':, ', result, answer, e) 
        
        
        
    pass

def create_heatmap(nn, xlim=1, ylim=1):
    print('creating heatmap...') 
    xrange = np.linspace(0, xlim, 100)
    yrange = np.linspace(0, ylim, 100) 
    uniform_data = np.array([[nn.activate([x, y]) for x in xrange] for y in yrange]).reshape((100, 100)) 
    ax = sns.heatmap(uniform_data, linewidth=0.5)
    plt.show()

def no_growth_test(num_nodes=3, fanout=0):
    test = simple_learn(len(xor_inputs[0]), 1, num_nodes, fanout=fanout)

    # print(test.fanout_encoding1)
    # input()

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
                result = test.fanout_activate(xor_inputs[i])

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
    #test.prune() 
    #test.add_hidden_node(num=3, use_intelligent_search=False) 
    t = 0
    while error > .5 and t < 100:
      
        test.prune_worst(1) 
        test.add_hidden_node(num=(num_nodes-len(test.biases1)), use_intelligent_search=False) 
        #test.biases2 = [np.random.rand() for i in range(test.noutputs)] 
        error, iter = run_learn_cycle()
        t += 1
        print(t, len(test.biases1))

    #create_heatmap(test) 

    # print('\nbiases before: ', len(test.biases1)) 
    # test.prune_similar() 
    # test.prune() 
    print('biases after: ', len(test.biases1)) 

    for i in range(len(xor_inputs)):
        result = test.fanout_activate(xor_inputs[i]) 
        answer = xor_outputs[i]
        e = answer - result 
        print(xor_inputs[i], ':, ', result, answer, e) 

    pass 


def fanout_test(num_nodes=3, fanout=0):

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

def time_test_run():

    pass 



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
            










