import numpy as np
import fanout_layer as fl

xor_inputs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]] 
xor_outputs = [0, 1, 1, 0, 1, 0, 0, 1]  


test = fl.fanout_layer(3, 1, 30, .1, fanout=0) 


def fanout_layer_xor_learn_test():
  
    iter = 0
    e = 2
    errors = [1, 1] 
    converged = False

    test_inputs = [[0], [1]]
    test_outputs = [1, 0] 

    def run_learn_cycle():
        errors = 5
        iter = 0
        max_change = 10
        while np.sum(np.abs(errors)) >= .2 and converged==False and iter<10000 and max_change > .01:
            e = 0
            errors = []
            changes = []
            for i in range(len(xor_inputs)):
                result = test.activate(xor_inputs[i])

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
      
        test.prune_worst(1) 
        test.add_hidden_node(num=(test.hidden-len(test.biases1)), use_intelligent_search=False) 
        error, iter = run_learn_cycle()
        t += 1
        print(t, len(test.biases1))

    print('biases after: ', len(test.biases1)) 

    for i in range(len(xor_inputs)):
        result = test.activate(xor_inputs[i]) 
        answer = xor_outputs[i]
        e = answer - result 
        print(xor_inputs[i], ':, ', result, answer, e) 

    pass 



#fanout_layer_xor_learn_test() 


'''layer implementation'''

layer1 = fl.fanout_layer(3, 1, 30, .1, fanout=0) 
output_layer = fl.fanout_layer(30, 1, 1, .1, fanout=0) 

def layered_fanout_test():
    iter = 0
    e = 2
    errors = [1, 1] 
    converged = False

    test_inputs = [[0], [1]]
    test_outputs = [1, 0] 

    def run_learn_cycle():
        errors = 5
        iter = 0
        max_change = 10
        while np.sum(np.abs(errors)) >= .2 and converged==False and iter<10000 and max_change > .01:
            e = 0
            errors = []
            changes = []
            for i in range(len(xor_inputs)):
                result = layer1.layer_activate(xor_inputs[i])
                result = output_layer.layer_activate(result) 

                answer = xor_outputs[i]

                error = answer - result
            
            
                errors.append(error) 

                for j in range(1):
                    max_bias_change1 = output_layer.layer_delta(error) 
                    max_bias_change2 = layer1.layer_delta(output_layer.backpropogate())    
                    max_weight_change1 = output_layer.layer_adjust(error) 
                    max_weight_change2 = layer1.layer_adjust(output_layer.backpropogate()) 

                max_weight_change = max(max_weight_change1, max_weight_change2) 

                changes.append(max_weight_change) 

            max_change = max(changes) 

            iter += 1
        print('iter: ', iter) 
        return np.sum(np.abs(errors)), iter  

    error, iter = run_learn_cycle() 

    t = 0
    while error > .5 and t < 100:
      
        external_weights = output_layer.w1 
        m = layer1.layer_prune_worst(external_weights, 1)
        output_layer.layer_prune_weights(m)
        output_layer.layer_add_weights()    #only does it once 
        layer1.layer_add_hidden_node(num=(layer1.hidden-len(layer1.biases1))) 
        error, iter = run_learn_cycle()
        t += 1
        print(t, len(layer1.biases1))

    print('biases after: ', len(layer1.biases1)) 

    for i in range(len(xor_inputs)):
        result = layer1.layer_activate(xor_inputs[i]) 
        result = output_layer.layer_activate(result)
        answer = xor_outputs[i]
        e = answer - result 
        print(xor_inputs[i], ':, ', result, answer, e) 

    pass 



#layered_fanout_test() 



'''multi-layer class test'''

net = fl.fanout_network(3, 1, 2, [3, 30, 1], fanout=0) 


def multi_class_test():
  
    e = 2
    errors = [1, 1] 
    converged = False

    def run_learn_cycle():
        errors = 5
        iter = 0
        max_change = 10
        while np.sum(np.abs(errors)) >= .2 and iter<10000:
            e = 0
            errors = []
            changes = []
            for i in range(len(xor_inputs)):
                result = net.activate(xor_inputs[i])

                answer = xor_outputs[i]

                error = answer - result
            
                errors.append(error) 

                net.delta(error)
                net.adjust() 
                #net.evolve() 

            if (iter%1000)==0:
                print("Iteration: ", iter) 

            iter += 1
        print('iter: ', iter) 
        net.evolve(force=True) 
        return np.sum(np.abs(errors)), iter  

    error = 2
    iter = 0

    t = 0
    while error > .5 and t < 100:
      
       
        error, iter = run_learn_cycle()
        t += 1
        print("Epoch: ", t)
        print("Error: ", error) 

    for i in range(len(xor_inputs)):
        result = net.activate(xor_inputs[i]) 
        answer = xor_outputs[i]
        e = answer - result 
        print(xor_inputs[i], ':, ', result, answer, e) 



multi_class_test() 






