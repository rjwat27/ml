import numpy as np
import fanout_layer as fl
import PhaseDomainNeuron as pdn
import pdn_net as pdnNet 
from matplotlib import pyplot as plt 

xor_inputs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]] 
xor_outputs = [0, 1, 1, 0, 1, 0, 0, 1]  



'''multi-layer class test'''

net = fl.fanout_network(3, 1, 3, [3, 6, 12, 1], fanout=0) 


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

    return np.array([net.hidden_layers[n].w1 for n in range(net.layers)]), np.array([net.hidden_layers[n].fanout_encoding1 for n in range(net.layers)])


# weights, fanout_codes = multi_class_test() 

# np.save('xor_weights', weights, allow_pickle=True) 
# np.save('xor_fanout_codes', fanout_codes, allow_pickle=True) 

# print("Done.\n\n\n") 



'''same but with phase-domain neuron'''
'''this is horrifically slow, indeed because weights are supposed to be trained classically before imported to a pdn model'''

pdn_net = fl.fanout_network_with_neuron_structures(3, 1, fanout=0, layers = 3, layer_sizes=[3, 10, 20, 1])

def multi_class_test_pdn():
  
    e = 2
    errors = [1, 1] 
    converged = False

    neuron = pdn_net.hidden_layers[-1].nodes[-1]
    inputs_over_time = [[0, 0, 0] for i in range(neuron.stream_max)] 

    def run_learn_cycle():
        errors = 5
        iter = 0
        max_change = 10
        # while np.sum(np.abs(errors)) >= .2 and iter<10000:
        while iter<100:
            e = 0
            errors = []
            changes = []
            for i in range(len(xor_inputs)):
 
                ## how long to hold a certain value in time sim
                for k in range(125):
                    inputs_over_time.append(xor_inputs[i])
                    inputs_over_time.pop(0)

                    result = pdn_net.activate(xor_inputs[i])[0]

                    answer = xor_outputs[i]

                    error = answer - result
                
                    errors.append(error) 

                    pdn_net.delta([error])
                    pdn_net.adjust() 
                    pdn_net.tick_neurons()
                    #net.evolve() 
                    print(k)

            if (iter%1000)==0:
                print("Iteration: ", iter) 

            iter += 1
        print('iter: ', iter) 
        pdn_net.evolve(force=True) 
        return np.sum(np.abs(errors)), iter  

    error = 2
    iter = 0

    t = 0
    while error > .5 and t < 100:
      
       
        error, iter = run_learn_cycle()
        t += 1
        print("Epoch: ", t)
        print("Error: ", error) 

    fig, axes = plt.subplots(5)

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2] 
    ax4 = axes[3]
    ax5 = axes[4]

    

    ax1.plot(range(neuron.stream_max), [a[0] for a in inputs_over_time]) 
    ax1.set_title('input 1') 


    ax2.plot(range(neuron.stream_max), [a[1] for a in inputs_over_time]) 
    ax2.set_title('input 2')

    ax3.plot(range(neuron.stream_max), [a[2] for a in inputs_over_time])
    ax3.set_title('input 3') 

    ax4.plot(range(neuron.stream_max), neuron.feedback_stream)
    ax4.set_title('feedback over time') 

    ax5.plot(range(neuron.stream_max), neuron.output_stream) 
    ax5.set_title('neuron output')

    plt.show() 


    # for i in range(len(xor_inputs)):
    #     result = net.activate(xor_inputs[i]) 
    #     answer = xor_outputs[i]
    #     e = answer - result 
    #     print(xor_inputs[i], ':, ', result, answer, e) 

#multi_class_test_pdn()


'''pdn net with imported weights'''
print("loading weights") 
'''FORMAT USED TO GENERATE WEIGHTS AND FANOUT CODES: 3, 1, 3, [3, 6, 12, 1], fanout=0 AS INPUT TO fl.fanout_network()'''
weights = np.load('xor_weights.npy', allow_pickle=True)
fanout_codes = np.load('xor_fanout_codes.npy', allow_pickle=True) 

brain = pdnNet.PDN_Network()

params = {'ninputs':3, 'noutputs':1, 'fanout':1, 'hidden':[6, 12]}

brain.configure(params=params)
brain.import_weights(weights, fanout_codes) 
brain.import_weights(weights, fanout_codes)


print("Successfully imported weights") 

