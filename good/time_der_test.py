import numpy as np

import fanout_layer as fl 

from matplotlib import pyplot as plt

xor_inputs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]] 
xor_outputs = [0, 1, 1, 0, 1, 0, 0, 1]  

min_bias = .01
max_bias = 1

def spiking_activation(x, bias):
    return np.maximum(0, np.multiply(x, bias)) 
def spiking_der(x, bias):
    return bias if x > 0 else .1

net = fl.fanout_network(3, 1, 2, [3, 16, 1], fanout=0, learning_rate=.1, ordered=False, growth=True) 
net.set_activation(spiking_activation) 
net.set_der(spiking_der) 
net.set_bias_bounds(min_bias, max_bias) 
net.randomize_biases()


error = 10 

epoch = 0

while error > .5 and epoch < 10:
    epoch += 1
    error, iter = fl.run_learn_cycle(net, xor_inputs, xor_outputs, .5, len(xor_inputs)) 
    print('Epoch: ', epoch)
    print('Iter: ', iter) 

#print results 
for i in range(len(xor_inputs)):
    result = net.activate(xor_inputs[i]) 
    answer = xor_outputs[i]
    e = answer - result 
    print(xor_inputs[i], ':, ', result, answer, e)
print('size of network: ', len(net.hidden_layers[0].biases1))
#save results








