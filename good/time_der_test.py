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

net = fl.fanout_network(3, 1, 2, [3, 12, 1], fanout=0, learning_rate=.1, ordered=False, growth=True) 
net.set_activation(spiking_activation) 
net.set_der(spiking_der) 
net.set_bias_bounds(min_bias, max_bias) 
net.randomize_biases()
#net.set_layers_to_grow([0]) 


def learn(net):
    error = 10 

    epoch = 0

    while error > .5 and epoch < 10:
        epoch += 1
        error, iter = fl.run_learn_cycle(net, xor_inputs, xor_outputs, .5, len(xor_inputs)) 
        print('Epoch: ', epoch)
        print('Iter: ', iter) 
        print('error: ', error) 
        print('size of network: ', len(net.hidden_layers[0].biases1))
        print('biases: ', net.hidden_layers[0].biases1)

    #print results 
    for i in range(len(xor_inputs)):
        result = net.activate(xor_inputs[i]) 
        answer = xor_outputs[i]
        e = answer - result 
        print(xor_inputs[i], ':, ', result, answer, e)
    print('size of network: ', len(net.hidden_layers[0].biases1))
    print('biases: ', net.hidden_layers[0].biases1) 
    print(net.hidden_layers[1].biases1) 


learn(net) 
input('add layer...') 
#now add layer to network
net.layers += 1
net.layer_sizes.insert(-1, 3)
net.hidden_layers.insert(-1, fl.actual_fanout_layer(net.layer_sizes[1], 3))
net.set_activation(spiking_activation) 
net.set_der(spiking_der) 
net.set_bias_bounds(min_bias, max_bias) 
net.hidden_layers[-2].randomize_biases()

#set only new layer to be the one learning 
net.set_layers_to_grow([-2])
net.set_layers_to_adjust([-2]) 

#learn again 
learn(net) 


weights = np.array([net.hidden_layers[n].w1 for n in range(net.layers)])
fanout_codes = np.array([net.hidden_layers[n].fanout_encoding1 for n in range(net.layers)])
biases =  np.array([net.hidden_layers[n].biases1 for n in range(net.layers)])


np.save('xor_weights', weights, allow_pickle=True) 
np.save('xor_biases', biases, allow_pickle=True) 
np.save('xor_fanout_codes', fanout_codes, allow_pickle=True) 
print('Saved Successfully') 

weights = np.load('xor_weights.npy', allow_pickle=True)

fanout_codes = np.load('xor_fanout_codes.npy', allow_pickle=True) 

biases = np.load('xor_biases.npy', allow_pickle=True)

net1 = fl.fanout_network(3, 1, 2, [3, np.shape(weights[0])[1], 1], fanout=0, learning_rate=.1, ordered=False, growth=False)

net1.import_weights(weights, fanout_codes)


#print(net1.hidden_layers[0].noutputs, net1.hidden_layers[1].ninputs) 
# input()

net1.import_biases(biases)

learn(net1)

print('starting distillation:')
input() 

net1.growth_flag = False 

for l in [0, 1]:
    while net1.hidden_layers[l].noutputs > 3:
        net1.distill(l, net1.hidden_layers[l].noutputs-1) 
        learn(net1) 
        print('layer size: ', net1.hidden_layers[l].noutputs) 
    input('next layer....') 

weights = np.array([net1.hidden_layers[n].w1 for n in range(net1.layers)])
fanout_codes = np.array([net1.hidden_layers[n].fanout_encoding1 for n in range(net1.layers)])
biases =  np.array([net1.hidden_layers[n].biases1 for n in range(net1.layers)])


np.save('xor_weights', weights, allow_pickle=True) 
np.save('xor_biases', biases, allow_pickle=True) 
np.save('xor_fanout_codes', fanout_codes, allow_pickle=True) 

for l in [0, 1, 2]:
    print('layer size: ', net1.hidden_layers[l].noutputs) 







