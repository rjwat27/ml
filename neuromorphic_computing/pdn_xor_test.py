import numpy as np
import PhaseDomainNeuron as pdn

import pdn_net as pn 

import weight_bias_transform as wbt 

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

xor_inputs = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]]) 
xor_outputs = np.array([0, 0, 0, 0, 1, 1, 1, 1]) 

xor_inputs2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs2 = np.array([0, 1, 1, 0])

inputs = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
targets = torch.tensor([[0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0]])

inputs2 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
targets2 = torch.tensor([[0.0], [1.0], [1.0], [0.0]]) 


pdn.energy_per_spike = 1#max_bias * 1.1 

import pytorch_weight_finder as pwf 

'''importing parameters from pytorch model'''


'''generating model from network similar to pdn'''
# loss, model = pwf.get_weights(inputs2, targets2, num_epochs=int(10e3), clipping='during', graph=True, quantized=True) 
# n_params = pwf.pytorch_params_to_numpy(model) 


# np.save('testtest', np.array(n_params, dtype=object), allow_pickle=True) 

n_params = np.load('testtest.npy', allow_pickle=True) 
#input(n_params) 

'''test with custom weights'''
# w1 = np.array([[.9, .9, .9], [.9, .9, .9], [.9, .9, .9]])
# w2 = np.array([[.6, -.9, .6], [0, .9, -.9], [0, 0, .6]])
# w3 = np.array([[.9, -.9, .9]])

# weights = [w1.T, w2.T, w3.T]
#input(np.shape(weights[1]))
#np.save('as;dlkfj', weights) 

# b1 = np.array([.5, .75, .99])
# b2 = np.array([.9, .9, .5])
# b3 = np.array([1.]) 

# biases = [b1, b2, b3]



#input()
# for n in n_params:
#     print(n, np.shape(n)) 
# input()

#input(n_params[4]) 
weights = [n_params[0].T, n_params[1].T, n_params[2].T, n_params[3].T]#, n_params[3]]
#weights = [n_params[0].T, n_params[1].T, n_params[2].T]
# np.save('asd;lkfj', weights) 


# np.flip(weights) 
biases = [n_params[4], n_params[5], n_params[6], n_params[7]]#, n_params[7]] 
#biases = [n_params[3], n_params[4], n_params[5]]

#net = pn.pdn_network(3, 1, [len(biases[0]), 1])
net = pn.pdn_network(2, 1, [len(biases[0]), len(biases[0]), len(biases[0]), 1]) 
#np.flip(biases)
#input(biases)
#input(weights[0])
net.weights = weights 
 
# print('beginning calibration')
# net.calibrate_network_biases([1, 1, 1], biases)

'''pytorch loading done'''


'''loading parameters from numpy model'''

def load():
    weights = np.load('xor_weights.npy', allow_pickle=True)

    fanout_codes = np.load('xor_fanout_codes.npy', allow_pickle=True) 

    biases = np.load('xor_biases.npy', allow_pickle=True)

    return weights, fanout_codes, biases 

# weights, fanouts, biases = load() 
# #input(biases) 
# import mlp as mlp
# #in case i want to distill before importing to pdn
# temp_net = mlp.mlp(3, 1, len(weights), [3, len(weights[0]), 1], growth=False)
# temp_net.import_biases(biases)
# temp_net.import_weights(weights, fanouts) 
# temp_net.update_layer_sizes() 

# temp_net, sizes = mlp.distill(temp_net, [3, 3, 1], xor_inputs, xor_outputs, .5, cohort=len(xor_inputs)) 

# net = pn.pdn_network(3, 1, [len(temp_net.hidden_layers[0].biases1)]) 

# print('done distilling: ', sizes) 
# for i in range(len(xor_inputs)):
#     an = temp_net.activate(xor_inputs[i])
#     print(xor_inputs[i], ': ', an, '; ', xor_outputs[i]) 

# def save(net):
#     weights = np.array([net.hidden_layers[n].w1 for n in range(net.layers)])
#     fanout_codes = np.array([net.hidden_layers[n].fanout_encoding1 for n in range(net.layers)])
#     biases =  np.array([net.hidden_layers[n].biases1 for n in range(net.layers)])
#     np.save('xor_weights', weights, allow_pickle=True) 
#     np.save('xor_biases', biases, allow_pickle=True) 
#     np.save('xor_fanout_codes', fanout_codes, allow_pickle=True) 
#     print('Saved Successfully') 

# save(temp_net) 

# weights, fanouts, biases = load()
# net.weights = weights 

# test_weights = np.linspace(-1, 1, 20)
# bit_version = np.array([wbt.simple_weight_transform(t) for t in test_weights])
# print(bit_version)
# print(test_weights)
# print([wbt.bits_to_weights(t) for t in bit_version])
# input()

#input(len(biases[0])) 


# input(np.sum(wbt.cap_bank)/(np.sum(wbt.cap_bank)+wbt.npn_capacitance))

# print(weights)

# for w in weights:
#     for i in range(np.shape(w)[0]):
#         for j in range(np.shape(w)[1]):
#             w[i][j] = wbt.bits_to_weights(wbt.simple_weight_transform(w[i][j])) 

# print(weights)
# input()


#biases[-1] = [biases[-1]]
# net.calibrate_network_biases([1, 1], biases)
# print('calibration successful?')

#net.Save() 
net.Load()
# input(net.weights)
# B = [n.vref for n in net.input_layer] 
# input(B)

# input(np.average(net.activate([1, 1, 1]))) 
# input(np.average(net.activate_burst([1, 1, 1], 1000))) 
# input(np.average(net.activate_burst([1, 1, 1], 1000))) 
# input(np.average(net.activate_burst([1, 1, 1], 1000))) 
# input(np.average(net.activate_burst([1, 1, 1], 1000))) 
# input(np.average(net.activate_burst([1, 1, 1], 1000))) 
#net.weights = net.weights.reverse()
n = net.output_layer[0]

# net.input_layer[0].update_vref(1)
# net.input_layer[1].update_vref(1)
# net.input_layer[2].update_vref(1) 

#input(biases) 
#n.update_vref(100) 


#input('Done')

#net.output_layer[0].update_vref(40) 

n1 = net.input_layer[0]
n2 = net.input_layer[1] 
#n3 = net.input_layer[2] 

n1.update_vref(.05)
n2.update_vref(.05) 
#n3.update_vref(.05)

# vals = net.input_layer[0].forward_burst(1, 1)
# input(net.input_layer[0].vref)

O = []
graphs = []
for i in xor_inputs2:
    graph = net.activate_burst(i, 10000)
    #graph = net.hidden_layer_burst(i, 10000, 1) 
    graphs.append(graph) 
    O.append(np.average(graph, axis=0))
    #O.append(np.average(graph, axis=1))
    #O.append(graph)
#o1 = net.activate_burst([1, 0, 0], 10000)
#f = np.fft.fft(o, axis=0) 
x = range(10000)
#print(np.fft.fftfreq(1000))
#f1 = np.fft.fft(o1, axis=0) 

#
#input(np.shape(o1))
# plt.plot(x, f, color='red')
# plt.plot(x, f1, 'green')
# plt.show()

#o = net.hidden_layer_burst([1, 1, 0], 1000, 0) 
# u = net.output_layer[0].INPUT_STREAM 
# t = net.hidden_layers[0][0].output_stream
#avg = np.average(o)#, axis=0) 
#print(avg)
for i in range(len(xor_inputs2)):
    print(xor_inputs2[i], ': ', O[i], '; ', xor_outputs2[i]) 



# plt.plot(range(len(n1.vco_bias_stream)), n1.vco_bias_stream)
# plt.plot(range(len(n2.vco_bias_stream)), n2.vco_bias_stream, color='magenta')
# plt.show()

fig, axes = plt.subplots(len(xor_inputs2), 1) 

for i in range(len(xor_inputs2)):
    axes[i].plot(x, graphs[i]) 
    #axes[i].title(str(xor_inputs2[i])) 

plt.show() 


#input()
#x = range(1000) 
#x = range(net.hidden_layers[0][0].stream_max)
#x = range(net.output_layer[0].stream_max) 

#plt.plot(x, t, color = 'blue')
#plt.plot(x, avg*np.ones(len(x)), color='red')
# plt.plot(x, net.output_layer[0].INPUT_STREAM)
# plt.plot(x, net.output_layer[0].vco_bias_stream, color='green')
# plt.plot(x, net.output_layer[0].output_stream, color='red')  
# plt.show()

print('Done.') 



