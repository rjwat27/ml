
import numpy as np 


import UIModule as ui 



'''here i will try to cascade networks, training one after another until a task is learned'''

num_inputs = 4

num_classes = 2
xor_outputs = [0, 1, 1, 0] 
'''create num_inputs-bits logic table'''
def make_inputs_outputs(num_inputs, num_classes):
    inputs = []
    w = len(np.binary_repr(2**(num_inputs)-1))
    for i in range(2**(num_inputs)):
        binary = np.binary_repr(i, width=w) 
        entry = []
        for j in range(len(binary)):
            entry.append(int(binary[j])) 

        inputs.append(entry) 

    output_classes = range(num_classes) 
    outputs = [np.random.choice(output_classes) for i in range(len(inputs))] 

    return inputs, outputs 

def make_xor_outputs(inputs):
    o = []
    for i in inputs:
        if np.sum(i) % 2 == 0:
            o.append(0)
        else:
            o.append(1)

    return o 


iterations = []
success = []
failure = [] 

#ui.ryan_test()
inputs, outputs = make_inputs_outputs(num_inputs, num_classes)
outputs = make_xor_outputs(inputs) 
net = ui.test_run(inputs, outputs) 

#print('num biases: ', len(net.biases1)) 

# for i in range(10):
#     print('Trial ', i) 
#     inputs, outputs = make_inputs_outputs(num_inputs, num_classes) 
#     iter, biases, error, net = ui.main()
#     # for a in net.w1:
#     #     print(a)  
#     if error < .2:
#         success.append(biases)
#     else:
#         failure.append(biases) 
#     print('main iter: ', iter) 
#     print('w2: ', net.w2) 
#     input() 
#     iterations.append(iter) 

# with open('bias analysis result.txt', 'w') as file:
#     opener = '-----Results-----\n\nLength of success:' + str(len(success)) + '\nLength of Failure: ' + str(len(failure)) 
#     results = '\n\nSuccess biases: \n\n' + str(success) + '\n\nFailure biases: \n\n' + str(failure) 

#     write_string = opener + results
#     file.write(write_string) 

print('Finished.')
#print('Results: ', iterations) 












