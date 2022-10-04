import numpy as np

def activate(x):
	return(1/(1+np.exp(-x)))
learningrate = 10
bias1 = 0
bias2 = 0
connection1 = np.random.rand()*2 - 1
connection2 = np.random.rand()*2 - 1

input1 = np.random.rand()*2 - 1 
input2 = 1.1
correct = 0.75
class Node:
    def __init__(self, key, bias):
        self.key = key
        self.bias = bias 
        self.input = 0
        self.delta = -1 

    def activate(x):
	    return(1/(1+np.exp(-x)))*2 - 1

class Connection:
    def __init__(self, key, weight, enabled):
        self.key = key
        self.weight = weight
        self.enabled = enabled 

def backpropogate(error, connection1, connection2):
    connection1 += learningrate*error
    connection2 += learningrate*error
    return connection1, connection2 
    
	
    

for i in range(100):
        output = activate(input1*connection1) 
        if input1 > 0:
            correct = 1
        else:
            correct = 0 
        error = correct - output
        connection1, connection2 = backpropogate(error, connection1, connection2) 
        print(input1)
        print("output: ", output) 
        input1 = np.random.rand()*2 - 1 
print(type(backpropogate)) 