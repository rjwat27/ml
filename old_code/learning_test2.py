import numpy as np
import NNModule as nn 
import learning_test3 as lt 
learningrate = .1
def make_data(dim, cat, size):
    data = {}
    for i in range(size):
        a = np.random.choice(range(cat))
        data[i] = [np.random.rand(dim), np.zeros(cat)]
        data[i][1][a] = 1 
    return data

and_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
#and_outputs = [1 ,0, 1, 1]
and_outputs = [[1, 0] ,[0, 1], [1, 0], [1, 0]]
con = {}
nodes = {}

# class Node():
#     def __init__(self, key, bias):
#         self.key = key
#         self.bias = bias
#         self.input = 0
#         self.output = 0
#         self.delta = 0
#     def activate(self, x):
#         result = relu(x + self.bias) 
#         self.output = result
#         self.input = x
#         return result 
#     def adjust(self, error):
#         self.bias += error*learningrate*relu_der(self.input + self.bias) 

# class End_Node():
#     def __init__(self, key, bias):
#         self.key = key
#         self.bias = bias
#         self.input = 0
#         self.output = 0
#         self.delta = 0
#     def activate(self, x):
#         self.input = x
#         self.output = x
#         return relu(x)#+self.bias)  
#     def adjust(self, error):
#         self.bias += error*relu_der(self.input + self.bias)*learningrate 

# class Connection():
#     def __init__(self, key, weight):
#         self.key = key
#         self.weight = weight
#     def adjust(self, error, node_start, node_end):
#         addend = relu_der(node_end.input + node_end.bias) * node_start.activate(node_start.input) 

#         self.weight += error*learningrate*addend 
    
#         # if self.weight > 1:
#         #     self.weight = 1
#         # elif self.weight < -1:
#         #     self.weight = -1
#         return error 

# class Layer():
#     def __init__(self, nodes):  #list of node objects 
#         self.nodes = nodes
#         self.size = len(nodes) 
#     def output(self, input):
#         return [self.nodes[i].activate(input[i]) for i in range(len(input))]

# def sigmoid(x):
#     return (1 / (1+np.exp(-x)))
# def sig_der(x):
#     return sigmoid(x)*(1-sigmoid(x)) if x>0 else .1  
# def relu(x):
#     return x if x>0 else 0   

# def relu_der(x):
#     if x > 0:
#         return 1 
#     else:
#         return .1

# def recursive_activate(node, input1, nodes, con):    #pass the node object
#     if node.key in range(-len(input1), 0):
#         print(input1[-node.key-1]) 
#         return node.activate(input1[-node.key-1]) 
#     else:
#         x = 0
#         for c in con:          
#             if c[1]==node.key:             
#                 x += recursive_activate(nodes[c[0]], input1, nodes, con)*con[c].weight 
#         return node.activate(x) 

# def net_activate(input, correct, nodes, con, layers):  #still just for one hidden layer
#     # output1 = layer1.output(np.dot(input, weights[0]))
#     # pred = output_layer.output(np.dot(output1, weights[1])) 
#     # pred = np.array(pred) 
#     layer1 = layers[1]
#     layer2 = layers[2] 
    

#     pred = np.array([recursive_activate(nodes[i], input, nodes, con) for i in range(len(correct))]) #only for size 2 outputs
    
#     # n1 = nodes[1].output
#     # n2 = nodes[2].output 

#     # n1 = node1.activate(input1*con[0].weight + input2*con[2].weight) 
#     # n2 = node2.activate(input1*con[1].weight + input2*con[3].weight) 
#     # pred = n1*con[4].weight + n2*con[5].weight#sigmoid(n1*con[4].weight + n2*con[5].weight)
#     error = correct - pred
#     #calculate deltas
#     nodes[0].delta=error        #for single output
#     for n in layer2.nodes:
#         n.delta = n.output * con[(n.key, 0)].weight * error 
#     for n in layer1.nodes:
#         x = 0
#         for m in layer2.nodes:
#             x += m.delta*con[(n.key, m.key)].weight*n.output 
#         n.delta = x 
#     # delta1 = n1 * con[4].weight * error  
#     # delta2 = n2 * con[5].weight * error

#     #edit weights
#     for c in con:
#         start_node = nodes[c[0]] 
#         end_node = nodes[c[1]] 
#         con[c].adjust(end_node.delta, start_node, end_node)      #later with more hidden layers pass end node delta instead of error
#     # for c in [4, 5]:
#     #     con[c].adjust(error)
#     # for c in [0, 2]: 
#     #     con[c].adjust(delta1)
#     # for c in [1, 3]:
#     #     con[c].adjust(delta2)

#     #edit node biases 
#     for n in layer1.nodes:      #still single hidden layer
#         n.adjust(n.delta) 
#     for n in layer2.nodes:
#         n.adjust(n.delta)   

   
#     return pred, nodes, con

# def train(inputs, outputs, hidden_size, sample_set, answer_set):   
#     #create nodes
#     nodes = {} 
#     node_list = []
#     size = hidden_size

#     for i in range(-inputs, 0):
#         n = End_Node(i, 0) 
#         nodes[n.key] = n 
#         node_list.append(n) 
#     input_layer = Layer(node_list)  #input layer
    
    
#     node_list = []
    
#     domain = np.linspace(-inputs/2, inputs/2, size) 
#     for i in range(size):
#         n = Node(i+outputs, domain[i])
#         nodes[n.key] = n
#         node_list.append(n)
#     layer1 = Layer(node_list) #hidden layer 1


#     node_end = End_Node(0, 0) 

#     node_list = []

#     domain = np.linspace(-size/2, size/2, size) 
#     for i in range(size):
#         n = Node(i+outputs+size, domain[i])
#         nodes[n.key] = n
#         node_list.append(n) 
#     layer2 = Layer(node_list)   #hidden layer 2

#     node_list = []

#     domain = np.linspace(-size/2, size/2, outputs) 
#     for i in range(outputs):
#         n = Node(i, 0)#domain[i]) 
#         nodes[n.key] = n
#         node_list.append(n)
#     if len(node_list)==1:
#         output_layer = Layer([node_end])  #brackets since output singular
#     else:
#         output_layer = Layer(node_list)     #output layer

#     layers = [input_layer, layer1, layer2, output_layer] 

#     #create connections... first hidden layer
#     con = {}

#     for i in range(-input_layer.size, 0):
#         for j in range(outputs, layer1.size+outputs):
#             if i != j:
#                 c = Connection((i, j), np.random.rand()*2-1) 
#                 con[c.key] = c
#     #connections... second hidden layer

#     for i in range(outputs, layer1.size+outputs):
#         for j in range(layer1.size+outputs, layer1.size+layer2.size+outputs):
#             if i != j:
#                 c = Connection((i, j), np.random.rand()*2-1) 
#                 con[c.key] = c 

#     #connections... output layer... only for one output 

#     for i in range(layer1.size+outputs, layer1.size+layer2.size+outputs):
#         for j in range(outputs):
#             if i != j:
#                 c = Connection((i, j), np.random.rand()*2 - 1) 
#                 con[c.key] = c 

   
#     total_error = 1
#     best=None
#     count = 1
#     while abs(total_error) > .1 and count < 1000:
#         for a in range(len(sample_set)):
#             pred = np.array([recursive_activate(nodes[i], sample_set[a], nodes, con) for i in range(outputs)])  
#             if abs(answer_set[a] - pred).all() >= .01:
                
#                 for i in range(3):
#                     input2 = sample_set[a]
#                     correct = answer_set[a] 
#                     pred, nodes, con = net_activate(input2, correct, nodes, con, layers)
#                     # for c in con:
#                     #     print(c, ': ', con[c].weight) 
#                     # input()
#                     #print('pred: ', pred) 
#             total_error=0
#             for i in range(4):
#                 ans = np.array([recursive_activate(nodes[i], sample_set[a], nodes, con) for i in range(outputs)])
#                 total_error += abs(answer_set[i] - ans)  
#                 #print(sample_set[i], ': ', ans) 
#             if best is None or total_error < best:
#                 best = total_error 
#         count += 1
#     return nodes, con, best, count 
  
  


#nodes, con, best, count = train(2, 2, 8, and_inputs, and_outputs) 

trainer = lt.Trainer(2, 2, and_inputs, and_outputs) 
success = 0
for i in range(1):
    nodes, con, best, count = trainer.train(4, 1000) 
    #print(i, ': ', best) 
    if best < .5:
        success += 1
#print('Success Rate: ', success/100) 
# print('tries: ', count)
# print('best error: ', best) 

print('and gate predictions: ')
for i in range(4):
    print(and_inputs[i], ': ', np.array([trainer.recursive_activate(trainer.nodes[j], and_inputs[i]) for j in range(trainer.outputs)]))#trainer.recursive_activate(nodes[0], and_inputs[i]))
    #print(and_inputs[i], ': ', trainer.recursive_activate(nodes[0], and_inputs[i])) 
    
    