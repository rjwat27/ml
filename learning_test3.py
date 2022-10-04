import numpy as np
import json, os
class Node():
    def __init__(self, key, bias):
        self.key = key
        self.bias = bias
        self.input = 0
        self.output = 0
        self.delta = 0
        self.learningrate = .1
        self.sources = []
    def activate(self, x):
        result = relu(x + self.bias)
        self.output = result
        self.input = x
        return result
    def adjust(self, error):
        self.bias += error*self.learningrate*relu_der(self.input + self.bias)

class End_Node():
    def __init__(self, key, bias):
        self.key = key
        self.bias = bias
        self.input = 0
        self.output = 0
        self.delta = 0
        self.learningrate = .1
        self.sources = []
    def activate(self, x):
        self.input = x
        self.output = x
        return relu(x)#+self.bias)
    def adjust(self, error):
        self.bias += error*relu_der(self.input + self.bias)*self.learningrate

class Connection():
    def __init__(self, key, weight):
        self.key = key
        self.weight = weight
        self.learningrate = .1
    def adjust(self, error, node_start, node_end):
        #print(self.key, ': ', type(error), type(node_start.input), type(node_end.input))
        addend = relu_der(node_end.input + node_end.bias) * node_start.activate(node_start.input)

        self.weight += error*self.learningrate*addend

        # if self.weight > 1:
        #     self.weight = 1
        # elif self.weight < -1:
        #     self.weight = -1
        return error

class Layer():
    def __init__(self, nodes):  #list of node objects
        self.nodes = nodes
        self.size = len(nodes)
    def output(self, input):
        return [self.nodes[i].activate(input[i]) for i in range(len(input))]

class Trainer():
    def __init__(self, inputs, outputs, sample_set=[], answer_set=[], learningrate=.1, feedback_function=None, filename='neural_network_data.txt'):
        self.inputs = inputs
        self.outputs = outputs
        self.learningrate = learningrate


        self.sample_set = sample_set
        self.answer_set = answer_set

        self.nodes = {}
        self.con = {}

        self.layers = []

        self.current_state = 100    #for reinforcement learning
        self.feedback_function = feedback_function
        self.current_state_list = []

        self.convolution_feedback = []


        #for saving and loading data
        self.filename = filename

    def set_sample_set(self, samples):
        self.sample_set = samples

    def prepare_node_sources(self):
        for c in self.con:
            self.nodes[self.con[c].key[1]].sources.append(self.con[c].key[0])

    def sigmoid(self, x):
            return (1 / (1+np.exp(-x)))
    def sig_der(self, x):
            return sigmoid(x)*(1-sigmoid(x)) if x>0 else .1
    def relu(self, x):
            return x if x>0 else .1
    def relu_der(self, x):
            if x > 0:
                return 1
            else:
                return .1

    def apply_feedback(self, pred, input1):
        #reinforcement error technique
        '''verify input1[-1] i.e. last element of input is indeed last price...turns out it wasnt'''
  
        feedback = self.feedback_function(pred, input1)   #opportunity awareness function later
        reward = feedback# - self.current_state)

        return reward
        self.current_state = feedback
        max = np.max(pred)
        for p in range(self.outputs):
            if max ==  pred[p]:
                response = [0 for i in range(self.outputs)]
                response[p] = reward
                return response
        return [0 for i in range(self.outputs)] #only does nothing if all outputs are equal
        # if max(pred)==pred[0]:
        #     return [feedback, 0]
        # elif max(pred)==pred[1]:
        #     return [0, feedback]

    def recursive_activate(self, node, input1):#, nodes, con):    #pass the node object
        #perhaps later implement node connection memory for speed
        if node.key in range(-len(input1), 0):
            return node.activate(input1[-node.key-1])
        else:


            x = 0
            # for c in self.con:
            #     if c[1]==node.key:
            #         x += self.recursive_activate(self.nodes[c[0]], input1)*self.con[c].weight

            for n in node.sources:
                x += self.recursive_activate(self.nodes[n], input1)*self.con[(self.nodes[n].key, node.key)].weight

            return node.activate(x)

    def net_activate(self, input1, correct):  #still just for one hidden layer
        layer1 = self.layers[1]
        layer2 = self.layers[2]
        output_layer = self.layers[3]
        nodes = self.nodes
        con = self.con

        #faster to just pass prediction as argument than re-evaluate...
        pred = np.array([self.recursive_activate(self.nodes[i], input1) for i in range(self.outputs)]) #only for size 2 outputs

        error = correct - pred


        # print('correct: ', correct)
        # print('pred: ', pred)
        #print('error', error)
        # input()
        #calculate deltas
        for i in range(self.outputs):
            self.nodes[i].delta = error[i]
        for n in layer2.nodes:
            x = 0
            for m in output_layer.nodes:
                x += m.delta*con[(n.key, m.key)].weight * n.output
            n.delta = x
        for n in layer1.nodes:
            x = 0
            for m in layer2.nodes:
                x += m.delta*con[(n.key, m.key)].weight*n.output
            n.delta = x

        #edit weights
        for c in con:
            start_node = nodes[c[0]]
            end_node = nodes[c[1]]
            #print(c, ': ', type(con[c].weight))
            con[c].adjust(end_node.delta, start_node, end_node)      #later with more hidden layers pass end node delta instead of error

        #edit node biases
        for n in layer1.nodes:      #still single hidden layer
            n.adjust(n.delta)
        for n in layer2.nodes:
            n.adjust(n.delta)


        return pred, nodes, con

    def train(self, hidden_size, iter=100):
        if not self.nodes:
            self.create_new_net(hidden_size)
        self.prepare_node_sources()
        inputs = self.inputs
        outputs = self.outputs

        total_error = 1
        best=None
        count = 1
        while abs(total_error) > .1 and count < iter:
            for a in range(len(self.sample_set)):
                pred = np.array([self.recursive_activate(self.nodes[i], self.sample_set[a]) for i in range(outputs)])
                if abs(self.answer_set[a] - pred).all() >= .01:
                    for i in range(3):
                        input2 = self.sample_set[a]
                        correct = self.answer_set[a]
                        error = correct - pred
                        pred, nodes, con = self.net_activate(input2, correct)
            total_error=0
            for i in range(4):
                ans = np.array([self.recursive_activate(self.nodes[i], self.sample_set[a]) for i in range(outputs)])
                total_error += np.sum(abs(self.answer_set[i] - ans))

            if best is None or total_error < best:
                best = total_error
            count += 1

        return self.nodes, self.con, best, count

    def reinforcement_train(self, hidden_size, iter=100): #state reader outside function to pass current state
        if not self.nodes:
            self.create_new_net(hidden_size)
        self.prepare_node_sources()
        inputs = self.inputs
        outputs = self.outputs

        total_error = 1
        best=None
        count = 1
        while abs(total_error) > .1 and count < iter:
            print('iteration: ', count) 
            for a in range(len(self.sample_set)):

                pred = np.array([self.recursive_activate(self.nodes[i], self.sample_set[a]) for i in range(outputs)])

                input2 = self.sample_set[a]
                #print(self.current_state_list) 
                feedback = self.apply_feedback(pred, self.current_state_list[a])
                print(pred)
                if True:#result >= .01:
                    for i in range(3):
                        pred, nodes, con = self.net_activate(input2, feedback)
            #total_error=0
            # for i in range(4):
            #     ans = np.array([self.recursive_activate(self.nodes[i], self.sample_set[a]) for i in range(outputs)])
            #     total_error += np.sum(abs(self.answer_set[i] - ans))

            if best is None or total_error < best:
                best = total_error
            count += 1

        return self.nodes, self.con, best, count

    def update_convolution_feedback(self, next_input, next_state):
        max_size = 18 #may change max size later

        current_size = len(self.convolution_feedback)
        self.convolution_feedback.append((next_input, next_state)) 

        if current_size == max_size:
            self.convolution_feedback.pop(0) 

    def convolved_feedback(self):
        avg = 0
        size = len(self.convolution_feedback)
        for c in self.convolution_feedback:
            avg += c[1] 
        return avg/size 


    def create_new_net(self, hidden_size):
        #create nodes
        inputs = self.inputs
        outputs = self.outputs
        nodes = {}
        node_list = []
        size = hidden_size

        for i in range(-inputs, 0):
            n = End_Node(i, 0)
            self.nodes[n.key] = n
            node_list.append(n)
        input_layer = Layer(node_list)  #input layer


        node_list = []

        domain = np.linspace(-inputs/2, inputs/2, size)
        for i in range(size):
            n = Node(i+outputs, domain[i])
            self.nodes[n.key] = n
            node_list.append(n)
        layer1 = Layer(node_list) #hidden layer 1


        #node_end = End_Node(0, 0)

        node_list = []

        domain = np.linspace(-size/2, size/2, size)
        for i in range(size):
            n = Node(i+outputs+size, domain[i])
            self.nodes[n.key] = n
            node_list.append(n)
        layer2 = Layer(node_list)   #hidden layer 2

        node_list = []

        domain = np.linspace(-size/2, size/2, outputs)
        for i in range(outputs):
            n = Node(i, 0)#domain[i])
            self.nodes[n.key] = n
            node_list.append(n)
        if len(node_list)==1:
            output_layer = Layer([node_list])  #brackets since output singular
        else:
            output_layer = Layer(node_list)     #output layer

        self.layers = [input_layer, layer1, layer2, output_layer]

        #create connections... first hidden layer
        con = {}

        for i in range(-input_layer.size, 0):
            for j in range(outputs, layer1.size+outputs):
                if i != j:
                    c = Connection((i, j), np.random.rand()*2-1)
                    self.con[c.key] = c
        #connections... second hidden layer

        for i in range(outputs, layer1.size+outputs):
            for j in range(layer1.size+outputs, layer1.size+layer2.size+outputs):
                if i != j:
                    c = Connection((i, j), np.random.rand()*2-1)
                    self.con[c.key] = c

        #connections... output layer... only for one output

        for i in range(layer1.size+outputs, layer1.size+layer2.size+outputs):
            for j in range(outputs):
                if i != j:
                    c = Connection((i, j), np.random.rand()*2 - 1)
                    self.con[c.key] = c

    def input_sensor(self, input1):
        self.inputs = len(input)

    def string_to_tuple(self, str):
        begin = 1
        end = str.find(',', begin)
        a = int(str[begin:end])
        begin = end+1
        end = str.find(')', begin)
        b = int(str[begin:end])
        return (a, b)
    def Save(self):
        file = open(self.filename, 'w')
        #for now only saves node and connection data
        data = {}
        node_data = {}
        for n in self.nodes:
            n = self.nodes[n]
            node_data[n.key] = n.bias
        connection_data = {}
        for c in self.con:
            c = self.con[c]
            connection_data[str(c.key)] = c.weight
        data['nodes'] = node_data
        data['connections'] = connection_data
        json.dump(data, file)

    def Load(self, fname=None):
        if fname==None:
            fname = self.filename
        local_dir = os.path.dirname(__file__)
        file_path = os.path.join(local_dir, fname)

        file = open(file_path, 'r')
        data = json.load(file)

        nodes = data['nodes']
        connections = data['connections']

        for n in nodes:
            self.nodes[int(n)] = Node(int(n), float(nodes[n]))

        for c in connections:
            key = self.string_to_tuple(c)
            self.con[key] = Connection(key, connections[c])

    def run_validation_set(self, validation_set):
        data = validation_set
        decisions = []
        for d in data:
            act = np.array([self.recursive_activate(self.nodes[i], d) for i in range(self.outputs)])
            decisions.append(act)
        return decisions

def sigmoid(x):
        return (1 / (1+np.exp(-x)))
def sig_der(x):
        return sigmoid(x)*(1-sigmoid(x)) if x>0 else .1
def relu(x):
    #print('input: ', x)
    return x if x>0 else 0
def relu_der(x):
        if x > 0:
            return 1
        else:
            return .1
