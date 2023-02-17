import numpy as np
from matplotlib import pyplot as plt
import time
from time import sleep 
import NNModule as nn 
import VisualizeModule as vm
class cell():
    def __init__(self, color):
        self.color = color
        self.elements = [] 

n = 4
cells = {}
for i in range(n**2):
    #a = np.random.choice(['red', 'green'])
    #c = cell(a) 
    if i % 2 == 0:
        c = cell('red')
        cells[0] = c 
    else:
        c = cell('red') 
        cells[1] = c 

xmax = 100
ymax = 100
net = None
data = []
for i in range(1000):
    x = np.random.choice(range(100)) 
    y = np.random.choice(range(100)) 
    point = [x, y]
    data.append(point) 

for d in data:
    x = int(d[0] / 25) if d[0]/25!=4 else 3 
    y = int(d[1] / 25) if d[1]/25!=4 else 3 
    # c = (4)*(y) + x 
    # cells[c].elements.append(d) 
    if x < 2:
        cells[0].elements.append(d)
    else:
        cells[1].elements.append(d) 

sample_set = []
answer_set = []
for c in cells:
    c = cells[c]
    for d in c.elements:
        sample_set.append(d)
        answer_set.append(1) if c.color=='green' else answer_set.append(0)

def train(sample_set, answer_set, iter, net=None):
    for i in range(iter):
        if net==None:
            net = nn.FNN(2, 2, 4, 0, [], []) 
            result = nn.GD2(net, sample_set=sample_set, answer_set=answer_set, batchsize=1) 
            net = result[0]
            error = result[1] 
        else:
            result = nn.GD2(net, sample_set=sample_set, answer_set=answer_set, batchsize=1) 
            net = result[0]
            error = result[1] 

    return net, error

new_samples = []
new_answers = []
#normalize
for i in range(len(sample_set)):
    index = i#np.random.choice(range(len(sample_set))) 

    new_samples.append([sample_set[index][0]/100, sample_set[index][1]/100]) 
    new_answers.append(answer_set[index]) 

sample_set = new_samples
answer_set = new_answers  
#randomize samples
def randomize(d1, d2):
    new_samples = []
    new_answers = []
    sample_set = d1
    answer_set = d2 
    for i in range(len(sample_set)):
        index = np.random.choice(range(len(sample_set))) 
        new_samples.append(sample_set[index]) 
        new_answers.append(answer_set[index]) 
        sample_set.pop(index) 
        answer_set.pop(index) 
    return new_samples, new_answers

#sample_set, answer_set = randomize(sample_set, answer_set) 




# import matplotlib.pyplot as plt

# plt.figure()
# plt.xlim(0, 5)
# plt.ylim(0, 5)

# for i in range(0, 5):
#     plt.axhspan(i, i+.2, facecolor='0.2', alpha=0.5)
#     plt.axvspan(i, i+.5, facecolor='b', alpha=0.5)

# ax.fill(x, y)                    # a polygon with default color
# ax.fill(x, y, "b")               # a blue polygon
# ax.fill(x, y, x2, y2)            # two polygons
# ax.fill(x, y, "b", x2, y2, "r")  # a blue and a red polygon

# plt.show()
vm.graph2(sample_set, answer_set, 'test2.png') 
#visualize learning 
old_bias = []
new_bias = []
for i in range(100): 
    sample_set, answer_set = randomize(sample_set, answer_set) 

    result = train(sample_set, answer_set, 1, net) 
    net = result[0]
    error = result[1] 

for node in net.nodes:
    n = net.nodes[node]
    print('node ', node, ": ", n.bias) 
for conn in net.connections:
    print('connection: ', conn, ': ', net.connections[conn].weight, ': ', net.connections[conn].enabled)  
print('error: ', error) 

# for j in range(1000):
#     print('point', sample_set[j]) 
#     print('answer: ', answer_set[j])
            
#     print('guess: ', net.activate(sample_set[j]), '\n\n')
#     sleep(1) 
input()  
ymax = 100
ymin = 0 
res = 4
shift = 96 / (res)
for x in np.linspace(2, 98, res):
    for y in np.linspace(2, 98, res):
        output = net.activate([x/100,y/100])
            
    print(output) 
    #         c = abs(output[0] - output[1])  
    #         d = [x-shift, x-shift, x+shift, x+shift]
    #         r = [y+shift, y-shift, y+shift, y-shift] 
    #         if output[0] > output[1]:
    #             plt.axvspan(xmin=x-shift, xmax=x+shift, ymin=y-shift, ymax=y+shift, color=(0, 0, c), alpha=.1, zorder=-1) 
    #             #plt.fill(d, r, color=(c, 0, 0), alpha=.1) 
    #         else:
    #             plt.axvspan(xmin=x-shift, xmax=x+shift, ymin=y-shift, ymax=y+shift, color=(c, 0, 0), alpha=.1, zorder=-1) 
    #             #plt.fill(d, r, color=(0, 0, c), alpha=.1) 
  
    # plt.savefig('test2.png') 
                #plt.axvspan(xmin=x-shift, xmax=x+shift, ymin=y-shift, ymax=y+shift, color=(c, 0, 0), alpha=.1) 

y_list = []







