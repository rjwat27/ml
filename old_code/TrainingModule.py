import numpy as np
import json, os

import NNModule as nn
import neat
from neat import genes
import ArchetypeModule as am 

local_dir = os.path.dirname(__file__)
config_file = os.path.join(local_dir, 'config-feedforward.cfg')


def fitness(genomes, training_data, config):
    for genome_id, genome in genomes:
        if not training_data:
            continue 
        fitness = len(training_data) 
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
         
        for i in range(len(training_data)-1):
            output = net.activate(training_data[i][0]) 
           
            for j in range(len(output)):
                fitness -= ([training_data[i][1]][j] - output[j])**2 /2

        genome.fitness = fitness / len(training_data) #(fitness - 100) / (max - 100) 

def string_back_to_tuple(s):
    a = s.find(',')
    b = s.find(')') 
    A = s[1:a]
    B = s[a+1:b] 
    A.replace(' ', '') 
    B.replace(' ', '') 
    A = int(A) 
    B = int(B) 
    return (A, B) 

def train(archetype, generations=1):

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
   
    p = neat.Population(config)
    #print(p.species.species[1].best.nodes)
    if archetype.expert != 0:
        expert = archetype.expert
        nodes = expert["nodes"]
        connections = expert["connections"]
        #for 10 members of initial population, modify to be replicas of expert 
        # test = p.species.species[1].members[1].connections[(-1, 0)]
        # print(test.__dict__) 
        #input() 
        for i in range(10):
            for n in nodes:
                params = {'key': int(n), 'bias': nodes[n]["bias"], 'response': 1.0, 'activation': 'sigmoid', 'aggregation': 'sum'}
                p.species.species[1].members[i+1].nodes[n] = genes.DefaultNodeGene(int(n))
                #p.species.species[1].members[i+1].nodes[n].init_attributes(params) 
                
            for c in connections: 
                c = string_back_to_tuple(c)  
                params = {'key': c, 'weight': connections[str(c)]["weight"], 'enabled': True}  #problems with gene class attibutes
                p.species.species[1].members[i+1].connections[c].weight = 1#genes.DefaultConnectionGene(c) 
                p.species.species[1].members[i+1].connections[c].init_attributes(params) 

    else:
        nodes = []
        connections = [] 
    

    best = p.run(fitness, archetype.training_data, generations) 
    
    archetype.set_expert(best) 

class trainer():
    def __init__(self, size):
        self.recent= []
        self.size = size

    def log(self, data):
        self.recent.append(data) 
        if len(self.recent) > self.size:
            self.recent.pop(0) 

    def train_2(arch):       #backpropogation without neat algo 
        a = arch
        sample_set = [a.training_data[i][0] for i in range(len(a.training_data))] 
        answer_set = [a.training_data[i][1] for i in range(len(a.training_data))] 
        result = nn.GD(a.size, 3, 10, 5, [], [], sample_set, answer_set)
        a.expert = result[0]




