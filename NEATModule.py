"""
NEAT Module 
"""

from __future__ import print_function
import matplotlib
import os
import neat, json
import numpy as np 
import NNModule 
#import visualize 

batchsize = 3

file = open("training_data2.txt", 'r') 
data = json.load(file) 


def randomdata():
    data_list = [] 
    for i in range(batchsize):
        data_list.append(json.loads(data[str(i)]))
        wallet = 100 
    prices_high = []
    prices_low = [] 
    volumes = []
    counts = [] 
    for tick in data_list:
        change = float(tick['priceChangePercent']) 
        top = float(tick['highPrice']) 
        low = float(tick['lowPrice'])
        volume = float(tick['volume']) 
        count =  float(tick['count'])
        prices_high.append(top)
        prices_low.append(low)
        volumes.append(volume)
        counts.append(count) 
        if change > 0:
            wallet = wallet*((change/100)+1)
    max = wallet 
    #normalize 
    prices_high_bounds = (np.max(prices_high), np.min(prices_high))
    prices_low_bounds = (np.max(prices_low), np.min(prices_low))  
    volumes_bounds = (np.max(volumes), np.min(volumes))
    counts_bounds = (np.max(counts), np.min(counts)) 
    for tick in data_list:
        top = float(tick['highPrice']) 
        low = float(tick['lowPrice'])
        volume = float(tick['volume']) 
        count =  float(tick['count'])

        tick['highPrice'] = (top - prices_high_bounds[1])/(prices_high_bounds[0] - prices_high_bounds[1]) 
        tick['lowPrice'] = (low - prices_low_bounds[1])/(prices_low_bounds[0] - prices_low_bounds[1])
        tick['volume'] = (volume - volumes_bounds[1])/(volumes_bounds[0] - volumes_bounds[1]) 
        tick['count'] = (count - counts_bounds[1])/(counts_bounds[0] - counts_bounds[1])  

#print(data_list) 
# # 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (1.0,),     (0.0,),     (1.0,),     (1.0,)]
data_list = 0 

def eval_genomes(genomes, training_data, config):#, predictions, actuals):
    for genome_id, genome in genomes:
        fitness = len(training_data) 
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
         
        for i in range(len(training_data)-1):
            
            fitness -= (training_data[i][1] - training_data[i][0])**2 /2

        genome.fitness = fitness / len(training_data) #(fitness - 100) / (max - 100)   


def run(config_file, NN):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Create the population, which is the top-level object for a NEAT run.
    
    p = neat.Population(config)
 

    # Add a stdout reporter to show progress in the terminal.
    #p.add_reporter(neat.StdOutReporter(False))
    #stats = neat.StatisticsReporter()
    #p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))
    NN = NNModule.FNN(2, 1, 0, 0, [], []) 
    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 100, NN)

    # Display the winning genome.
    #print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    # print('\nOutput:')
    
    #print(p.species.species[1].__dict__)    #important
    
    #print(p.species.species[1].__dict__) 
    #print(config.genome_config.num_outputs) 
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #winner.nodes[0].bias = 1.1111111
    return winner
 

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    NN = None
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.cfg')
    winner = run(config_path, NN)
    