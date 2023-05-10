import numpy as np
from matplotlib import pyplot as plt 
import market_forecast.DataModule as dm, ArchetypeModule as am  
#module for graphical representation of bot performance

#learning visual
Archetype_Manager = am.Archetype_Manager("archetype_data.txt") 
Watcher = dm.Watcher() 
DataBase = dm.DataBase('BTCUSD', "tick_data_BTCUSD.txt")

#DataBase.Load() 



#price graph with decisions

def graph(history, decisions, image = 'bitcoin.png'):       #verify timing 
    if len(history) > 100:
        history = history[-100:-1]
        decisions = decisions[-100:-1] 
    x = range(len(history))
    #fig = plt.figure()
    plt.clf() 
    plt.plot(x, history, color='blue')
    for i in x:
        if decisions[i-1]==1:
            plt.scatter(i, history[i], marker=5, color='green')
        elif decisions[i-1]==-1:
            plt.scatter(i, history[i], color='red', marker=5) 
    plt.savefig(image) 


def graph2(sample_set, answer_set, image):       #this is for the learning test with cell visuals

    for i in range(len(sample_set)):
        a = sample_set[i]
        if answer_set[i]==1:
            plt.scatter(a[0], a[1], marker='o', s=3, color='green') 
        else:
            plt.scatter(a[0], a[1], marker='o', s=3, color='red') 
    plt.savefig('test2.png') 

def graph3(samples, image='test2.png'):       #this is for the learning test with cell visuals

    x = [samples[0] for s in samples] 
    y = [samples[1] for s in samples] 
    #fig = plt.figure()
    plt.clf()
    plt.plot(x, y, color='red', marker='o') 

    #plotting for net learning
    plt.savefig(image) 
    
def decision_graph(history, decisions, image='bitcoin_test_set.png', plot=False):     #plot price history with buy/sell decisions 
    x = range(len(history)) 

    plt.plot(x, history, color='blue')  #decisions to be 1, 0, or -1 for buy, sell, or nothing

    for i in x:
        if decisions[i]==1:
            plt.scatter(i, history[i], marker=5, color='green')
        elif decisions[i]==-1:
            plt.scatter(i, history[i], color='red', marker=5)
    plt.savefig(image)
    if plot==True:
        plt.show() 
    

def plain_graph(history): 
    x = range(len(history)) 

    plt.plot(x, history, color='blue')

    plt.show()