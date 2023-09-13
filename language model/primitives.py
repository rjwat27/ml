import numpy as np


nouns = np.load('nouns.npy', allow_pickle=True) 
verbs = np.load('verbs.npy', allow_pickle=True) 
adj = np.load('adj.npy', allow_pickle=True) 

#plurality: multiply by 2**15
SINGULAR = 0 
PLURAL = 1  

#time/tense: multiply by 2**13
PRESENT = 0 
PAST = 1 
FUTURE = 2 
COND = 3  

#types: multiply by 2^10
THING = 0 
VERB = 1 
ADJ = 3 



#create primitives and initiate mappings
primitives = {}
dictionary = {} #ascii words to primitives/primitive compositions
associations = {}

idx = 1
for n in nouns:
    prim = np.uint16(idx)
    dictionary[n] = prim 
    idx += 1

for v in verbs:
    prim = np.uint16(idx + VERB*(2**10))
    dictionary[v] = prim
    idx += 1 

for a in adj:
    prim = np.uint16(idx + ADJ*(2**10)) 
    dictionary[a] = prim 
    idx += 1

for d in dictionary:
    primitives[dictionary[d]] = d

print(len(dictionary))

connections = {}
for p in primitives:
    connections[p] = {} 


np.save('connections', connections, allow_pickle=True) 
np.save('dictionary', dictionary, allow_pickle=True)
np.save('primitives', primitives, allow_pickle=True) 















