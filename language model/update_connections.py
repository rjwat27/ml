import numpy as np


con = np.load('connections.npy', allow_pickle=True).item() 
prim = np.load('primitives.npy', allow_pickle=True).item() 
samples = np.load('phrases_small.npy', allow_pickle=True)


#only unidirectional for now
def sample_update(sample):
    words = [s for s in sample if (s in prim)] 
    if len(words) > 1:
        for s in words:
            for t in words[1:]:
                if (s in prim) and (t in prim):
                    con[s][t] = .1 if con[s][t]==None else con[s][t] + .1 

iter = 0
for s in samples:
    iter += 1
    print(iter) 
    sample_update(s.split())

count = [c for c in con if len(con[c]) > 0]
print(len(count))

# np.save('connections', con, allow_pickle=True) 

# print('Done') 


