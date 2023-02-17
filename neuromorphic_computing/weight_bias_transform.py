import numpy as np

from matplotlib import pyplot as plt


base_capacitance = 10e-9
npn_capacitance = 50e-15

cap_bank = np.array([base_capacitance*i for i in [.25, .5, 1, 2]]) 
print(np.sum(cap_bank))

def sig(x):
    return 1/(1+np.exp(-x)) 

def freq_from_v(self, v):
    v2 = np.ones(len(self.vco_curve))*v 
    idx = np.argmin(abs(self.vco_curve[:,0]*1000-v2))
    # print(abs(self.vco_curve[:,0]*1000-v2))
    # input()

    '''test with linear curve'''
    b = self.vco_curve[0][1]
    e = self.vco_curve[-1][1]
    #return (b + (e-b)*(v/np.shape(self.vco_curve)[0]))
    return self.vco_curve[idx][1] 

def weight_to_cap_transform(weight):
    w_bar = np.zeros(5)
    w = abs(weight)
    new_bank = cap_bank * (1-w)
    #get sign
    sign = 1 if weight>0 else 0

    val = npn_capacitance * w

    for i in range(1, 5):
        w_bar[i] = 1
        if np.dot(w_bar[1:], new_bank) > val:
            w_bar[i] = 0
            continue 
        else:
            continue 

    w_bar[0] = sign 

    return w_bar





    pass

#test weight transform
test_weight = 1

print([weight_to_cap_transform(j) for j in np.linspace(0, 1, 15)]) 



