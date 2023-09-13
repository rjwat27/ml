
import numpy as np 


#test moving market distribution model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
#ax2 = fig.add_subplot(2,1,2) 

x = np.linspace(0, 10, 3000) 

u1 = 2
u2 = 7

S = 2
B = 2.4 

v = .12

sig = .95

buyers = np.exp(-(x-u1)**2/2) * B
sellers = np.exp(-(x-u2)**2/2) * S


# plt.plot(x, buyers)
# plt.plot(x, sellers) 
# plt.show()

def speculation():
    return np.sum(buyers*x) - np.sum(sellers*x) 

beta = .001
alpha = .001
def update_buyers_sellers(spec):
    new_b = buyers[:]
    new_s = sellers[:]

    new_b = np.maximum(np.zeros(len(x)), buyers - sellers) 
    new_s = np.maximum(np.zeros(len(x)), sellers - buyers)

    new_bb = np.zeros(len(x)) 
    new_ss = np.zeros(len(x))
    #return new_bb, new_ss
    for i in range(len(new_b)-1):
        new_bb[i+1] += sig*new_b[i]
        new_ss[i+1] += (1-sig)*new_s[i]

    for i in range(1, len(new_bb)):
        new_bb[i-1] += (1-sig)*new_b[i] 
        new_ss[i-1] += sig*new_s[i]
    #speculation shift 
    # buyer_shift = int(beta*spec)
    # seller_shift = int(alpha*spec) 

    # new_bb = new_b[:]
    # new_ss = new_s[:] 

    # new_bb[buyer_shift:] = new_bb[0:-buyer_shift] 
    # if buyer_shift > 0:
    #     new_bb[-1] += np.sum(new_b[-buyer_shift:])
    # elif buyer_shift < 0:
    #     new_bb[0] += np.sum(new_b[0:buyer_shift]) 
    # else:
    #     new_bb = new_b 

    # new_ss[seller_shift:] = new_ss[0:-seller_shift] 
    # if seller_shift > 0:
    #     new_ss[-1] += np.sum(new_s[-seller_shift:])
    # elif seller_shift < 0:
    #     new_ss[0] += np.sum(new_s[0:seller_shift]) 
    # else:
    #     new_ss = new_s 

    # return new_bb, new_ss 

    return new_bb, new_ss



population = [np.sum(buyers)+np.sum(sellers)]

#plt.ion()

graph1 = ax1.plot(x, buyers, color='blue')[0]
graph2 = ax1.plot(x, sellers, color='green')[0] 
plt.ylim([0, 5])
#graph3 = ax2.plot(population)[0] 

plt.pause(1)
def animate(i):
    spec = speculation() 
    buyers, sellers = update_buyers_sellers(spec)
    #population += [np.sum(buyers)+np.sum(sellers)]
    # ax1.clear()
    # ax2.clear()
    # ax1.plot(x, buyers, color='blue')
    # ax1.plot(x, sellers, color='red') 
    # ax2.plot(population) 
    graph1.set_ydata(buyers)


    graph2.set_ydata(sellers)
    #graph3.set_ydata(population)
    #plt.draw()
    #plt.pause(.01)
    if i==1000:
        print('done.') 
ani = animation.FuncAnimation(fig, animate, interval=10, frames = 1000)
plt.show()