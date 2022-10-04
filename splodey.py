import numpy as np


for i in range(epochs):
    data = {}
    for j in range(i*EPOCH_SIZE, (i+1)*EPOCH_SIZE):
        data[j] = tick_data[j+1] 
    epoch_data[i] = data 

samples = [] 
answers = [0] 

training_prices = []
data = [[database.tick_data[i+1] for i in range(begin, begin+DEPTH)] for begin in range(DATA_SIZE-DEPTH)] 

#from data extract every 25 or DEPTH/2 for 50% overlap between samples
data = data[0::int(DEPTH/OVERLAP_FACTOR)] 

for d in data:
    #result = [database.strip(j, desired_keys) for j in data]#database.strip(d, desired_keys)
    price = database.raw_strip(d, ['lastPrice']) 
    result = database.strip(d, desired_keys) 
    training_prices.append(price[-1]) 
    samples.append(result.flatten())

#create answer set
for i in range(len(data)-1):
    d_initial = data[i]
    d_final = data[i+1] 
    avg_price_initial = np.average(database.raw_strip(d_initial, ['lastPrice']))  
    avg_price_final = np.average(database.raw_strip(d_final, ['lastPrice'])) 
    #print(avg_price_final/avg_price_initial)
    if avg_price_final/avg_price_initial >= 1+step:
        answers.append(-1)
    elif avg_price_final/avg_price_initial <= 1-step:
        answers.append(1)
    else:
        answers.append(0) 

'''create validation data'''

# database.load_validation_data() 
# data = [[database.validation_data[i+1] for i in range(begin, begin+DEPTH)] for begin in range(DATA_SIZE-DEPTH)] 

#from data extract every 25 or DEPTH/2 for 50% overlap between samples
data = data[0::int(DEPTH/OVERLAP_FACTOR)] 

validation_samples = []
validation_prices = []
validation_answers = [0] 
for d in data:
    #result = [database.strip(j, desired_keys) for j in data]#database.strip(d, desired_keys)
    price = database.raw_strip(d, ['lastPrice']) 
    result = database.strip(d, desired_keys) 
    training_prices.append(price[-1]) 
    validation_samples.append(result.flatten())

#create answer set
'''NOTE!!! switched buy/sell labels to check for bug!...And for some weird reason works??'''
for i in range(len(data)-1):
    d_initial = data[i]
    d_final = data[i+1] 
    avg_price_initial = np.average(database.raw_strip(d_initial, ['lastPrice']))  
    avg_price_final = np.average(database.raw_strip(d_final, ['lastPrice'])) 
    validation_prices.append(avg_price_final) 
    #print(avg_price_final/avg_price_initial)
    if avg_price_final/avg_price_initial >= 1+step:
        validation_answers.append(-1)
    elif avg_price_final/avg_price_initial <= 1-step:
        validation_answers.append(1)
    else:
        validation_answers.append(0) 
