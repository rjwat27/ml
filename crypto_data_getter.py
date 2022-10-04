#collect crypto data for network training 
import numpy as np
import datetime, time, requests, random 
from time import sleep 
import json 

base_endpoint = "https://api.binance.us" 


def GetPrices(errornum):
  try:
    Data1 = requests.get(base_endpoint+"/api/v3/ticker/24hr?symbol=BTCUSD") 
    
    Data = Data1.json() 
    return Data 
    Data = Data[0:60]  
    
    for entry in Data:
        price = entry['lastPrice'] 
        symbol = entry['symbol'] 

    #     cryptodictionary[symbol] = price 
    
    # LogHistory(cryptodictionary) 
    
    
  except:
      errornum += 1
      print(Data1.status_code)
      print("Error ocurred in fetching price data" )
      print("Error #: ", errornum ) 
      Data = "error" 

  return Data 

tick_data = {} 
for i in range(1000):
    response = GetPrices(0) 
    response['tag'] = i 
    tick = json.dumps(response) 
    
    tick_data[i] = tick 
    sleep(15) 

file = open("training_data4.txt", 'w') 
json.dump(tick_data, file) 
# for tick in [tick_data]:
#     json.dump(tick, file)  

print("Done") 