
import numpy as np 
import market_forecast.DataModule as dm

class Market():
    def __init__(self, max=1000, start=100):
        self.tick_data = [] 
        self.size = 0
        self.market_eval = 0 
        self.max = max 
        self.arch_history = [] 
        self.price_history = [] 
        self.data_history = [] 

        self.current_price = 1 

        self.wallet = start
        self.coins = 0
        self.assets = start 

        self.decisions = [] 

        self.start_amount = start 

        self.net_gain = 0 

        self.ticker = dm.Watcher() 

        self.test_price_log = [] 


    def arch_log(self, archetype):
        self.arch_history.append(archetype) 
    
    def data_log(self, data):
        self.data_history.append(data) 

    def price_log(self, price):
        #price = self.ticker.price('BTCUSD') 
        self.price_history.append(price) 
        self.size += 1 
        self.current_price = price 

        if self.size > self.max:
            self.price_history.pop(0) 
            #self.tick_data.pop(0) 

        return price 

    def buy(self):
        if self.wallet != 0:
            self.coins = self.wallet / self.current_price 
            self.wallet = 0 
            #self.decisions.append(1) 
        if len(self.decisions) > self.max: 
            self.decisions.pop(0) 

    def sell(self):
        if self.coins != 0:
            self.wallet = self.coins * self.current_price 
            self.coins = 0 
            #self.decisions.append(-1) 
        if len(self.decisions) > self.max: 
            self.decisions.pop(0) 

    def trigger_decide(self, prediction, action_threshold):
        #svm returns scalar here
        if prediction >= action_threshold:
            self.buy() 
            self.decisions.append(1) 
        elif prediction <= -action_threshold:
            self.sell() 
            self.decisions.append(-1) 
        else:
            self.decisions.append(0)
            if len(self.decisions) > self.max: 
                self.decisions.pop(0) 

    def decide(self, eval):
        if eval == 1:
            self.buy() 
        elif eval == -1 or eval == 2:
            self.sell() 

        self.decisions.append(eval)
        if len(self.decisions) > self.max: 
            self.decisions.pop(0) 

    def stat(self):
        #sum assets
        self.assets = self.wallet + self.coins*self.current_price 
        self.net_gain = self.assets - self.start_amount 

    def training_data_prep(self, depth=20, foresight=20, jump=.001):
        training_data = []
        for i in range(depth, len(self.price_history)-foresight):
            up = False
            down = False
            up_jump = 0
            down_jump = 0
            change = 0
            for j in range(i, i+foresight):
                x = self.price_history[i]
                y = self.price_history[j]
                if y/x >= 1+jump:
                    up = True
                    up_jump = y/x - 1
                if y/x <= 1-jump:
                    down = True
                    down_jump = 1 - y/x  
                
            if up and down:
                if up_jump==down_jump:
                    change = 0
                elif up_jump>down_jump:
                    change = 1
                else:
                    change = -1 
            elif up:
                change = 1
            elif down:
                change = -1 
            entry = (self.arch_history[i], self.data_history[i], change) 
            training_data.append(entry) 
        return training_data 

