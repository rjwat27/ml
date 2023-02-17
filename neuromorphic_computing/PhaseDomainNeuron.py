'''
Author: Ryan Watson

Definition for the phase domain neuron
simulation code for real-time simulations

'''

import numpy as np
from matplotlib import pyplot as plt 
import threading
from threading import Thread 
import keyboard

import scipy.io
mat = scipy.io.loadmat('vco_tc_fixed.mat')



spike_width = 50e-9 #seconds
#vref = 150 #mV
v_change = 100 #mV
base_cap = 50 #picoFarad 
vco_curve = mat['vco_tc_aligned'] 



class PDN():

    def __init__(self, vref=150, bit_width=4, key = 0):

        self.key = key

        self.stream_max = int(20e3)

        self.feedback_value = 0
        self.feedback_stream = [0 for i in range(self.stream_max)] 

        self.backsignal_value = 0
        self.backsignal_stream = [0 for i in range(self.stream_max)]

        self.learning_rate = 1e6


        '''live testing params'''

        self.vref = vref
        self.vchange = 0
        self.bits = bit_width

        self.vco_curve = vco_curve 
        self.local_freq = self.freq_from_v(vref) 
        #print('a;sldkjf: ', self.local_freq)
        self.period = 1/self.local_freq/2   #180deg phase in seconds 
        self.local_vco_stream = [0 for i in range(self.stream_max)] 

        self.vco_bias = 0
        self.vco_bias_stream = [0 for i in range(self.stream_max)]
        self.vco_bias_thread = None 

        self.bias_vco = 0
        self.bias_vco_stream = [0 for i in range(self.stream_max)]
        self.freq_stream = [0 for i in range(self.stream_max)] 
        self.bias_vco_thread = None 

        self.input_value = 0 
        self.INPUT_STREAM = [0 for i in range(self.stream_max)]
        self.input_thread = None 

        self.output_value = 0
        self.output_stream = [0 for i in range(self.stream_max)] 
        self.output_thread = None 

        self.dac_values = []
        self.capacitor_values = []
        self.total_cap = 0

        self.phase = 0
        self.phase_stream = [0 for i in range(self.stream_max)] 
        self.phase_thread = None 

        self.beta = 1e-6
        self.leak = 0 
        self.spike_flag = False 
        self.input_spike_flag = False 
        self.spiking_duration = 0 

        self.time_step = 0 #nanoseconds... for testing purposes 
        self.time_res = 1e-9    #seconds

        self.threads = []


        self.generate_capacitor_and_dac_values() 

    '''called externally'''
    '''TODO add cap divider vup vdown'''
    def push_input(self, val):
        self.input_value += val 
        
    def push_leak(self):
        self.leak = (self.phase/self.period/2) * self.beta 

    def push_feedback(self):
        self.feedback_stream.append(self.feedback_value)
        self.feedback_stream.pop(0) 
        self.feedback_value = 0 

    def push_backsignal(self):
        self.backsignal_stream.append(self.backsignal_value)
        self.backsignal_stream.pop(0) 

    '''snag best frequency for vco bias v. No interpolation yet'''
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

    def generate_capacitor_and_dac_values(self):
        for i in range(self.bits):
            self.capacitor_values.append(.5**(i-1)) 
            self.dac_values.append(1)

        self.total_cap = np.dot(self.dac_values, self.capacitor_values) + base_cap #i have...questions

    '''called externally'''
    def change_dac_values(self, dac_values):    #pass string of dac values as [MSB...LSB]
        self.dac_values = dac_values 

    def update_vref(self, bias):
        self.vref = bias
        self.local_freq = self.freq_from_v(self.vref) 

    def get_dac_value(self):
        '''this is definitely not the use case for vchange that taylor had in mind'''
        self.vchange += self.INPUT_STREAM[-1] 
        self.vchange = max(-15, self.vchange)
        self.vchange = min(15, self.vchange) 
        #print(self.vchange) 
        self.total_cap = np.sum(self.capacitor_values) + base_cap 
        return self.INPUT_STREAM[-1]#*self.total_cap#(np.dot(self.dac_values, self.capacitor_values) / self.total_cap * self.INPUT_STREAM[-1])#self.vchange) 


    '''consider creating separate class model for phase-frequency detector->get specs, '''
    '''for now measure time distance between zeros/common values'''
    def get_phase2(self):
        common_value = self.local_vco_stream[0] 

        i = 0
        for v in self.vco_bias_stream:
            if v/(common_value+.0001) > .99:
                break 
            i += 1 

        self.phase = i*self.time_res

        return self.phase 
    
    def get_phase(self):
        #corr = np.convolve(self.bias_vco_stream, self.local_vco_stream)[0:50] 
        tmp = 2*self.period*1/self.freq_stream[-1] 
        s = (tmp) / self.time_res
        space = int(s)
        length = (2*self.period/self.time_res) * (1/self.freq_stream[-1])/self.time_res
        idx = int(length) 
        #sum = np.sum(np.multiply(np.array(self.bias_vco_stream[-10000:-1]), np.array(self.local_vco_stream[-10000:-1]))) / 10000
        #print(sum)
        #temp = 1/(2*np.pi*self.local_freq) 
        self.phase = self.freq_stream[-1]/self.local_freq#(sum) 

        #print(self.phase) 
        return self.phase 

    def spike(self):
        if self.spike_flag:
            self.output_value = 1   #units of idk 
            self.spiking_duration += 1
        else:
            self.output_value = 0 

        # tmp = self.period*1/self.freq_stream[-1] 
        #print(np.average(np.square(self.phase_stream[-1000:-1])))
        # if abs(np.average(np.square(self.phase_stream[-100:-1]))) >= .25:
        #     self.spike_flag = True 

        if self.phase >= 1:
            self.spike_flag = True 

        #print(self.spiking_duration, self.spiking_duration*self.time_res) 
        if self.spiking_duration*self.time_res >= spike_width:
            #print('got here')
            self.spiking_duration = 0
            self.drain() 
            self.spike_flag = False 

    def drain(self):
        self.vco_bias = 0   #all we got for now 
        self.vchange = 0 
    
    '''state machine tick functions'''

    def tick_input(self):
        '''for now, leak is constant, instead of regularly pulsed'''
        #self.push_input() 
        self.push_leak() 
        new = self.input_value# + self.leak 
        #print(self.input_value) 
        self.INPUT_STREAM.append(new)
        self.INPUT_STREAM.pop(0) 
        self.input_value = 0 

    def tick_dac(self):
        self.vco_bias += self.get_dac_value() 
        #print(self.vco_bias) 
        self.vco_bias_stream.append(self.vco_bias) 
        self.vco_bias_stream.pop(0) 

    def tick_bias_vco(self):
        '''average derivative update method'''
        old_freq = self.freq_stream[-1] 
        freq = self.freq_from_v(self.vco_bias) 
        self.freq_stream.append(freq)
        self.freq_stream.pop(0)  
        #phase_align = (np.arccos(self.bias_vco_stream[-1])*(freq/(2*np.pi) * self.time_step*self.time_res)) #% (2*np.pi) 
        #phase_align = self.time_step*self.time_res*(freq - old_freq)  
        #period = 1/(freq + 1) 
        #phase = np.arcsin(self.bias_vco_stream[-1]) 
        #phase = (self.time_step*self.time_res) / period 
        #phase_align = phase - self.freq_stream[-1]*(self.time_step)*self.time_res
        #delta = -(freq*np.cos(phase))*self.time_res# + freq*np.sin(freq*(self.time_step*self.time_res))*self.time_res

        time = (old_freq / freq) * self.time_step 
        
        new = np.sin(2*np.pi*freq*self.time_step*self.time_res) 
        #new = np.cos(phase)
        #new = (self.bias_vco_stream[-1] + delta) 
        # new = 1 if new > 1 else new
        # new=-1 if new < -1 else new 
        self.bias_vco_stream.append(new)
        self.bias_vco_stream.pop(0) 

    def tick_local_vco(self):
        new = np.sin(2*np.pi*self.local_freq*self.time_step*self.time_res) 
        self.local_vco_stream.append(new) 
        self.local_vco_stream.pop(0) 

    def tick_phase_comparator(self):
        phase = self.get_phase() 
        self.phase_stream.append(phase)
        self.phase_stream.pop(0) 

    def tick_output_spike(self):
        self.spike() 
        self.output_stream.append(self.output_value)
        self.output_stream.pop(0) 

    def tick_feedback(self):
        self.push_feedback()  
        self.adjust()  #TODO uncomment me i am commented for testing
        self.push_backsignal() 

    def tick(self):
        self.tick_input()
        self.tick_dac()
        self.tick_bias_vco()
        self.tick_local_vco()
        self.tick_phase_comparator()
        self.tick_output_spike()
        self.tick_feedback() 
        #self.plot_tick() 
        self.time_step += 1

    def tick_loop(self):
        while True:
            self.tick() 

            

    '''testbench plotting tools'''
    def handle_close(self):
        return 0

    def set_global_plots(self):
        '''decimate for faster plotting'''

        self.fig, self.axes = plt.subplots(6) 

        self.ax_input = self.axes[0]
        self.ax_vbias = self.axes[1] 
        self.ax_bias_vco = self.axes[2]
        self.ax_local_vco = self.axes[3] 
        self.ax_phase = self.axes[4]
        self.ax_output = self.axes[5] 
     
        self.fig.canvas.mpl_connect('close_event', self.handle_close())
      
        #plt.autoscale(enable=True)  
        plt.ion()

        x = range(self.stream_max)  #nothin fancy

        y0 = self.INPUT_STREAM
        y0[0] = 2
     

        y1 = self.vco_bias_stream 
        y1[0] = 400 #?

        y2 = self.bias_vco_stream   #yes i know, egregious naming that i will not fix 
        y2[0] = 1.5
        y2[1] = -1.5

        y3 = self.local_vco_stream
        y3[0] = 1.5
        y3[1] = -1.5

        y4 = self.phase_stream
        y4[0] = 1 #?
        y4[1] = -1

        y5 = self.output_stream
        y5[0] = 2



        self.line0, = self.ax_input.plot(x, y0)
        self.line1, = self.ax_vbias.plot(x, y1)
        self.line2, = self.ax_bias_vco.plot(x, y2)
        self.line3, = self.ax_local_vco.plot(x, y3)
        self.line4, = self.ax_phase.plot(x, y4)
        self.line5, = self.ax_output.plot(x, y5)

        self.plot_selection = [0, 1, 2, 3, 4, 5] 

        self.plot_functions = [self.plot_input, self.plot_vbias, self.plot_vco_bias, self.plot_local_vco, self.plot_phase, self.plot_output] 
    
        plt.pause(.1)

    def plot_input(self):
        y0 = self.INPUT_STREAM
        self.line0.set_ydata(y0) 

    def plot_vbias(self):
        y1 = self.vco_bias_stream 
        self.line1.set_ydata(y1) 

    def plot_vco_bias(self):
        y2 = self.bias_vco_stream
        self.line2.set_ydata(y2)

    def plot_local_vco(self):
        y3 = self.local_vco_stream
        self.line3.set_ydata(y3)

    def plot_phase(self):
        y4 = self.phase_stream
        self.line4.set_ydata(y4) 

    def plot_output(self):
        y5 = self.output_stream
        self.line5.set_ydata(y5) 

    def rotate_plots(self):
        print('enter string numbers 1-6 to choose graphs to plot i.e. 1246') 
        response = input()
        self.plot_selection = []
        for i in range(len(response)):
            num = int(response[i])
            self.plot_selection.append(num) 


    def plot_tick(self):
        for p in self.plot_selection:
            self.plot_functions[p]()
            
        plt.autoscale(enable=True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

    def user_interact(self):
        while True:
            input()
            self.input_value = 1 

    def start_threads(self):
        
        # t = threading.Thread(target=self.plot_input)
        # self.threads.append(t)
      
        # t = threading.Thread(target=self.tick_loop)
        # self.threads.append(t)
    

        # t.start()
      
        # t = threading.Thread(target=self.test_regular_pulse_input) 
        # self.threads.append(t) 

        # for t in self.threads:
        #     t.start() 

        pass

    def test_regular_pulse_input(self):
        while True:
            for i in range(100):
                self.input_value = 1
            for i in range(100):
                self.input_value = 0 

    '''functional tools'''

    def forward(self, input_val):
        self.push_input(input_val) 
        return self.output_stream[-1] 

    def backward(self, feedback):
        self.feedback_value += feedback

        return self.backsignal_stream[-1] 

    def output(self):
        return self.output_stream[-1] 

    def backpropagate(self):
        return self.backsignal_value
        return self.backsignal_stream[-1] 




    def adjust(self):
        corr = np.correlate(self.feedback_stream[-50:-1], self.output_stream[-50:-1]) * self.time_res 
        #print(np.average(corr))
        change = -self.learning_rate*np.sum(corr)
        #print(change) 
        self.vref = min(0, self.vref + change) 
        self.vref = max(340, self.vref) 

        self.local_freq = self.freq_from_v(self.vref) 

        self.backsignal_value = self.feedback_stream[-1]  
        