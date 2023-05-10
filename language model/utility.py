import numpy as np
import remote as rm

f = open('phrases_small', 'r') 

text = f.read()

words = []

start = text.find(' ')
end = text.find('\n', start)   

while start != -1:
    words.append(text[start:end])
    start = text.find(' ', end) 
    end = text.find('\n', start) 


words_new = []
for w in words:
    w_new = [i for i in w if i.isalpha() or i==' ']
    
    words_new.append(''.join(w_new)[3:]) 

words = words_new

#add example sentences for each primitive from dictionary
# prim = np.load('dictionary.npy', allow_pickle=True).item()

# for p in prim:
#     defin, ex = rm.define(p) 
#     if len(ex) > 1:
#         words.append(ex) 
#     elif len(defin) > 1:
#         words.append(defin) 


f.close()

np.save('phrases_small', words, allow_pickle=True) 

words = np.load('phrases_small.npy', allow_pickle=True) 

print(words) 
print(len(words)) 






