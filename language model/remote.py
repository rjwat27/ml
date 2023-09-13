import numpy as np

import requests as r

dictionary_base_address = 'https://api.dictionaryapi.dev/api/v2/entries/en/'



def define(word):
    print('looking up ', word)
    try:
        result = r.get(dictionary_base_address+word).json()[0]
    except:
        print('fail') 
        return '', '' 
    #for now this only returns the top definition and example sentence 
    #find first definition with an example sentence 
    i = 0
    m = len(result['meanings'][0]['definitions'])
    while i < m and not ('example' in result['meanings'][0]['definitions'][i]):
        i += 1
    if i < m:
        return result['meanings'][0]['definitions'][i]['definition'], result['meanings'][0]['definitions'][i]['example']
    else:
        return result['meanings'][0]['definitions'][0]['definition'], ''


def get_word_details(word):
    try:
        result = r.get(dictionary_base_address+word).json()[0]
        return result
    except:
        print('fail') 
        return '', '' 

def get_word_logical(word):
        try:
            result = r.get(dictionary_base_address+word).json()[0]
            parts = [m['partOfSpeech'] for m in result['meanings']]
            return parts
        except:
            print('fail') 
            return None
















