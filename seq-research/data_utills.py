import tensorflow as tf
import numpy as np
from random import sample
def split_data_set(x,y,ratio=[0.7,0.3]):
    data_len=len(x)
    lens=[int(data_len*rati) for rati in ratio]
    trainx,trainy=x[:lens[0]],y[:lens[0]]
    testx,testy=x[lens[0]:],y[lens[0]:]
    return (trainx,trainy), (testx,testy)

def batch_gen(x, y, batch_size):
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), 7)
        yield x[sample_idx].T, y[sample_idx].T

def decode(sequence, lookup, separator=''):
    return separator.join([ lookup[element] for element in sequence if element ])
