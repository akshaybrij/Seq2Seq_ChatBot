import tensorflow as tf
import numpy as np
import random
import sys
import nltk
from nltk.tokenize import word_tokenize
import itertools
from collections import defaultdict
import pickle
EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
FileName="data/chat.txt"
limit={
'maxq':20,
'minq':2,
'maxa':20,
'mina':2,
}
UNK='unk'
def read_lines(filename):
    return open(filename,encoding="utf8").read().split('\n')

def filter_line(line,whitelist):
    return ''.join([x for x in line if x in whitelist])

def _index(tokenized_data,vocab_size):
    freq_dist=nltk.FreqDist(itertools.chain(*tokenized_data))
    vocab_common=freq_dist.most_common(vocab_size)
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab_common]
    word2index= dict([(w,k) for k,w in enumerate(index2word)])
    return freq_dist,index2word,word2index

def filtererd_data(seq):
    filtered_q=[]
    filtered_a=[]
    for i in range(0,len(seq),2):
        try:
         qlen, alen = len(seq[i].split(' ')), len(seq[i+1].split(' '))
         if qlen >= limit['maxq'] and qlen <= limit['minq']:
            if alen >= limit['maxa'] and alen <= limit['mina']:
                filtered_q.append(seq[i])
                filtered_a.append(seq[i+1])
        except Exception as e:
         pass
    return (filtered_a,filtered_q)

def zero_pad(qtokenized,atokenized,w2idx):
    data_len=len(qtokenized)
    idx_q=np.zeros([data_len,limit['maxq']],dtype=np.int32)
    idx_a=np.zeros([data_len,limit['maxa']],dtype=np.int32)
    for i in range(data_len):
        q_indice=pad_seq(qtokenized[i],w2idx,limit['maxq'])
        a_indice=pad_seq(atokenized[i],w2idx,limit['maxa'])
        idx_q[i]=np.array(q_indice)
        idx_a[i]=np.array(a_indice)
    return idx_a,idx_q

def pad_seq(tokens,w2id,leng):
    indices=[]
    for i in tokens:
        if i in w2id:
            indices.append(w2id[i])
    return indices+[0]*(leng-len(leng))

def process_data():
    read_data=read_lines('chat.txt')
    read_data_=[w.lower() for w in read_data]
    line=[filter_line(line,EN_WHITELIST) for line in read_data_]
    q_line,a_line=filtererd_data(line)
    qtokenized=[word_tokenize(q) for q in q_line]
    atokenized=[word_tokenize(a) for a in a_line]
    freq_dist,i2w,w2i=_index(qtokenized+atokenized,6000)
    idx_a,idx_q=zero_pad(qtokenized,atokenized,w2i)
    np.save('idx_q.npy',idx_q)
    np.save('idx_a.npy',idx_a)
    metadata={
    'w2i':w2i,
    'i2w':i2w,
    'freq_dist':freq_dist
    }
    with open('metadata.pkl','wb') as f:
        pickle.dump(metadata,f)

def load_data(PATH=''):
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a


if __name__ =="__main__":
    process_data()
