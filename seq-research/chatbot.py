import tensorflow as tf
import numpy as np
import data_utills
import seq2seqwrap
import data

metadata,idx_q,idx_a=data.load_data('ds/')
(trainX,trainY),(testX,testY)=data_utils.split_data_set(idx_q,idx_a)
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024
model=seq2seqwrap.Seq2Seq(xseq_len=xseq_len, yseq_len=yseq_len,
                                       xvocab_size=xvocab_size,
                                       yvocab_size=yvocab_size,
                                       ckpt_path='ckpt/',
                                       emb_dim=emb_dim,
                                       num_layers=3)

train_batch_gen=data_utils.rand_batch_gen(trainX,trainy,batch_size)
sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)
