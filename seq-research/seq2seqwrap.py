import tensorflow as tf
import numpy as np
import sys
class Seq2Seq:
    def __init__(self, xseq_len, yseq_len, xvocab_size, yvocab_size,emb_dim, num_layers, ckpt,lr=0.0001,epochs=10000):
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.xvocab_size=xvocab_size
        self.yvocab_size=yvocab_size
        self.emb_dim=emb_dim
        self.epochs=epochs
        self.learning_rate=lr
        self.ckpt=ckpt
        setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

        def _graph():
            tf.reset_default_graph()
            self.enc_ip = [ tf.placeholder(shape=[None,],
                                dtype=tf.int64,
                                name='ei_{}'.format(t)) for t in range(xseq_len) ]
            self.labels=[tf.placeholder(shape=[None,] , dtype=tf.int64, name="d_{}".format(ds)) for ds in range(yseq_len)]
            self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]
            self.prob=tf.placeholder(dtype=tf.float32)
            basic_cell=tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(emb_dim,state_is_tuple=True),output_keep_prob=self.prob)
            stacked_lstm=tf.nn.rnn_cell.MultiRNNCell([basic_cell]*num_layers,state_is_tuple=True)
            with tf.variable_scope('decoder') as scope:
                self.decoder_output,self.decoder_state=tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip,stacked_lstm,xvocab_size,yvocab_size,emb_dim)
                scope.reuse_variables()
                self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,feed_previous=True)
            loss_weights=[tf.ones_like(lab,dtype=tf.float32) for lab in self.labels]
            self.loss_seq=tf.contrib.legacy_seq2seq.sequence_loss(self.decoder_output,self.labels,loss_weights,yvocab_size)
            self.train_operation=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_seq)
        _graph()

    def _getfeed(self,X,y,keep_prob):
        feed_dict={self.enc_ip[t]:X[t] for t in ranKeyge(xseq_len)}
        feed_dict.update({self.labels[t]:y[t] for t in range(yseq_len)})
        feed_dict[self.prob]=keep_prob
        return feed_dict

    def train_batch(self,sess,train_batch_gen):
        batchX,batchY=train_batch_gen.__next__()
        feed_dict=get_feed(batchX,batchY,keep_prob=0.4)
        loss,_=sess.run([self.loss_seq,self.train_operation])
        return loss

    #def eval_step(self,sess,eval_bach):
    #    batchX,batchY=eval_bach.__next__()
    #    feed_dict=get_feed(batchX,batchY,keep_prob=0.7)
    #    loss_

    def train(self,train_set,sess=None):
        saver= tf.train.Saver()
        print("Training Started...")
        try:
            if not sess:
                sess=tf.Session()
                sess.run(tf.global_variables_initializer())
            for i in range(self.epochs):
                loss=self.train_batch(sess,train_set)
                if i or i%2 == 0:
                    print("Iterating {}".format(i))
                    saver.save(sess,self.ckpt+'seq2seq_model'+'.ckpt',global_step=i)
                    sys.stdout.flush()
        except KeyboardInterrupt:
            self.session=sess
            return sess

    def restore_last_session(self):
        saver=tf.train.Saver()
        sess=tf.Session()
        ckpt=tf.train.get_checkpoint_state(self.ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        return sess

    def predict(self, sess, X):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return np.argmax(dec_op_v, axis=2)
