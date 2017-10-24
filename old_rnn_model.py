import functools
import tensorflow as tf
import numpy as np
import re
import itertools
from collections import Counter
#from keras.utils import np_utils
import config_reader
import utils

config = config_reader.read_config(utils.abs_path_of("config/default.ini"))
batch_size = config.getint("RNN_SEQUENCE_CLASSIFICATION", "rnn_batch_size")


def unpickle(file):
    import _pickle as cPickle
    #import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def batch_iter(data, batch_size, num_epochs, shuffle=True):

    data = list(data)
    data = np.array(data)
    data_size = len(data)
    
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

    #def __init__(self, data, target, dropout, num_hidden = config.getint("RNN_SEQUENCE_CLASSIFICATION", "rnn_num_hidden"), num_layers = config.getint("RNN_SEQUENCE_CLASSIFICATION", "rnn_num_layers")):
    def __init__(self, data, target, dropout ): 
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = config.getint("RNN_SEQUENCE_CLASSIFICATION", "rnn_num_hidden")
        self._num_layers = config.getint("RNN_SEQUENCE_CLASSIFICATION", "rnn_num_layers")
        self.prediction
        self.train_op
        self.valid_op

    @lazy_property
    def prediction(self):
        # Recurrent network.
        def single_sell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self._num_hidden), output_keep_prob=self.dropout)
        
        network = tf.contrib.rnn.MultiRNNCell([single_sell() for _ in range(self._num_layers)])
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction
    '''    
    @lazy_property
    def cost(self):
        with tf.name_scope('cross_entropy'):
            cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        with tf.name_scope('graphs'):
        	tf.summary.scalar("train_loss", cross_entropy)
        return cross_entropy
    '''
    @lazy_property
    def train_op(self):
    	learning_rate = config.getfloat("TRAINING", "learning_rate")
    	with tf.name_scope('training'):
            cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            mistakes = tf.not_equal( tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
            with tf.name_scope('graphs'):
            	tf.summary.scalar("train_loss", cross_entropy)
            	tf.summary.scalar("train_error", error)
            optimizer.minimize(cross_entropy)
    	return error
        #return optimizer.minimize(self.cost)
    '''
    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        with tf.name_scope('graphs'):
        	tf.summary.scalar("train_error", error)
        return error
    '''
    @lazy_property
    def valid_op(self):
    	with tf.name_scope("validation"):
    		cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))       	
    		mistakes = tf.not_equal( tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
    		error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
    		with tf.name_scope('graphs'):
    			tf.summary.scalar("valid_loss", cross_entropy)
    			tf.summary.scalar("valid_error", error)
    	return error

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        with tf.name_scope("weights"):
            weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        with tf.name_scope("biases"):
            bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @lazy_property
    def summary_op(self):
        return tf.summary.merge_all()

    @lazy_property
    def summary_op_test(self):
        return tf.summary.scalar("Validatiion_error", self.error)


