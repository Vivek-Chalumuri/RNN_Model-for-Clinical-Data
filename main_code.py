import functools
import tensorflow as tf
import numpy as np
import re
import itertools
from collections import Counter
from keras.utils import np_utils
import config_reader
import utils
from rnn_model import *

def run_RNN(params, trainX, trainY, ValidX, ValidY):

    config = config_reader.read_config(utils.abs_path_of("config/default.ini"))
    training_iters = config.getint("RNN_SEQUENCE_CLASSIFICATION", "rnn_training_iters")
    validation_interval = config.getint("PROCESS", "validation_interval")
    #train_batchsize = config.getint("RNN_SEQUENCE_CLASSIFICATION", "rnn_batch_size")
    #keep_prob = config.getfloat("TRAINING", "dropout_keep_probability", fallback=1.0)
    train_batchsize = params['rnn_batch_size']
    keep_prob = params['dropout_keep_probability']
    validation_window = params['validation_window']

    train_len, rows, row_size = trainX.shape
    batches = batch_iter( zip(trainX, trainY), train_batchsize, training_iters)
    
    _, rows, row_size = trainX.shape
    num_classes = trainY.shape[1]

    with tf.name_scope('input'):
        data = tf.placeholder(tf.float32, [None, rows, row_size])
        target = tf.placeholder(tf.float32, [None, num_classes])
        dropout = tf.placeholder(tf.float32)

    model = SequenceClassification(data, target, dropout, params)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i,batch in enumerate(batches):
        batch_x, batch_y = zip(*batch)
      
        _, train_accuracy = sess.run([model.optimize, model.accuracy], {data: batch_x, target: batch_y, dropout: keep_prob})

        print('Batch {:2d} Train_accuracy {:3.1f}%'.format(i + 1, 100 * train_accuracy))
        val_loss = []
        v_count = 0
        if (i+1) % validation_interval == 0:
            accuracy, summary_test = sess.run([model.accuracy, model.summary_op_test], {data: ValidX, target: ValidY, dropout: 1})
            print("********************************************")
            print('Validation_accuracy {:3.1f}%'.format(100 * accuracy))
            print("********************************************")
            loss = -1*accuracy
            val_loss.append(loss)
            v_count += 1
            if v_count > validation_window:
                Validation_Loss = np.mean(val_loss[-validation_window:])
            else:
                Validation_Loss = np.mean(val_loss)
    return Validation_Loss
#if __name__ == '__main__':
    #main()