from main_code import run_RNN
import hyperopt
import tensorflow as tf

import functools
import numpy as np
import re
import itertools
from collections import Counter
from keras.utils import np_utils
import config_reader
import utils
from rnn_model import *

config = config_reader.read_config(utils.abs_path_of("config/default.ini"))
train_file = config.get_rel_path("PATHS", "training_file")
validation_file = config.get_rel_path("PATHS", "validation_file")

trainX = np.asarray(unpickle(train_file)['data'])
trainY = np.asarray(unpickle(train_file)['label'])
trainY = np_utils.to_categorical(trainY, 2)
ValidX = np.asarray(unpickle(validation_file)['data'])
ValidY = np.asarray(unpickle(validation_file)['label'])
ValidY = np_utils.to_categorical(ValidY, 2)


def objective(args):

    params = {}

    params['rnn_num_layers'] = args['rnn_num_layers']
    params['rnn_num_hidden'] = args['rnn_num_hidden']
    params['learning_rate'] = args['learning_rate']
    params['rnn_batch_size'] = args['rnn_batch_size']
    params['dropout_keep_probability'] = args['dropout_keep_probability']  
    params['validation_window'] = args['validation_window']

    with tf.Graph().as_default():
        loss = -1 * run_RNN(params, trainX, trainY, ValidX, ValidY)
    
    return loss


def optimize():

    config = config_reader.read_config(utils.abs_path_of("config/default.ini"))


    space = {

        'learning_rate': hyperopt.hp.choice('learning_rate', [0.0001, 0.001]),
        'rnn_num_layers': hyperopt.hp.choice('rnn_num_layers', [2,3,4]),
        'rnn_num_hidden': hyperopt.hp.choice('rnn_num_hidden', [200, 300, 400]),
        'rnn_batch_size': hyperopt.hp.choice('rnn_batch_size', [50, 100, 200]),
        'dropout_keep_probability': hyperopt.hp.choice('dropout_keep_probability',[0.5,0.4,0.6]),
        'validation_window': hyperopt.hp.choice('validation_window',[5])
        
    }

    best_model = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=200)

    print(best_model)
    print(hyperopt.space_eval(space, best_model))

    
if __name__ == '__main__':
    optimize()
