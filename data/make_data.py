import _pickle as cPickle
import csv
import sys, os

def make_data(filename, avg_visits=10):
    temp_reader = csv.reader(open(filename))
    num_cols = len(next(temp_reader))
    your_list = list(temp_reader)
    del temp_reader
    data_dict = {}
    data_dict['data'] = []
    data_dict['label'] = []
    for i in range(len(your_list)//avg_visits):
        data_dict['data'].append([item[1:-1] for item in your_list[i*avg_visits:(i+1)*avg_visits]])
        l1 = [item[-1] for item in your_list[i*avg_visits:(i+1)*avg_visits]]
        data_dict['label'].append(l1[0])

    return data_dict

train_data = make_data("small_long(train).csv",10)
cPickle.dump(train_data, open( "train.p", "wb" ))
del train_data
validate_data = make_data("small_long(valid).csv",10)
cPickle.dump(validate_data, open( "valid.p", "wb" ))
del validate_data