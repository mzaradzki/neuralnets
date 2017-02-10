from __future__ import division, print_function
import math, os, json, sys, re
import cPickle as pickle
from glob import glob
import numpy as np
#from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain

#import pandas as pd
#import PIL
#from PIL import Image
#from numpy.random import random, permutation, randn, normal, uniform, choice
#from numpy import newaxis
#import scipy
#from scipy import misc, ndimage
#from scipy.ndimage.interpolation import zoom
#from scipy.ndimage import imread
#from sklearn.metrics import confusion_matrix
import bcolz
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.manifold import TSNE

#from IPython.lib.display import FileLink

#import theano
#from theano import shared, tensor as T
#from theano.tensor.nnet import conv2d, nnet
#from theano.tensor.signal import pool

import keras
#from keras import backend as K
from keras.utils.data_utils import get_file
#from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
#from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
#from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
#from keras.layers.core import Flatten, Dense, Dropout, Lambda
#from keras.regularizers import l2, activity_l2, l1, activity_l1
#from keras.layers.normalization import BatchNormalization
#from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.layer_utils import layer_from_config
#from keras.metrics import categorical_crossentropy, categorical_accuracy
#from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
#from keras.preprocessing.text import Tokenizer

#np.set_printoptions(precision=4, linewidth=100)


def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def onehot(x):
    return to_categorical(x)


def wrap_config(layer):
    return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}


def copy_layer(layer):
    return layer_from_config(wrap_config(layer))


def copy_layers(layers):
    return [copy_layer(layer) for layer in layers]


def copy_weights(from_layers, to_layers):
    for from_layer,to_layer in zip(from_layers, to_layers):
        to_layer.set_weights(from_layer.get_weights())


def copy_model(m):
    res = Sequential(copy_layers(m.layers))
    copy_weights(m.layers, res.layers)
    return res


def insert_layer(model, new_layer, index):
    res = Sequential()
    for i,layer in enumerate(model.layers):
        if i==index: res.add(new_layer)
        copied = layer_from_config(wrap_config(layer))
        res.add(copied)
        copied.set_weights(layer.get_weights())
    return res

'''
def adjust_dropout(weights, prev_p, new_p):
    scal = (1-prev_p)/(1-new_p)
    return [o*scal for o in weights]
'''

def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.nb_sample)])

'''
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''

def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=1)
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
    test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
    return (val_batches.classes, batches.classes, onehot(val_batches.classes), onehot(batches.classes),
        val_batches.filenames, batches.filenames, test_batches.filenames)


def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]
