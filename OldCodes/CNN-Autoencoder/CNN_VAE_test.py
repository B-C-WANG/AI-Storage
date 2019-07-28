import sys, os
os.environ['KERAS_BACKEND'] = 'theano'

from theano import config
config.floatX = 'float32'
config.optimizer = 'fast_compile'

from collections import OrderedDict
from keras.layers import InputLayer, BatchNormalization, Dense, Convolution2D, Deconvolution2D, Activation, Flatten, Reshape
import numpy as np
import pymc3 as pm
from pymc3.variational import advi_minibatch
from theano import shared, config, function, clone, pp
import theano.tensor as tt
import keras
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.datasets import fetch_mldata



from keras.models import Sequential

class CNN_VAE:


    @staticmethod
    def get_params(model):
        """Get parameters and updates from Keras model
        """
        shared_in_updates = list()
        params = list()
        updates = dict()

        for l in model.layers:
            attrs = dir(l)
            # Updates
            if 'updates' in attrs:
                updates.update(l.updates)
                shared_in_updates += [e[0] for e in l.updates]

            # Shared variables
            for attr_str in attrs:
                attr = getattr(l, attr_str)
                if type(attr) is tt.sharedvar.TensorSharedVariable:
                    if attr is not model.get_input_at(0):
                        params.append(attr)

        return list(set(params) - set(shared_in_updates)), updates




def test():
    mnist = fetch_mldata("MNIST original")
    print(mnist.keys())
    # theano channel first
    data = mnist['data'].reshape(-1,1, 28, 28).astype("float32")
    data /= np.max(data)
