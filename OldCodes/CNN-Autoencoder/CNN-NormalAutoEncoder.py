from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import warnings
from Utils import *

class CNN_NormalAutoEncoder:


    def __init__(self,
                 input_shape,
                 output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input = Input(shape=self.input_shape)




    def load_configurations(self):
        '''

        TODO: the configuration of the auto-encoder can be load from files

        :return:
        '''
        pass

    def CNN_encoder(self,
                    filters,
                    activations,
                    kernel_size,
                    strides,
                    padding,
                    max_pooling_size,
                    max_pooling_stride,
                    max_pooling_padding):
        '''
        the parameter was set from list
        like filters = [64,32,16]
        activations = ["relu","relu"] or ["relu"] * 6
        '''
        x = self.input

        for i in range(len(filters)):
                x = Conv2D(filters=filters[i],
                           kernel_size=kernel_size[i],
                           strides=strides[i],
                           activation=activations[i],
                           padding=padding[i]
                           )(x)
                x = MaxPooling2D(pool_size=max_pooling_size,
                              strides=max_pooling_stride,
                              padding=max_pooling_padding
                              )(x)

        self.encoder = x
        self.encoder_shape = self.encoder.shape()




    def CNN_decoder(self,
                    filters,
                    activations,
                    kernel_size,
                    strides,
                    padding,
                    up_sampling_size,
                    ):
        '''
        the parameter was set from list
        like filters = [64,32,16]
        activations = ["relu","relu"] or ["relu"] * 6
        '''

        x = Conv2D(filters=filters[0],
                       kernel_size=kernel_size[0],
                       strides=strides[0],
                       activation=activations[0],
                       padding=padding[0]
                       )(self.encoder)
        if len(filters) >=3:
                for i in range(1,len(filters)-1):
                    x = Conv2D(filters=filters[i],
                               kernel_size=kernel_size[i],
                               strides=strides[i],
                               activation=activations[i],
                               padding=padding[i]
                               )(x)
                    x = UpSampling2D(size=up_sampling_size)(x)

        self.decoder = Conv2D(filters=filters[-1],
                       kernel_size=kernel_size[-1],
                       strides=strides[-1],
                       activation=activations[-1],
                       padding=padding[-1])(x)

        self.decoder_shape = self.decoder.shape()


    def extend_encoder_with_dense(self):
        '''
        add dense layer after the encoder

        :return:
        '''
        pass



    def build_encoder_decoder_model(self,optimizer,loss,**kwargs):

        try:
            _ = self.encoder
        except:
            raise ValueError("the model encoder is not exist, set the model first")

        try:
            _ = self.decoder
        except:
            raise ValueError("the model decoder is not exist, set the model first")



        self.auto_encoder_model = Model(inputs=self.input,outputs=self.decoder)
        self.auto_encoder_model.compile(optimizer=optimizer,loss=loss,*kwargs)

        self.encoder_model = Model(inputs=self.auto_encoder_model.input,
                              outputs=self.encoder)







    def return_model(self):
        '''
        return the total model

        :return:
        '''
        try:
            _ = self.auto_encoder_model
            _ = self.encoder_model
        except:
            raise ValueError("the auto-encoder and encoder is not exist, build the model first")

        Log.color_print("the auto_encoder model is")
        self.auto_encoder_model.summary()

        return self.auto_encoder_model,self.encoder_model



