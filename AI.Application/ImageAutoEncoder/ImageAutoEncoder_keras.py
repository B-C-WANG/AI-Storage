
import numpy as np
from keras.models import  Model
from keras.layers import Input,Conv2D,\
    Activation,UpSampling2D,MaxPool2D
from keras.callbacks import  TensorBoard
import matplotlib.pyplot as plt


class AutoConv2DEncoder:
    '''
    Use tensorflow as backend
    '''

    def __init__(self,image_width,image_height,image_channel):
        self._input_shape = (image_width,image_height,image_channel)
        self.build_model()

    def build_model(self):

        image_input = Input(shape=self._input_shape)

        conv1 = Conv2D(filters=16,
                       kernel_size=(3,3),
                       padding="SAME",
                       activation="relu")(image_input)

        conv1 = MaxPool2D(pool_size=(2,2),
                          strides=(2,2),
                          padding="SAME")(conv1)

        conv2 = Conv2D(filters=8,
                       kernel_size=(3, 3),
                       padding="SAME",
                       activation="relu")(conv1)

        conv2 = MaxPool2D(pool_size=(2, 2),
                          strides=(2, 2),
                          padding="SAME")(conv2)

        conv3 = Conv2D(filters=8,
                       kernel_size=(3, 3),
                       padding="SAME",
                       activation="relu")(conv2)

        self.encoded = MaxPool2D(pool_size=(2, 2),
                          strides=(2, 2),
                          padding="SAME")(conv3)





        deconv1 = Conv2D(filters=8,
                       kernel_size=(3, 3),
                       padding="SAME",
                       activation="relu")(self.encoded)

        deconv1 = UpSampling2D(size=(2, 2))(deconv1)

        deconv2 = Conv2D(filters=8,
                       kernel_size=(3, 3),
                       padding="SAME",
                       activation="relu")(deconv1)

        deconv2 = UpSampling2D(size=(2, 2))(deconv2)

        # no padding='SAME' here

        deconv3 = Conv2D(filters=16,
                       kernel_size=(3, 3),
                       activation="relu")(deconv2)

        deconv3 = UpSampling2D(size=(2, 2))(deconv3)

        self.decoded = Conv2D(filters=1,
                         kernel_size=(3,3),
                         padding="SAME",
                         activation="sigmoid")(deconv3)

        self.encode = Model(image_input,self.encoded)

        self.auto_encoder = Model(image_input, self.decoded)

        self.auto_encoder.compile(optimizer="adadelta",
                                  loss="binary_crossentropy")

    def fit(self,train_input_image,train_output_image,valid_input_image,
            valid_output_image,save_weights=True,
            epochs=5,
            batch_size=32,
            shuffle=True,
            log_dir = "/tem/autoconv2dencoder"
            ):


        self.auto_encoder.fit(x=train_input_image,y=train_output_image,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              validation_data=(valid_input_image, valid_output_image),
                              callbacks=[TensorBoard(log_dir=log_dir)])

    def autoencoder_predict(self,images):
        return self.auto_encoder.predict(images)

    def save_weights(self,filename='autoconv2dencoder_wiehgts.h5'):
        self.auto_encoder.save_weights('autoconv2dencoder_wiehgts.h5')
        print("save weights success")

    def load_weights(self,filename='autoconv2dencoder_wiehgts.h5'):
        try:
            self.auto_encoder.load_weights('autoconv2dencoder_wiehgts.h5',by_name=True)
            print("load wights success")
        except Exception as a:
            print('\033[1;31;40m str(Exception):\t', str(a), "'\033[0m'")

    def show_original_and_new(self,test_data):
        decoded_imgs = self.autoencoder_predict(test_data)

        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(1,n):

            ax = plt.subplot(2, n, i)
            plt.imshow(test_data[i].reshape(self._input_shape[:2]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


            ax = plt.subplot(2, n, i + n)
            plt.imshow(decoded_imgs[i].reshape(self._input_shape[:2]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig("original_and_new_data.png",dpi=300)
        plt.show()

    def show_encode_result(self,test_data):
        encoded_imgs =self.encode.predict(test_data)

        input_shape = encoded_imgs.shape

        # encoded_imgs have the shape (None0, width', height', 8)

        # only first img is needed
        encoded_img = encoded_imgs[0,:,:,:]

        encoded_img.reshape((input_shape[1],input_shape[2],input_shape[3]))

        shape = encoded_img.shape

        for i in range(shape[2]):
            ax = plt.subplot(1, shape[2], i+1)
            plt.imshow(encoded_img[:,:,i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig("encoded_result.png",dpi=300)
        plt.show()







# mnist_demo

from keras.datasets import mnist

(x_train, _),(x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.reshape(x_train,(len(x_train),28,28,1))
x_test = np.reshape(x_test,(len(x_test),28,28,1))



mnist_autoencoder = AutoConv2DEncoder(28,28,1)

def train():
    mnist_autoencoder.fit(train_input_image=x_train,
                          train_output_image=x_train,
                          epochs=5,
                          batch_size=128,
                          valid_input_image=x_test,
                          valid_output_image=x_test)
    mnist_autoencoder.save_weights()
def load():
    mnist_autoencoder.load_weights()
    mnist_autoencoder.show_original_and_new(x_test)
def show_encoded_result():
    mnist_autoencoder.load_weights()
    mnist_autoencoder.show_encode_result(x_test)


#train()
#load()
#show_encoded_result()