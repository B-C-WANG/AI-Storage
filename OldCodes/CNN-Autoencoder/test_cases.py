from AutoEncoder import  *



def test_case_1():


    autoencoder = AutoEncoder(input_shape=(28,28,1),output_shape=(28,28,1))
    autoencoder.CNN_encoder(filters=[16,8],
                            activations=["relu"]*2,
                            kernel_size=[(3,3)]*2,
                            strides=[(2,2)]*2,
                            padding=["same"]*2,
                            max_pooling_size=(2,2),
                            max_pooling_stride=(1,1),
                            max_pooling_padding="SAME")
    autoencoder.CNN_decoder(filters=[8, 8, 16,1],
                            activations=["relu","relu","relu","sigmoid"],
                            kernel_size=[(3, 3)]*4,
                            strides=[(1, 1)]*4,
                            padding=[ "same"]*4,
                            up_sampling_size=(2,2)
                            )


    autoencoder.build_encoder_decoder_model(optimizer="adam",loss="binary_crossentropy")

    auto_encoder_model, encoder_model = autoencoder.return_model()


if __name__ == "__main__":
    test_case_1()
