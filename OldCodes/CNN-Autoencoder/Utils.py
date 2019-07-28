import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import os
import warnings


#TODO: collect all the Utils need from other projects




class Data:
    @staticmethod
    def split_data(X,y,split_train_ratio):

        if X.shape[0] != y.shape[0]:
            raise ValueError("the sample number of X must be the same as y")
        sample_number = X.shape[0]
        train_number = int(sample_number * split_train_ratio)


        print("split data to: ",X[:train_number].shape,y[:train_number].shape,
               X[train_number:].shape,y[train_number:].shape)

        return X[:train_number],y[:train_number],\
               X[train_number:],y[train_number:]


    @staticmethod
    def transpose_concatenate(a,b):

        aa = np.transpose(a)
        bb = np.transpose(b)
        cc = np.concatenate([aa, bb], axis=0)
        return np.transpose(cc)





    @staticmethod
    def standard_transform_all(data_set,dir_name,name):
        '''
        do normalization for all the data_set, and save models for every line reverse
        :param data_set:
        :return: normalized data
        '''
        IOUtils.check_and_create_dir(dir_name)
        result = []
        for i in range(data_set.shape[1]):
            a = StandardScaler()
            a.fit(data_set[:, i].reshape(-1, 1))

            joblib.dump(a, "{}/{}_{}_standard_transform.m".format(dir_name,name,i))
            b = a.transform(data_set[:, i].reshape(-1, 1))
            # print(b)
            result.append(b)
        result = np.array(result)
        result = np.transpose(result)[0, :, :]

        return result


    @staticmethod
    def standard_reverse_all(data_set,dir_name,name):
        '''
        reverse the normalization from the saved model
        :param data_set:
        :return:
        '''
        IOUtils.check_and_create_dir(dir_name)
        result = []
        for i in range(data_set.shape[1]):
            a = joblib.load("{}/{}_{}_standard_transform.m".format(dir_name,name,i))

            b = a.inverse_transform(data_set[:, i].reshape(-1, 1))

            result.append(b)
        result = np.array(result)
        result = np.transpose(result)[0, :, :]

        return result

    @staticmethod
    def standard_transform_one(data_list, dir_name, name):
        IOUtils.check_and_create_dir(dir_name)
        a = StandardScaler()
        a.fit(data_list.reshape(-1, 1))

        joblib.dump(a, "{}/standard_transform_{}.m".format(dir_name,name))
        b = a.transform(data_list.reshape(1, -1))
        b = np.transpose(b)
        print(b.shape)
        return b

    @staticmethod
    def load_and_transform_one(data_list, dir_name, name):
        IOUtils.check_and_create_dir(dir_name)
        a = joblib.load("{}/standard_transform_{}.m".format(dir_name, name))
        b = a.transform(data_list.reshape(1, -1))
        b = np.transpose(b)

        return b


    @staticmethod
    def standard_reverse_one(data_list, dir_name, name):
        IOUtils.check_and_create_dir(dir_name)
        a = joblib.load("{}/standard_transform_{}.m".format(dir_name, name))
        b = a.inverse_transform(data_list.reshape(1, -1))
        b = np.transpose(b)
        return b

    @staticmethod
    def shuffle_x_and_y(x, y):
        '''
        shuffle two data set

        :param x:
        :param y:
        :return:
        '''
        shuffle_list = list(range(x.shape[0]))
        if x.shape[0] != y.shape[0]:
            raise ValueError("the sample number of x must be the same as y")
        if len(y.shape) != 1:
            raise ValueError("the shape of y must be (None, )")
        if len(x.shape) != 2:
            raise ValueError("the shape of x must be (None, A)")
        new_x = []
        new_y = []
        np.random.shuffle(shuffle_list)
        for i in shuffle_list:
            new_x.append(x[i, :])
            new_y.append(y[i])
        new_x = np.array(new_x)
        new_y = np.array(new_y)
        return new_x, new_y


    #TODO: to be finished
    @staticmethod
    def make_cross_validation_data_set(folder_number):
        '''
        make cross validation data set for train and test

        e.g. folder_number is [60,60,30] then return a list of data which has [[len60],[len60],[len30]]
        :return:
        '''
        x_folders = []
        g_folders = []
        k_folders = []

        start = 0
        for i in folder_number:
            x_folders.append(data[start:start + i])
            g_folders.append(y1[start:start + i])
            k_folders.append(y2[start:start + i])
            start += i
        # test
        for i in x_folders:
            print(i.shape)
        for i in g_folders:
            print(i.shape)

        # folder for training
        folder_index = list(range(len(folder_number)))
        print(folder_index)
        x_train_f = []
        g_train_f = []
        k_train_f = []
        x_test_f = []
        g_test_f = []
        k_test_f = []
        for i in folder_index:
            train_folder_index = list(range(len(folder_number)))
            del train_folder_index[i]

            x_train_f_temp = []
            g_train_f_temp = []
            k_train_f_temp = []
            for j in train_folder_index:
                x_train_f_temp.append(x_folders[j])
                g_train_f_temp.append(g_folders[j])
                k_train_f_temp.append(k_folders[j])

            a = np.concatenate(x_train_f_temp, axis=0)
            b = np.concatenate(g_train_f_temp, axis=0)
            c = np.concatenate(k_train_f_temp, axis=0)

            x_train_f.append(a)
            g_train_f.append(b)
            k_train_f.append(c)

            x_test_f.append(x_folders[i])
            g_test_f.append(g_folders[i])
            k_test_f.append(k_folders[i])

        print("x_train_f:", len(x_train_f), "_", x_train_f[0].shape)
        print("g_train_f:", len(g_train_f), "_", g_train_f[0].shape)
        print("k_train_f:", len(k_train_f), "_", k_train_f[0].shape)

        print("x_test_f:", len(x_test_f), "_", x_test_f[0].shape)
        print("g_test_f:", len(g_test_f), "_", g_test_f[0].shape)
        print("k_test_f:", len(k_test_f), "_", k_test_f[0].shape)

        return x_train_f, g_train_f, k_train_f, x_test_f, g_test_f, k_test_f


class Log:

    #TODO:  make a package "Log"

    @staticmethod
    def color_print(string, color="red"):

            color_list = {"black":"030", "red":"031", "green":"032", "yellow":"033", "blue":"034", "purple":"035","cyan": "036", "white":"037"}
            if type(color_list) == int:
                __color = color_list.keys()[color]
            else:
                __color = color_list[color]
            before = "\033[1;{};40m".format(__color)
            after = "\033[0m"
            return before + string + after


    @staticmethod
    def log_align(string, length=0):

        slen = len(string)
        re = string
        placeholder = ' '
        if length > slen:
            return re + placeholder * (length - slen)
        else:
            return re


class IO:


    @staticmethod
    def get_all_dir_in_path(path):
        base_path = os.path.realpath(path)

        dirList = []

        files = os.listdir(base_path)
        for i in files:
            if os.path.isdir(os.path.join(base_path,i)):
                dirList.append(os.path.join(base_path,i))
        return dirList


    @staticmethod
    def check_and_create_dir(dir_name):
        '''
        check if a dir named <die_name> exists, otherwise create one
        :param dir_name: str
        :return: None
        '''
        if dir_name not in os.listdir():
            LogUtils.color_print("creating the dir named '{}' ... ".format(dir_name),color="blue")
            os.makedirs(dir_name)
        else:
            LogUtils.color_print("the dir '{}' exists".format(dir_name),color="blue")

class NNModel:


    @staticmethod
    def save_weights(keras_model,dir_name,model_name,save_weights=True):
        IO.check_and_create_dir(dir_name)
        model = keras_model
        if save_weights:
            model.save_weights("{}\\{}_model_weights.m".format(dir_name,model_name))

    @staticmethod
    def load_weights(keras_model,dir_name,model_name,load_weights=True):
        if load_weights:
            try:
                mode.load_weights("{}\\{}_model_weights.m".format(dir_name,model_name))
            except Exception as e:
                warnings.warn("failed to load the weights, got error: " + str(e))

    #TODO: to be finished
    @staticmethod
    def weights_visualization(keras_model):
        model = keras_model
        try:
                model.load_weights("model_weights.m")
                print("load weights successfully")
        except:
                print("failed to load weights, please train the model and save the weights")

        weights = model.get_weights()

        for i in range(len(weights)):
            print("the shape of weights {} is {}".format(i, weights[i].shape))

        from PIL import Image

        for i in range(len(weights)):
            try:
                # if the second shape is None, then go to except
                print(weights[i].shape[1])
                fig = weights[i].reshape(weights[i].shape[0],-1)
                fig = np.sum(fig,axis=1)
                fig = fig.reshape(fig.shape[0],1)
                print(fig)

            except:
                fig = weights[i].reshape(weights[i].shape[0],1)

            image = Image.fromarray(fig,'P')
            image.save("weights_{}.png".format(i))


    #TODO: to be finished
    @staticmethod
    def reconstruct_input_from_output(model, weights_name, input_matrix_shape, energy, iterations=20, step=0.01,
                                      max_loss=None):
        import keras.backend as K
        K.set_learning_phase(1)
        model = model
        model.load_weights(weights_name)

        input_ = model.input
        output_energy = model.output
        loss = np.square(energy - output_energy)

        grads = K.gradients(loss, input_)[0]

        outputs = [loss, grads]

        input_matrix = np.random.random((input_matrix_shape))
        input_matrix = input_matrix.reshape((1, -1))

        def gradient_ascent(x, iterations, step, max_loss):
            fetch_loss_and_grads = K.function([input_], outputs)

            def eval_loss_and_grads(x):
                outs = fetch_loss_and_grads([x])
                loss_value = outs[0]
                grad_values = outs[1]
                return loss_value, grad_values

            for i in range(iterations):
                loss_value, grad_values = eval_loss_and_grads(x)
                if max_loss is not None and loss_value > max_loss:
                    break
                print('..Loss value at', i, ':', loss_value)
                # there is -=, if +=, like deep dream, will add gradient
                x -= step * grad_values
            return x

        input_matrix = gradient_ascent(input_matrix,
                                       iterations=iterations,
                                       step=step,
                                       max_loss=max_loss)
        return input_matrix

    #TODO: to be finished
    @staticmethod
    def reconstruct_input_according_output(model,weights_file_name,input,output,iterations=20,step=0.01,max_loss=None):
        import keras.backend as K
        #K.set_learning_phase(1)
        K.set_learning_phase(1)
        return NN_model_build.reconstruct_input_from_output(
            model=model,
            weights_name=weights_file_name,
            input_matrix_shape=input[1:],
            energy=output,
            step=step,
            max_loss=max_loss,
            iterations=iterations
        )


    #TODO: to be finished
    @staticmethod
    def weights_visualization(keras_model, weights_name):
        model = keras_model

        try:
            model.load_weights(weights_name)
            print("load weights successfully")
        except:
            print("failed to load weights, please train the model and save the weights")

        weights = model.get_weights()

        for i in range(len(weights)):
            print("the shape of weights {} is {}".format(i, weights[i].shape))

        from PIL import Image

        for i in range(len(weights)):
            try:
                # if the second shape is None, then go to except
                # print(weights[i].shape[1])
                fig = weights[i].reshape(weights[i].shape[0], -1)
                fig = np.sum(fig, axis=1)
                fig = fig.reshape(fig.shape[0], 1)
                print(fig)

            except:
                fig = weights[i].reshape(weights[i].shape[0], 1)

            image = Image.fromarray(fig, 'P')
            image.save("weights_{}.png".format(i))







    @classmethod
    def NN_load_weights(cls,load_weights=True,model=None,file_name=None):
        if load_weights:
            try:
                model.load_weights(file_name)
                print("load weights of {} successfully".format(file_name))
            except Exception as e:
                cls.red_print(e)

    @classmethod
    def NN_save_weights(cls,save_weights=True,model=None,file_name=None):
        #cls.check_and_create_dir(file_name)
        if save_weights:
            try:
                model.save_weights(file_name)
                Utils.red_print("save weights successfully")
            except Exception as e:
                cls.red_print(e)




class Plot:
    @staticmethod
    def plot_3D_scatter(x, y, z, title="None", *args):


        ax = plt.figure().add_subplot(111, projection='3d')
        plt.title(title)
        ax.scatter(x, y, z,*args)

        plt.savefig("{}_3D_scatter.png".format(title), dpi=300)
        plt.show()

    @staticmethod
    def plot_3D_structure_all_atoms(position_info):
        '''
        plt the structure of all atoms, need position information for support
        :return:
        '''

        d = position_info
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(list(d[0, :, 0]), list(d[0, :, 1]), list(d[0, :, 2]))
        plt.show()

    @staticmethod
    def plot_many_x_y(x_list,y_list,symbol_list,fig_file_name,show=True,alpha=0.7,*args):



        for i in range(len(x_list)):
            plt.plot(x_list[i], y_list[i],symbol_list[i],alpha=alpha ,*args)
        plt.savefig(fig_file_name,dpi=300)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_many_3D_and_1_3D(array_s,array,fig_name="None"):
        # TODO: plot all atoms in red, and plot the generated result in green
        '''
        bugs here, abandon

        :param array_s:
        :param array:
        :param fig_name:
        :return:
        '''
        d = array_s
        print("array_s shape:",array_s.shape)
        print("array shape:",array.shape)
        ax = plt.figure().add_subplot(111, projection='3d')
        if len(d.shape) == 3:

            for i in range(d.shape[0]):
                for j in range(3):
                    ax.scatter(list(d[i,:, j]), list(d[i, :, j]), list(d[i, :, j]),"go")
        if array is not None:
            ax.scatter(list(array[:, 0]), list(array[:, 1]), list(array[:, 2]),"ro")
        plt.savefig(fig_name)
        plt.show()


class Error:

    @staticmethod
    def ratio_error(predict,real):
        return np.abs(np.mean(np.abs(np.abs(predict - real) / np.abs(real))))




















