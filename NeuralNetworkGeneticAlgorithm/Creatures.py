

import pickle
import numpy as np
import copy
from keras.models import Sequential, Input
from keras.layers import Dense, Activation

import random



class Brain(object):

    @staticmethod
    def surgery_mix_brain_weights_get_creatures(offer_brain_list):
        '''
        randomly choose each float of weights from the list
        '''
        assert isinstance(offer_brain_list, list)
        assert isinstance(offer_brain_list[0], Brain)
        offer_brain_weights = [i.get_weights() for i in offer_brain_list]
        assert isinstance(offer_brain_weights[0],list)

        num = len(offer_brain_weights)
        child_weights = []


        for weight_index in range(len(offer_brain_weights[0])):
            aim_weights = np.array([offer_brain_weights[i][weight_index].flatten() for i in range(num)])

            choices = []
            for i in range(num):
                choices.append(np.zeros_like(offer_brain_weights[0][weight_index].flatten()))

            for i in range(choices[0].shape[0]):
                random_number = random.randint(0,num-1)
                choices[random_number][i] = 1

            choices = np.array(choices)
            new_weights = np.sum(choices * aim_weights,axis=0).reshape(offer_brain_weights[0][weight_index].shape)
            child_weights.append(new_weights)

        new_brain = Brain(layer_structure=offer_brain_list[0].layer_structure)
        new_brain.set_weights(child_weights)

        return Creatures(brain=  new_brain)



    def __init__(self,layer_structure):
        self.layer_structure = layer_structure
        self.__build_network()

    def __build_network(self):
        self.network = Sequential()
        self.network.add(Dense(self.layer_structure[1],input_dim=self.layer_structure[0]))
        if len(self.layer_structure) > 2:
            for neural_num in self.layer_structure[2:-1]:
                self.network.add(Dense(neural_num))
                self.network.add(Activation("tanh"))
            self.network.add(Dense(self.layer_structure[-1]))

    def get_weights(self):
        return self.network.get_weights()

    def set_weights(self,weights):
        self.network.set_weights(weights)

    def predict(self,input):
        return self.network.predict(input)

    def look_brain_info(self):
        self.network.summary()

    def weights_add_gaussian(self,std):
        new_weights = []
        for weight in self.get_weights():
            weight += np.random.normal(scale=std,size=weight.shape)
            weight[weight>1] = 1
            weight[weight<-1] = -1
            new_weights.append(weight)


        self.set_weights(new_weights)

    def _debug(self):
        self.look_brain_info()
        self.get_weights()
        print(self.predict((np.array([[0]*10]))))

    def copy(self):
        brain_copy = Brain(layer_structure=self.layer_structure)
        brain_copy.set_weights(self.get_weights())
        return brain_copy


class CreaturesGroup(object):

    def __init__(self,evolve_strategy):
        self.group = []
        self.evolve_strategy = evolve_strategy


    def add(self,creature):
        self.group.append(creature)

    def sort_by_score(self):
        to_sort_list = [(i.score,i) for i in self.group]
        sorted_list = sorted(to_sort_list, key=lambda x:-x[0])
        return [i[1] for i in sorted_list]

    def evolve(self):
        return  self.evolve_strategy.evolve(self)

    def get_mean_std_of_score(self):
        score = []
        for c in  self.group:
            score.append(c.score)
        return np.mean(score), np.std(score)

    def clear(self):
        self.group = []




class Creatures(object):

    def __init__(self,brain):

        assert isinstance(brain, Brain)
        self.brain = brain
        self.score = 0

    def score_record(self,score):
        self.score += score

    def input_observations_output_actions(self,observations):

        return self.brain.predict(observations)

    def save(self):
        with open("saved_brain.pkl", "wb") as f:
            pickle.dump(self.brain.get_weights(),f)

    def copy(self):
        c = Creatures(brain=self.brain.copy())
        c.score = 0
        return c


if __name__ == '__main__':
    a_brain = Brain(layer_structure=[10,2,5,1])
    a = Creatures(a_brain)

    b_brain = Brain(layer_structure=[10, 2, 5, 1])
    b = Creatures(b_brain)

    def test_brain_build():
        a = Brain(layer_structure=[10,2,5,1])
        print(a.get_weights())
        a._debug()
        print("before weights",a.get_weights())
        a.weights_add_gaussian(0.0005)
        print("after weights",a.get_weights())


    def test_brain_mix():



        brain_1 = Brain(layer_structure=[10,1])

        w1 = brain_1.get_weights()[0]
        w2 = brain_1.get_weights()[1]

        brain_2 = Brain(layer_structure=[10,1])
        brain_3 = Brain(layer_structure=[10,1])



        brain_1.set_weights([np.zeros_like(w1)+1,np.zeros_like(w2)+1])
        brain_2.set_weights([np.zeros_like(w1)+2,np.zeros_like(w2)+2])
        brain_3.set_weights([np.zeros_like(w1)+3,np.zeros_like(w2)+3])

        new_brain = Brain.surgery_mix_brain_weights_get_creatures([brain_2,brain_1,brain_3])
        print(new_brain.get_weights())








