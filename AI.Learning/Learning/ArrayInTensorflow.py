import tensorflow as tf
import numpy as np

def test_to_float():
    input = np.array([1,2,3,4])
    output = tf.to_float(input)
    print(output)
    sess = tf.Session()
    output = sess.run(output)
    print(output)
    sess.close()


def test_MC_pi(sim_number):
    input1 = tf.random_uniform(shape=(sim_number,))
    input2 = tf.random_uniform(shape=(sim_number,))
    distance = tf.sqrt(tf.square(input1) + tf.square(input2))
    in_circle = tf.to_float(distance<1.)
    pi = tf.reduce_sum(in_circle)/sim_number
    pi = tf.Session().run(pi)
    print(4*pi)





import time
#test_to_float()
n = time.time()
test_MC_pi(200000000)
n1 = time.time()
print(n1-n)