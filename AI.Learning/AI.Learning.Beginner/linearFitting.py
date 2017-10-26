import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


##setting parameter:

line_k = 1 # the slope

line_b = 1.3 # the intercept



data_size = 1000

learning_rate = 0.001

train_steps = 5000

batch_size = 10

## setting data

X = np.array([i for i in range(data_size)]).reshape([data_size,1])





Y = X * line_k + line_b




def plot_and_show(x,y,k,b):
    plt.plot(x,y,"ro")
    plt.plot([1,2],[1*k+b,2*k+b],'g-')
    plt.show()

#plot_and_show(X,Y)


#1\Modeling

k = tf.Variable(tf.random_normal(shape=[1,1],stddev=1))

b = tf.Variable(tf.random_normal(shape=[1,1],stddev=1))

y = tf.placeholder(tf.float32,shape=[None,1])

x = tf.placeholder(tf.float32,shape=[None,1])

y_ = tf.add(tf.matmul(x,k),b)

error = tf.reduce_mean(tf.square(y - y_))#误差函数

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)



with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())




    for i in range(train_steps):

        start = (i * batch_size) % data_size

        end = min(start + batch_size, data_size)

        sess.run(train_step,feed_dict={x:X[start:end],y:Y[start:end]})

        if i %1000 ==0:
            total_error = sess.run(error,feed_dict={x:X,y:Y})
            print("after %d training step(s), error on all data is %g" % (i, total_error))
    k = sess.run(k)[0][0]
    b =sess.run(b)[0][0]
    print("final k:",k)
    print("final b:",b)



plot_and_show(X,Y,k,b)

