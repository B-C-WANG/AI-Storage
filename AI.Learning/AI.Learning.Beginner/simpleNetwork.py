import tensorflow as tf
from numpy.random import RandomState

#a simple network, with a hidden layer

class simpleNetwork():


    def __init__(self,input_size,hidden_size):

        ##1\Modeling


        self.input_size = input_size


        self.w1 = tf.Variable(tf.random_normal(shape = [2,hidden_size],stddev=1,seed=1))

        self.w2 = tf.Variable(tf.random_normal(shape = [hidden_size,1],stddev=1,seed=1))

        self.x = tf.placeholder(dtype = tf.float32,shape = (None,2),name="x-input")

        self.y_ = tf.placeholder(dtype = tf.float32,shape =(None,1),name = "y-input")

        ##2\the Network: [input_size,2] to [input_size,hidden_size] to [input_size,1]

        self.a = tf.matmul(self.x,self.w1)

        self.y = tf.matmul(self.a,self.w2)

        ##3\Set error function and train_step

        self.cross_entropy = - tf.reduce_mean(self.y_ * tf.log(tf.clip_by_value(self.y,1e-10,1.0)))



        self.train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cross_entropy)


        rdm = RandomState()

        self.X = rdm.rand(input_size,2)# input

        self.Y = [ [float(int(x1+x2<1)) ] for (x1,x2) in self.X] #output, return a array of x1+x2 <1 shape: [input_size,1]






    def training(self,step,batch_size):





        ##4\show original data##



        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            print("Before training: ")
            print("w1:", sess.run(self.w1))
            print("w2:", sess.run(self.w2))

        ##5\training

            steps = step
            batch_size = batch_size
            for i in range(steps):
                start = (i* batch_size)% self.input_size#循环选取batch_size大小的数据

                end = min(start + batch_size,self.input_size)
                sess.run(self.train_step,feed_dict={self.x: self.X[start:end],  self.y_: self.Y[start:end]})

                if i%1000 ==0:

                    total_cross_entropy = sess.run(self.cross_entropy,feed_dict={self.x:self.X,self.y_:self.Y})

                    print("after %d training step(s), cross_entropy on all data is %g"%(i,total_cross_entropy))

            print("final w1",sess.run(self.w1))
            print("final w2",sess.run(self.w2))


simpleNetwork = simpleNetwork(input_size=200,hidden_size=3)
simpleNetwork.training(step=5000,batch_size=10)










