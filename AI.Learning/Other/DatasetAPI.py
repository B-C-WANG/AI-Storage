import tensorflow as tf
import numpy as np

print(tf.__version__)
def from_array_and_iteration():

    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0,2.0,3.0,4.0,5.0]))
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()

    for i in dataset.__dir__():
        print(i)
        print(getattr(dataset,i))
    print("____________")
    print(dataset.output_shapes)
    with tf.Session() as  sess:

        for _ in range(5):
            print(sess.run(one_element))


from_array_and_iteration()


