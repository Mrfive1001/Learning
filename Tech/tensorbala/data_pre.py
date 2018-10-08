import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


def input_fn():
    data = np.load('all_samples.npy')
    data = tf.data.Dataset.from_tensor_slices({'X':data[:,:2],'Y':data[:,2:]})
    data = data.repeat(10)
    data = data.shuffle(200)
    data = data.batch(10)
    iters = data.make_one_shot_iterator()
    return data

estimator = tf.estimator.DNNRegressor(hidden_units=[10,10,10],feature_columns=[0,1])
estimator.train(input_fn=input_fn, steps=3000)

for iteration in iters:
    # print(iteration)
    pass