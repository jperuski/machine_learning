# -*- coding: utf-8 -*-
"""
Created on Wed May 30 22:26:13 2018
@author: jperuski
"""

# baseline imports for python 3.6 using conda install
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as skld
import tensorflow as tf

# Data TBD
x, y_raw = _, _
y = y_raw.reshape((x.shape[0], 1))

# Open a tensorflow session
sess = tf.Session()

# Define the logistic function
def tf_logistic(x_var, w_var, b_var):
    logistic_transform = 1 / tf.exp(tf.matmul(x_var, w_var) + b_var)
    return(logistic_transform)

def fit_linear_gd_opt(tf_sess, x_var, y_var, l_rate, epoch):

    # declare static local vars, placeholders, and variable to fit (wgt)
    losses = np.empty(shape=[epoch], dtype=np.float32)
    n_obs, n_feat = x_var.shape
    # Graph placeholders
    feat = tf.placeholder(tf.float32,[None, n_feat])
    resp = tf.placeholder(tf.float32,[None, 1])
    # Set graph variables - weights and biases
    wgt = tf.Variable(tf.zeros([n_feat, 1]))
    bias = tf.Variable(tf.zeros([1]))

    # Construct model
    y_est = tf_logistic(feat, wgt, bias)
    # Minimize error using binary cross entropy
    loss = tf.reduce_mean(-tf.reduce_sum(resp * tf.log(y_est), reduction_indices=1))

    # method calls
    # Initialize the weights and bias
    init = tf.global_variables_initializer()
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(l_rate).minimize(loss)

    # initialize tf varibles
    tf_sess.run(init)

    # training steps
    for ii in range(epoch):
        _ , losses[ii] = tf_sess.run([optimizer, loss],
                  feed_dict = {feat: x_var, resp: y_var})

    # prediction after final step
    y_out = tf_sess.run(y_est, feed_dict = {feat: x_var})
    return (losses, wgt.eval(session = tf_sess), y_out)

l_rate = 0.1
epochs = 5000
loss_vec, wgt_fit, pred = fit_linear_gd_opt(tf_sess = sess, x_var = x, y_var = y,
                                 l_rate = l_rate, epoch = epochs)

# confirm smooth decreasing error / loss
plt.plot(np.arange(epochs), loss_vec)
plt.ylabel('Loss')
plt.xlabel('Epochs')

# Test model accuracy
pred_match = tf.equal(tf.round(pred), tf.convert_to_tensor(y, dtype=tf.float32))
accuracy = tf.reduce_mean(tf.cast(pred_match, tf.float32))
