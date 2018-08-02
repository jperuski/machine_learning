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
x, y_raw = skld.make_classification(n_samples=1000, n_features=20,
                                    n_informative=2, n_redundant=2, n_repeated=0, 
                                    n_classes=2, n_clusters_per_class=1,
                                    weights=None, flip_y=0.01, class_sep=1.0,
                                    hypercube=True, shift=0.0, scale=1.0,
                                    shuffle=True, random_state=None)
y_raw = 2 * y_raw - 1
y = y_raw.reshape((x.shape[0], 1))

x = x.astype(np.float32)

# Open a tensorflow session
sess = tf.Session()

# Define the logistic function
def tf_logistic(y_var, y_est):
    logistic_transform = 1 / tf.exp(-1.0 * y_var * y_est)
    return(logistic_transform)

def fit_logistic_gd_opt(tf_sess, x_var, y_var, l_rate, epoch):
    
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
    y_est = tf.matmul(feat, wgt) + bias
    y_est_loss = tf_logistic(resp, y_est)
    # Minimize error using binary cross entropy
    loss = tf.reduce_sum(y_est_loss)

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

l_rate = 0.01
epochs = 500
loss_vec, wgt_fit, pred = fit_logistic_gd_opt(tf_sess = sess, x_var = x, y_var = y,
                                 l_rate = l_rate, epoch = epochs)

# confirm smooth decreasing error / loss
plt.plot(np.arange(epochs), loss_vec)
plt.ylabel('Loss')
plt.xlabel('Epochs')

# Test model accuracy
pred_match = tf.equal(tf.round(pred), tf.convert_to_tensor(y, dtype=tf.float32))
accuracy = tf.reduce_mean(tf.cast(pred_match, tf.float32))
