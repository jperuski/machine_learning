# -*- coding: utf-8 -*-
"""
author: jperuski - 2018
notes: see github for license details 
"""

# baseline imports for python 3.6 using conda install
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as skld
from sklearn import linear_model as lm
import tensorflow as tf

# diabetes data 
# see: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
x, y_raw = skld.load_diabetes(return_X_y = True)
y = y_raw.reshape((x.shape[0], 1))

sess = tf.Session()

# function to solve least squares fit using matrix formulation
# note: exploring numpy array to tensor conversion (and vice versa) and linear algebra 
# B = (X^T * X)^-1 * X^T * Y
# args: tf_sess - an intialized tensorflow session
#       x_var - the design matrix
#       y_var - the response
def linalg_reg(tf_sess, x_var, y_var):
    x_tf = tf.convert_to_tensor(x_var, dtype=tf.float32)
    x_t_tf = tf.matrix_transpose(x_tf)
    y_tf = tf.convert_to_tensor(y_var, dtype=tf.float32)
    prod1 = tf.matmul(x_t_tf, x_tf)
    inv_prod1 = tf.matrix_inverse(prod1)
    prod2 = tf.matmul(inv_prod1, x_t_tf)
    prod3 = tf.matmul(prod2, y_tf)

    return prod3.eval(session=tf_sess)

# tensorflow linear algebra solution
linalg_wgt = linalg_reg(tf_sess = sess, x_var = x, y_var = y)

# tensorflow gradient descent approach to fitting least squares
# loss function: sum of squared error (sse)
# note: totally unnecessary but useful for exploring the basic tensor \n
#       structure, placeholders, variables, and training procedure
# args: tf_sess - an intialized tensorflow session
#       x_var - the design matrix
#       y_var - the response
#       l_rate - the learning rate
#       epoch - number of epochs in the training procedure
def fit_linear_gd_opt(tf_sess, x_var, y_var, l_rate, epoch):
   
    # declare static local vars, placeholders, and variable to fit (wgt)
    errs = np.empty(shape=[epoch], dtype=np.float32)
    n_obs, n_feat = x.shape
    feat = tf.placeholder(tf.float32,[None, n_feat])
    resp = tf.placeholder(tf.float32,[None, 1])
    wgt = tf.Variable(tf.ones([n_feat, 1]))
    
    # method calls
    init = tf.global_variables_initializer()    
    y_est = tf.matmul(feat, wgt)
    sse = tf.reduce_sum(tf.square(resp - y_est))
    wgt_optimizer = tf.train.GradientDescentOptimizer(l_rate).minimize(sse)

    # initialize tf varibles
    tf_sess.run(init)

    # training steps
    for ii in range(epoch):
        tf_sess.run(wgt_optimizer, feed_dict = {feat: x_var, resp: y_var})
        errs[ii] = tf_sess.run(sse, feed_dict = {feat: x_var, resp: y_var})

    # prediction
    y_out = tf_sess.run(y_est, feed_dict = {feat: x_var})
    return (errs, wgt.eval(session = tf_sess), y_out)

# high iteration required for convergence to known solution as a result of \n
#   naive initialization
l_rate = 0.1
epochs = 5000 
err_vec, wgt_fit, pred = fit_linear_gd_opt(tf_sess = sess, x_var = x, y_var = y, 
                                 l_rate = l_rate, epoch = epochs)

# confirm smooth decreasing error / loss
plt.plot(np.arange(epochs), err_vec)
plt.ylabel('SSE')
plt.xlabel('Iterations')

# test native tensorflow least squares solution
# note: no regularization, documentation flags this function as high error \n
#       with weight penalization
tf_native_ls = tf.matrix_solve_ls(
    matrix = tf.convert_to_tensor(x, np.float32),
    rhs = tf.convert_to_tensor(y, np.float32),
    l2_regularizer=0.0,
    fast=True
)
tf_solve_ls_wgts = tf_native_ls.eval(session = sess)

#  scikit-learn fit as a baseline for comparison
#lr = lm.LinearRegression()
#lr.fit(x, y)
#sk_lr_wgt = lr.coef_

sess.close()