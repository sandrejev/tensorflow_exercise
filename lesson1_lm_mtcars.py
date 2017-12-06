import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
#from tensorflow.python import debug as tf


mtcars = pd.read_csv("mtcars.csv")

with tf.name_scope("LM") as scope:
    var_slope = tf.Variable(0.1, dtype=tf.float32)
    var_offset = tf.Variable(0.01, dtype=tf.float32)
    data = tf.placeholder(tf.float32)
    model = data*var_slope + var_offset

with tf.name_scope("LossFunction") as scope:
    outcome = tf.placeholder(tf.float32)
    model_loss = tf.reduce_sum(tf.square(outcome - model))

with tf.name_scope("LM_vars") as scope:
    summary_slope = tf.summary.scalar("slope", tf.reduce_mean(var_slope))
    summary_offset = tf.summary.scalar("offset", tf.reduce_mean(var_offset))
    summary_loss = tf.summary.scalar("loss", model_loss)
summary_all = tf.summary.merge_all()

model_data = {data: mtcars.hp, outcome: mtcars.disp}
optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(model_loss)
with tf.Session() as sess:
    log = tf.summary.FileWriter("./logs", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    c_prev = 1e+100
    for i in range(1000):
        _, c = sess.run([train, model_loss], model_data)
        if i % 10 == 0:
            summary = sess.run(summary_all, model_data)
            log.add_summary(summary, i)

    curr_slope, curr_offset, curr_loss = sess.run([var_slope, var_offset, model_loss], model_data)
    print("disp = {:.2f}*hp + {:.2f}".format(curr_slope, curr_offset))

