import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Cost function
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
prediction_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
prediction_accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))

# Algorithm
train_step = tf.train.FtrlOptimizer(2).minimize(cross_entropy)


with tf.name_scope("Softmax") as scope:
    summary_slope = tf.summary.scalar("W", tf.reduce_mean(W))
    summary_offset = tf.summary.scalar("b", tf.reduce_mean(b))
    summary_loss = tf.summary.scalar("xentropy", cross_entropy)
summary_all = tf.summary.merge([summary_slope, summary_offset, summary_loss])

with tf.Session() as sess:
    log = tf.summary.FileWriter("./logs")
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, {x: batch_xs, y_true: batch_ys})
        log_summary = sess.run(summary_all, {x: batch_xs, y_true: batch_ys})
        log.add_summary(log_summary, i)

    print(sess.run(prediction_accuracy, {x: mnist.test.images, y_true: mnist.test.labels}))

