#
# Load previously trained NN model for NMinst digit classification
#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], './SavedModel_NMIST_NN')
    y_predict = tf.get_collection("y")[0]

    result = sess.run(y_predict, feed_dict={
        "x:0": mnist.validation.images,
        "y_true:0": mnist.validation.labels,
        "dropout/keep_prob:0": 1.0})

    print(tf.confusion_matrix(
        predictions=[np.argmax(x, 0) for x in mnist.validation.labels],
        labels=[np.argmax(x, 0) for x in result]).eval())
