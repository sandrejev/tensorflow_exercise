import pickle
import numpy as np
import tensorflow as tf
from random import sample
import os
import shutil
import datetime
from tensorflow.contrib.keras import preprocessing

# from tensorflow.python import debug as tf_debug

# https://www.kaggle.com/dhayalkarsahilr/easy-image-augmentation-techniques-for-mnist
def augment_data(dataset, dataset_labels, augementation_factor=1, use_random_rotation=True, use_random_shear=True,
                 use_random_shift=True, use_random_zoom=True):
    augmented_image = []
    augmented_image_labels = []

    for num in range(0, dataset.shape[0]):
        for i in range(0, augementation_factor):
            # original image:
            augmented_image.append(dataset[num])
            augmented_image_labels.append(dataset_labels[num])

            if use_random_rotation:
                try:
                    augmented_image.append(preprocessing.image.random_rotation(dataset[num], 20, row_axis=0, col_axis=1, channel_axis=2))
                    augmented_image_labels.append(dataset_labels[num])
                except:
                    print("Failed rotate")

            if use_random_shear:
                try:
                    augmented_image.append(preprocessing.image.random_shear(dataset[num], 0.2, row_axis=0, col_axis=1, channel_axis=2))
                    augmented_image_labels.append(dataset_labels[num])
                except:
                    print("Failed shear")

            if use_random_shift:
                try:
                    augmented_image.append(preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
                    augmented_image_labels.append(dataset_labels[num])
                except:
                    print("Failed shift")

            if use_random_zoom:
                try:
                    augmented_image.append(preprocessing.image.random_zoom(dataset[num], (0.9, 0.9), row_axis=0, col_axis=1, channel_axis=2))
                    augmented_image_labels.append(dataset_labels[num])
                except:
                    print("Failed zoom")

    return np.array(augmented_image), np.array(augmented_image_labels)

def show(im, title=None):
    from PIL import Image
    im = Image.fromarray((im*255).astype(np.uint8), "RGB")
    im.show(title=title)


def dim(var, d):
    if isinstance(d, list):
        return [v.value for d_i, v in enumerate(var.get_shape()) if d_i in d]
    return var.get_shape()[d].value


def weights(shape, name="W", wd=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial, name=name)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='wd')
        tf.add_to_collection('L2', weight_decay)

    return var


def bias(shape, name="b"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# TODO: z-score
# mean = np.mean(x_train,axis=(0,1,2,3))
# std = np.std(x_train,axis=(0,1,2,3))
# x_train = (x_train-mean)/(std+1e-7)
# x_test = (x_test-mean)/(std+1e-7)

cifar_data = pickle.load(open("CIFAR_data/cifar.pcl", "rb"))
# print(cifar_data['train']['x'][0, 0:9, 0:9, 0:3])
# train_images = cifar_data['train']['x'] / 255.
# train_images = np.cast(train_images * 255., np.uint8)
# print(train_images[0, 0:9, 0:9, 0:3])
train_labels = cifar_data['train']['y']
train_images = cifar_data['train']['x'] / 255.0
#train_images = (train_images - np.mean(train_images, axis=0)) / np.std(train_images, axis=0)

print(train_images.shape)
print(train_labels.shape)
train_images, train_labels = augment_data(train_images, train_labels, augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, use_random_zoom=True)
# train_images = np.stack((train_images, train_images_a), axis=0)
# train_labels = np.stack((train_labels, train_labels_a), axis=0)
print(train_images.shape)
print(train_labels.shape)


test_images = cifar_data['test']['x'] / 255.0
#test_images = (test_images - np.mean(test_images, axis=0)) / np.std(test_images, axis=0)
test_labels = cifar_data['test']['y']
labels = cifar_data['labels']


# i = 101
# show(train_images[i], labels[np.argmax(train_labels[i])])
# exit()
has_lrn_layer = False
has_n_layers = 3
x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x")
y_ = tf.placeholder(tf.float32, [None, len(labels)], name="y_")

with tf.name_scope("conv1"):
    W_conv1 = weights([5, 5, dim(x, 3), 64])
    b_conv1 = bias([dim(W_conv1, 3)])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    tf.summary.histogram('histogram', b_conv1)
    #tf.summary.histogram('sparsity', tf.nn.zero_fraction(W_conv1))
    # [?, 384]
    # [?, 32, 32, 64]

with tf.name_scope("pool1"):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # [?, 16, 16, 64]

with tf.name_scope("conv2"):
    W_conv2 = weights([5, 5, dim(h_pool1, 3), 64])
    b_conv2 = bias([dim(W_conv2, 3)])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    tf.summary.histogram('histogram', b_conv2)
    #tf.summary.histogram('sparsity', tf.nn.zero_fraction(W_conv2))
    # [?, 16, 16, 64]

with tf.name_scope("pool2"):
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # [?, 8, 8, 64]

with tf.name_scope('fc1'):
    h_pool = h_pool2
    h_pool_flat = tf.reshape(h_pool, [-1, np.prod(dim(h_pool, [1, 2, 3]))])
    # [?, 4096]

    W_fc1 = weights([dim(h_pool_flat, 1), 384], wd=None)
    b_fc1 = bias([dim(W_fc1, 1)])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
    # tf.summary.histogram('histogram', h_fc1)
    # tf.summary.histogram('sparsity', tf.nn.zero_fraction(h_fc1))
    # [?, 384]

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
    W_fc2 = weights([dim(h_fc1_drop, 1), 192], wd=None)
    b_fc2 = bias([dim(W_fc2, 1)])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # tf.summary.histogram('histogram', h_fc2)
    # tf.summary.histogram('sparsity', tf.nn.zero_fraction(h_fc2))
    # [?, 192]

with tf.name_scope('softmax1_debug'):
    input_debug = h_fc2
    flat_debug = tf.reshape(input_debug, [-1, np.prod(dim(input_debug, [1, 2, 3]))])

    W_sm_debug = weights([dim(flat_debug, 1), len(labels)])
    b_sm_debug = bias([dim(W_sm_debug, 1)])

    h_softmax_debug = tf.nn.softmax(tf.matmul(flat_debug, W_sm_debug) + b_sm_debug +
                                    tf.reduce_sum(tf.get_collection('L2')))
    # [?, 10]

with tf.name_scope('optimizer'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_softmax_debug))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(h_softmax_debug, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    h_accuracy = tf.reduce_mean(correct_prediction)

# Summary
summary_xentropy = tf.summary.scalar("xentropy", cross_entropy)
summary_accuracy = tf.summary.scalar("accuracy", h_accuracy)
summary_all = tf.summary.merge_all()


tf.add_to_collection("y", h_conv1)

log_path = "./CIFAR_model/{}".format(datetime.datetime.today().strftime('%m-%d__%H-%M-%S'))
if os.path.exists(log_path):
    print("Remove {}...".format(log_path))
    shutil.rmtree(log_path)

saver = tf.saved_model.builder.SavedModelBuilder(log_path)
with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    train_writer = tf.summary.FileWriter(log_path, sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        train_sample = sample(range(len(train_labels)), 500)
        train_i = {x: train_images[train_sample], y_: train_labels[train_sample], keep_prob: 0.5}
        # show(train_images[train_sample][0], labels[np.argmax(train_labels[train_sample])])

        train_step.run(feed_dict=train_i)
        if i % 100 == 0:
            val_y, val_y_ = sess.run([tf.argmax(h_softmax_debug, 1), tf.argmax(y_, 1)], feed_dict=train_i) # tf.reduce_max(h_softmax, 1) +
            #print(" ".join("{}_{}".format(a, b) for a, b in zip(val_y_, val_y)))
            train_i[keep_prob] = 1.0
            val_summary, val_accuracy, val_xentropy = sess.run([summary_all, h_accuracy, cross_entropy], feed_dict=train_i)
            train_writer.add_summary(val_summary, i)

            val_accuracy_test = []
            for _ in range(10):
                test_sample = sample(range(len(test_labels)), 1000)
                test_i = {x: test_images[test_sample], y_: test_labels[test_sample], keep_prob: 1.0}
                val_accuracy_test_i = sess.run([h_accuracy], feed_dict=test_i)
                val_accuracy_test.append(val_accuracy_test_i)

            val_summary_accuracy_test = tf.Summary(value=[tf.Summary.Value(tag="accuracy_test",
                                                                           simple_value=np.mean(val_accuracy_test))])
            train_writer.add_summary(val_summary_accuracy_test, i)

            print("[{}] Accuracy: {:.2f} ({:.2f}), Entropy: {}".format(i, val_accuracy, np.mean(val_accuracy_test), val_xentropy))

    saver.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.TRAINING],
        signature_def_map={
            "model": tf.saved_model.signature_def_utils.predict_signature_def(inputs= {"x": x}, outputs= {"y": h_softmax_debug})
        },
        assets_collection=None)
    saver.save()

    # Evaluate classifier
    accuracy = sess.run(h_accuracy, feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
    print("Final accuracy: {}".format(accuracy))


labels = pickle.load(open("CIFAR_data/batches.meta", 'rb'), encoding='latin1')['label_names']



# train_batches = []
# for i in range(5):
#     with open("CIFAR_data/data_batch_{}".format(i+1), 'rb') as fo:
#          train_batches.append(pickle.load(fo, encoding='latin1'))
# train_features = np.moveaxis(np.concatenate([batch['data'].reshape(-1, 3, 32, 32) for batch in train_batches]), 1, 3)
# train_labels_class = np.concatenate([np.array(batch['labels'], dtype=np.uint8) for batch in train_batches]).flatten()
# train_labels = np.zeros((len(train_labels_class), len(labels)))
# train_labels[np.arange(len(train_labels_class)), train_labels_class] = 1
#
# test_batches = []
# with open("CIFAR_data/test_batch", 'rb') as fo:
#     test_batches.append(pickle.load(fo, encoding='latin1'))
# test_features = np.moveaxis(np.concatenate([batch['data'].reshape(-1, 3, 32, 32) for batch in test_batches]), 1, 3)
# test_labels_class = np.concatenate([np.array(batch['labels'], dtype=np.uint8) for batch in test_batches]).flatten()
# test_labels = np.zeros((len(test_labels_class), len(labels)))
# test_labels[np.arange(len(test_labels_class)), test_labels_class] = 1
#
# for i in sample(range(len(test_labels)), 5):
#     show(test_features[i], labels[np.argmax(test_labels[i])])
#
# cifar_data = {'train': {'y':train_labels, 'x': train_features}, 'test': {'y':test_labels, 'x': test_features}, 'labels': labels}
# pickle.dump(cifar_data, open("CIFAR_data/cifar.pcl", "bw"))