# Imports for ImageWithFeaturesDataGenerator
from glob import glob
from PIL import Image
import numpy as np
from tensorflow.python.keras._impl.keras import backend as K
import tensorflow as tf




class BBox(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xmin = self.x - self.w/2.
        self.xmax = self.x + self.w/2.
        self.ymin = self.y - self.h/2.
        self.ymax = self.y + self.h/2.

    @staticmethod
    def from_corners(xmin, xmax, ymin, ymax):
        x = (xmax + xmin)/2.
        y = (ymax + ymin)/2.
        w = xmax - xmin
        h = ymax - ymin

        return BBox(x, y, w, h)

    @property
    def center(self):
        return (self.y, self.x)

    def grid_position(self, img_size, grid_shape):
        cell_shape = tuple(s[1] / float(s[0]) for s in zip(grid_shape, img_size))
        grid_coords = tuple(sum(self.center[i] > np.linspace(-1e-10, img_size[i], num=grid_shape[i]+1))-1 for i in range(2))
        within_grid_coords = tuple((bb - (g+0.5)*c)/c for bb,g,c in zip(self.center, grid_coords, cell_shape))

        return grid_coords, within_grid_coords

    def as_list(self):
        return [self.x, self.y, self.w, self.h]

    def iou(self, box2):
        intersect_w = self.interval_overlap([self.xmin, self.xmax], [box2.xmin, box2.xmax])
        intersect_h = self.interval_overlap([self.ymin, self.ymax], [box2.ymin, box2.ymax])

        intersect = intersect_w * intersect_h
        union = self.w * self.h + box2.w * box2.h - intersect

        return float(intersect) / union

class YoloImageGenerator(object):
    def __init__(self, classes, target_size, grid_shape=(7,7), anchors_size=2):
        self.classes = classes
        self.target_size = target_size
        self.anchors_size = anchors_size
        self.grid_shape = grid_shape

    def flow_from_directory(self, images_path, callback, batch_size=10):
        images = glob(images_path)
        batch_index = 0
        total_batches_seen = 0
        n = len(images)

        while 1:
            current_index = (batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                batch_index += 1
            else:
                current_batch_size = n - current_index
                batch_index = 0
            total_batches_seen += 1

            batch_x = np.zeros((current_batch_size,) + self.target_size + (3,), dtype=K.floatx())
            batch_y = None
            batch_images_paths = images[current_index:current_index + current_batch_size]
            for i, image_path in enumerate(batch_images_paths):
                x, y = callback(self, image_path)
                if i == 0:
                    batch_y = np.zeros((current_batch_size,) + y.shape, dtype=K.floatx())

                batch_x[i] = x
                batch_y[i] = y

            print("batch {}: {}".format(batch_index, np.sum(batch_y)))

            yield batch_x, batch_y


#grid=0, batch_size=0, anchors_size=0, debug=False
def yolo_loss(y_true, y_pred):
    anchors_size = 2 # TODO: detect automatically
    pred_box_xy = tf.concat([y_pred[..., 0:5 * anchors_size:5], y_pred[..., 1:5 * anchors_size:5]], 3)
    pred_box_wh = tf.concat([y_pred[..., 2:5 * anchors_size:5], y_pred[..., 3:5 * anchors_size:5]], 3)
    pred_box_conf = y_pred[..., 4:5 * anchors_size:5]
    pred_class = y_pred[..., (anchors_size * 5):]

    true_box_xy = tf.concat([y_true[..., 0:5 * anchors_size:5], y_true[..., 1:5 * anchors_size:5]], 3)
    true_box_wh = tf.concat([y_true[..., 2:5 * anchors_size:5], y_true[..., 3:5 * anchors_size:5]], 3)
    true_box_conf = y_true[..., 4:5 * anchors_size:5]
    true_class = y_true[..., (anchors_size * 5):]

    # Create two matrices. One containing 1s for every box having an object, and another with oposite values
    obj_threshold = 0.6
    true_box_obj1 = tf.to_float(true_box_conf >= obj_threshold)
    true_box_noobj1 = tf.to_float(true_box_conf < obj_threshold)
    true_box_obj2 = tf.tile(true_box_obj1, (1, 1, 1, anchors_size))
    true_cell_obj1 = tf.reduce_max(true_box_obj1, reduction_indices=[3])

    # Calculate parts of the loss function
    loss_xy = tf.reduce_sum(true_box_obj2 * tf.squared_difference(pred_box_xy, true_box_xy))
    loss_wh = tf.reduce_sum(true_box_obj2 * tf.squared_difference(tf.sqrt(pred_box_wh), tf.sqrt(true_box_wh)))
    loss_conf = tf.reduce_sum(true_box_obj1 * tf.squared_difference(pred_box_conf, true_box_conf))
    loss_notconf = tf.reduce_sum(true_box_noobj1 * tf.squared_difference(pred_box_conf, true_box_conf))
    loss_class = tf.reduce_sum(true_cell_obj1 * tf.reduce_sum(tf.squared_difference(pred_class, true_class), reduction_indices=(3)))

    # this is the original loss function with original coefficients from the paper
    # https://arxiv.org/pdf/1506.02640.pdf
    loss = 5. * loss_xy + 5. * loss_wh + loss_conf + 0.5 * loss_notconf + loss_class

    #loss = tf.Print(loss, [tf.reduce_sum(true_box_xy), tf.reduce_sum(true_box_wh), tf.reduce_sum(true_box_conf), tf.reduce_sum(true_class)], summarize=1000, message="\ntrue columns:\n")
    #loss = tf.Print(loss, [tf.reduce_sum(pred_box_xy), tf.reduce_sum(pred_box_wh), tf.reduce_sum(pred_box_conf), tf.reduce_sum(pred_class)], summarize=1000, message="\npred columns:\n")
    #loss = tf.Print(loss, [loss_xy, loss_wh, loss_conf, loss_notconf, loss_class], summarize=1000, message="\nloss parts:\n")

    return loss
