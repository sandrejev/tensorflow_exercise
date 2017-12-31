from glob import glob
import numpy as np
from tensorflow.python.keras._impl.keras import backend as K
import tensorflow as tf
import cv2




class BBox(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

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

    @property
    def xmin(self):
        return self.x - self.w / 2.

    @property
    def xmax(self):
        return self.x + self.w / 2.

    @property
    def ymin(self):
        return self.y - self.h / 2.

    @property
    def ymax(self):
        return self.y + self.h / 2.

    def resize(self, scaler):
        self.x *= scaler[1]
        self.y *= scaler[0]
        self.w *= scaler[1]
        self.h *= scaler[0]
    #
    # def grid_position(self, img_size, grid_shape):
    #     cell_shape = tuple(s[1] / float(s[0]) for s in zip(grid_shape, img_size))
    #     grid_coords = tuple(sum(self.center[i] > np.linspace(-1e-10, img_size[i], num=grid_shape[i]+1))-1 for i in range(2))
    #     within_grid_coords = tuple((bb - (g+0.5)*c)/c for bb,g,c in zip(self.center, grid_coords, cell_shape))
    #
    #     return grid_coords, within_grid_coords

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

            #print("batch {}: {}".format(batch_index, np.sum(batch_y)))

            yield batch_x, batch_y


#grid=0, batch_size=0, anchors_size=0, debug=False
def yolo_loss(y_true, y_pred):
    anchors_size = 2 # TODO: detect automatically
    pred_box_xy = tf.tanh(tf.concat([y_pred[..., 0:5 * anchors_size:5], y_pred[..., 1:5 * anchors_size:5]], 3))
    pred_box_wh = tf.sigmoid(tf.concat([y_pred[..., 2:5 * anchors_size:5], y_pred[..., 3:5 * anchors_size:5]], 3))
    pred_box_conf = tf.sigmoid(y_pred[..., 4:5 * anchors_size:5])
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
    loss_xy = tf.reduce_mean(tf.reduce_sum(true_box_obj2 * tf.squared_difference(pred_box_xy, true_box_xy), reduction_indices=(1,2,3)))
    loss_wh = tf.reduce_mean(tf.reduce_sum(true_box_obj2 * tf.squared_difference(tf.sqrt(pred_box_wh), tf.sqrt(true_box_wh)), reduction_indices=(1,2,3)))
    loss_conf = tf.reduce_mean(tf.reduce_sum(true_box_obj1 * tf.squared_difference(pred_box_conf, true_box_conf), reduction_indices=(1,2,3)))
    loss_notconf = tf.reduce_mean(tf.reduce_sum(true_box_noobj1 * tf.squared_difference(pred_box_conf, true_box_conf), reduction_indices=(1,2,3)))
    loss_class = tf.reduce_mean(tf.reduce_sum(true_cell_obj1 * tf.reduce_sum(tf.squared_difference(pred_class, true_class), reduction_indices=(3)), reduction_indices=(1,2)))

    # this is the original loss function with original coefficients from the paper
    # https://arxiv.org/pdf/1506.02640.pdf
    loss = 5. * loss_xy + 5. * loss_wh + loss_conf + 0.5 * loss_notconf + loss_class

    #loss = tf.Print(loss, [tf.reduce_sum(true_box_xy), tf.reduce_sum(true_box_wh), tf.reduce_sum(true_box_conf), tf.reduce_sum(true_class)], summarize=1000, message="\ntrue columns:\n")
    #loss = tf.Print(loss, [tf.reduce_sum(pred_box_xy), tf.reduce_sum(pred_box_wh), tf.reduce_sum(pred_box_conf), tf.reduce_sum(pred_class)], summarize=1000, message="\npred columns:\n")
    #loss = tf.Print(loss, [loss_xy, loss_wh, loss_conf, loss_notconf, loss_class], summarize=1000, message="\nloss parts:\n")

    return loss


def draw_grid(img, grid_shape):
    cell_shape = tuple(s[1] // s[0] for s in zip(grid_shape, img.shape))
    for x in range(grid_shape[1]):
        cv2.line(img, (cell_shape[1]*x, 0), (cell_shape[1]*x, img.shape[0]-1), (0,255,0), 1)
    for x in range(grid_shape[0]):
        cv2.line(img, (0, cell_shape[0]*x), (img.shape[1]-1, cell_shape[0]*x), (0,255,0), 1)


def draw_grid_values(img, grid, offset=(0.7, 0.3), fontScale=1):
    draw_grid(img, grid.shape)
    cell_shape = tuple(s[1] // s[0] for s in zip(grid.shape, img.shape))
    for (y,x), cl in np.ndenumerate(grid):
        pt = (round((x+offset[1])*cell_shape[1]), round((y+offset[0])*cell_shape[0]))
        cv2.putText(img, str(cl), pt, cv2.FONT_HERSHEY_DUPLEX, fontScale, color=(0,255,0), thickness=1)


def draw_bbox(img, bbox):
    cv2.line(img, (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmin), int(bbox.ymax)), (255,0,0), 2)
    cv2.line(img, (int(bbox.xmax), int(bbox.ymin)), (int(bbox.xmax), int(bbox.ymax)), (255,0,0), 2)
    cv2.line(img, (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmax), int(bbox.ymin)), (255,0,0), 2)
    cv2.line(img, (int(bbox.xmin), int(bbox.ymax)), (int(bbox.xmax), int(bbox.ymax)), (255,0,0), 2)

def position_absolute2grid(img, grid_shape, yx):
    yx = np.array(yx)
    cell_shape = np.array(img.shape[:2]) / np.array(grid_shape)
    yx_rep = np.stack([np.repeat(yx[0], grid_shape[0]), np.repeat(yx[1], grid_shape[1])])
    grid_coords = np.stack([np.linspace(-1e-10, img.shape[0], num=grid_shape[0] + 1)[:grid_shape[0]],
                            np.linspace(-1e-10, img.shape[1], num=grid_shape[1] + 1)[:grid_shape[1]]])
    grid_coords = np.sum(yx_rep > grid_coords, 1) - 1
    within_grid_coords = (yx - (grid_coords + 0.5) * cell_shape) / cell_shape * 2

    return grid_coords, within_grid_coords

def draw_predicted_boxes(img, prediction, classes):
    bnum = int((prediction.shape[2] - len(classes)) / 5)
    grid_shape = np.array(prediction.shape[:2])
    cell_shape = np.array(img.shape[:2]) / np.array(grid_shape)

    prediction_yxhw = prediction[:,:,np.tile(np.arange(4), (bnum,1)) + np.reshape(np.repeat(np.arange(0, bnum*5, 5), 4), (bnum,4))]
    prediction_conf = prediction[...,np.arange(4, bnum*5, 5)]
    cell_shape = np.array(img.shape[:2]) / np.array(prediction_yxhw.shape[:2])
    for gy,gx,b in np.ndindex(*prediction_yxhw.shape[:3]):
        if prediction_conf[gy,gx,b] < 0.1:
            continue

        yxhw = prediction_yxhw[gy,gx,b,:]
        yx = (np.array([gy, gx])+0.5)*cell_shape + yxhw[:2]*cell_shape/2
        hw = yxhw[2:4] * img.shape[:2]
        print(yxhw, yx, hw)
        #xywh = xywh.astype(int)
        bb = BBox(yx[1], yx[0], hw[1], hw[0])
        draw_bbox(img, bb)

    draw_grid(img, grid_shape)