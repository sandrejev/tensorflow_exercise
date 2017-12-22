from __future__ import print_function
from tensorflow import keras
from tensorflow.python.keras import regularizers, layers, models, datasets, callbacks, preprocessing, optimizers
from datetime import datetime

batch_size = 128
num_classes = 10
epochs = 60
l2_weights = 1e-4
img_x, img_y = 32, 32

##############################
# Load data
##############################
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 3)
input_shape = (img_x, img_y, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

##############################
# Create model
##############################
model = models.Sequential()
for cnn_depth in range(1, 3):
    model.add(layers.Conv2D(32*cnn_depth, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(l2_weights)))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32*cnn_depth, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(l2_weights)))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Dropout(rate=0.1 + 0.025*cnn_depth))

model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

opt_rms = optimizers.RMSprop(lr=0.0005,decay=1e-6)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_rms,
              metrics=['accuracy'])

# Tensorboard
tensorboard = callbacks.TensorBoard(
    log_dir="./K_CIFAR_model/{}".format(datetime.today().strftime('%m-%d__%H-%M-%S')),
    histogram_freq=0,
    write_graph=True)
tensorboard.set_model(model)

##############################
# Train model
##############################
datagen = preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_test,y_test), callbacks=[tensorboard], steps_per_epoch=x_train.shape[0] // batch_size)
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test),
#           callbacks=[tensorboard])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
