import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np


mtcars = pd.read_csv("mtcars.csv")

with tf.Session() as sess:
    features_columns = [tf.feature_column.numeric_column("hp")]
    regressor = tf.estimator.LinearRegressor(feature_columns=features_columns)
    input_fn = tf.estimator.inputs.pandas_input_fn(x=mtcars[["hp"]], y=mtcars.disp, shuffle=False, num_epochs=10000)
    test_fn = tf.estimator.inputs.numpy_input_fn(x={"hp": np.array([0,1])}, shuffle=False, num_epochs=1)

    regressor.train(input_fn, steps=1000)
    results = list(regressor.predict(test_fn))
    print(results)
    offset = results[0]['predictions'][0]
    slope = results[1]['predictions'][0] - offset
    print("disp = {:.2f} * hp + {:.2f}".format(slope, offset))