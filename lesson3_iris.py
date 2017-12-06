import tensorflow as tf
import numpy as np

classes = {0: "Iris setosa", 1: "Iris versicolor", 2: "Iris virginica"}

# Load data
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename="IRIS_data/iris_training.csv",
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename="IRIS_data/iris_test.csv",
    target_dtype=np.int,
    features_dtype=np.float32)

feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(training_set.data)}, y=np.array(training_set.target),
    num_epochs=None, shuffle=True)
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(test_set.data)}, y=np.array(test_set.target),
    num_epochs=1, shuffle=False)

# Create and train classifier
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir="./IRIS_model")
classifier.train(input_fn=train_input_fn, steps=1000)

# Evaluate classifier
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": new_samples}, num_epochs=1, shuffle=False)
predicted_classes = ", ".join([classes[int(p["classes"][0])] for p in classifier.predict(input_fn=predict_input_fn)])
print("Predictions from new data: {}\n" .format(predicted_classes))
