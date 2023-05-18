import keras.layers
import tensorflow as tf
import tensorflowjs as tfjs
from keras import utils
from keras import callbacks
import csv
import pandas as pd
from data import *
from sklearn.model_selection import train_test_split

tfjs_final_module_dir = "final_model"


def loadCSV(path):
    df = pd.read_csv(path)
    df.drop(["filename"], axis=1, inplace=True)
    pose_list = df.pop("pose_name").unique()

    pose_count = df.pop("pose_no")

    df_use = df.astype("float64")

    pose_count_binary = utils.to_categorical(pose_count)

    return df_use, pose_count_binary, pose_list


def get_size(keypoints, chest_multiplier=1.5):
    # Hips and shoulders centre
    centre_hips = (tf.gather(keypoints, BodyPart.LEFT_HIP, axis=1) + tf.gather(keypoints, BodyPart.RIGHT_HIP,
                                                                               axis=1)) / 2
    centre_shoulder = (tf.gather(keypoints, BodyPart.LEFT_SHOULDER, axis=1) +
                       tf.gather(keypoints, BodyPart.RIGHT_SHOULDER, axis=1)) / 2

    # get mid of chest to hips
    chest_mid = tf.linalg.norm(centre_shoulder - centre_hips)

    pose_centre = tf.expand_dims(centre_hips, axis=1)
    pose = tf.broadcast_to(pose_centre, [tf.size(keypoints) // 34, 17, 2])

    distance = tf.gather(params=keypoints - pose, indices=0, axis=0,
                         name="Distance from each point in posenet alg to pose centre")
    mean_distance = tf.linalg.norm(distance, axis=0)
    maximum_pose_points_distance = tf.reduce_max(mean_distance)

    max_pose_size = tf.maximum(chest_mid * chest_multiplier, maximum_pose_points_distance)

    return max_pose_size


def pose_keypoints(keypoints):
    pose_centre = tf.broadcast_to(tf.expand_dims((tf.gather(keypoints, BodyPart.LEFT_HIP, axis=1) +
                                                  tf.gather(keypoints, BodyPart.RIGHT_HIP)) / 2, axis=1),
                                  [tf.size(keypoints) // 34, 17, 2])
    keypoints_centred = keypoints - pose_centre

    size = get_size(keypoints_centred)
    keypoints_centred /= size

    return keypoints_centred


def embed_keypoints(keypoints_and_keypoint_scores):
    reshape_input = keras.layers.Reshape((17, 3))(keypoints_and_keypoint_scores)

    keypoints = pose_keypoints(reshape_input[:, :, 2])
    embedKeypoints = keras.layers.Flatten()(keypoints)

    return embedKeypoints


def preprocess_data(data):
    final_x_train = []
    for ele in range(data.shape[0]):
        final_x_train.append(tf.reshape
                             (embed_keypoints(tf.reshape(tf.convert_to_tensor(data.iloc[ele]), (1, 51))), 34))
    return tf.convert_to_tensor(final_x_train)


X, Y, pose_names = loadCSV("train_data.csv")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
test_x, test_y, _ = loadCSV("test_data.csv")

x_train_pass = preprocess_data(X)
x_test_pass = preprocess_data(x_test)
test_x_pass = preprocess_data(test_x)

# Creating additional layers on top of existing neural network

model = keras.models.Sequential()
model.add(keras.Input(shape=34))
model.add(keras.layers.Dense(128, activation=tf.nn.relu6))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation=tf.nn.relu6))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(pose_names), activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

check_path = "weights.best.hdf5"
check = keras.callbacks.ModelCheckpoint(check_path, monitor="val_accuracy", verbose=1,
                                        save_best_only=True, mode="max")
earlystop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=30)

print("TRAINING INITIALIZED.....")
trainer = model.fit(x_train_pass, y_train, epochs=200, batch_size=16,
                    validation_data=(x_test_pass, y_test), callbacks=[check, earlystop])


print("TRAINING ENDED\n\n\n\nSTARTING EVALUATION.....")
loss, acc = model.evaluate(test_x_pass, test_y)
print("Loss: ", loss)
print("Accuracy: ", acc)

tfjs.converters.save_keras_model(model, tfjs_final_module_dir)
print("Model saved at", tfjs_final_module_dir)