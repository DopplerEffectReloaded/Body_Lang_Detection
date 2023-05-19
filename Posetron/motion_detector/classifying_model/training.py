import keras.layers
import tensorflow as tf
import tensorflowjs as tfjs
from keras import utils
from keras import callbacks
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
    centre_hips = (tf.gather(keypoints, BodyPart.LEFT_HIP.value, axis=1) + tf.gather(keypoints, BodyPart.RIGHT_HIP.value,
                                                                               axis=1)) / 2
    centre_shoulder = (tf.gather(keypoints, BodyPart.LEFT_SHOULDER.value, axis=1) +
                       tf.gather(keypoints, BodyPart.RIGHT_SHOULDER.value, axis=1)) / 2

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
    pose_centre = (tf.gather(keypoints, BodyPart.LEFT_HIP.value, axis=1) + tf.gather(keypoints, BodyPart.RIGHT_HIP.value, axis=1)) / 2
    pose_centre = tf.expand_dims(pose_centre, axis=1)
    pose_centre = tf.broadcast_to(pose_centre, [tf.size(keypoints) // 34, 17, 2])
    keypoints_centred = keypoints - pose_centre

    size = get_size(keypoints_centred)
    keypoints_centred /= size

    return keypoints_centred


def embed_keypoints(keypoints_and_keypoint_scores):
    reshape_input = keras.layers.Reshape((17, 3))(keypoints_and_keypoint_scores)

    keypoints = pose_keypoints(reshape_input[:, :, :2])
    embedKeypoints = keras.layers.Flatten()(keypoints)

    return embedKeypoints


def preprocess_data(data):
    final_x_train = []
    for ele in range(data.shape[0]):
        embed = embed_keypoints(tf.reshape(tf.convert_to_tensor(data.iloc[ele]), (1, 51)))
        final_x_train.append(tf.reshape(embed, 34))
    return tf.convert_to_tensor(final_x_train)


X, y, class_names = loadCSV('train_data.csv')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
X_test, y_test, _ = loadCSV('test_data.csv')

processed_X_train = preprocess_data(X_train)
processed_X_val = preprocess_data(X_val)
processed_X_test = preprocess_data(X_test)

inputs = tf.keras.Input(shape=34)
layer = keras.layers.Dense(128, activation=tf.nn.relu6)(inputs)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)
model = keras.Model(inputs, outputs)


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add a checkpoint callback to store the checkpoint that has the highest
# validation accuracy.
checkpoint_path = "weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=20)

# Start training
print('--------------TRAINING----------------')
history = model.fit(processed_X_train, y_train,
                    epochs=200,
                    batch_size=1,
                    validation_data=(processed_X_val, y_val),
                    callbacks=[checkpoint, earlystopping])

print('-----------------EVAUATION----------------')
loss, accuracy = model.evaluate(processed_X_test, y_test)
print('LOSS: ', loss)
print("ACCURACY: ", accuracy)

tfjs.converters.save_keras_model(model, tfjs_final_module_dir)
print('tfjs model saved at ', tfjs_final_module_dir)
