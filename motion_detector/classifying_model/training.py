import tensorflow as tf
import tensorflowjs as tfjs
from keras import utils
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


def get_centre(keypoints, left_part, right_part):
    left_point = tf.gather(keypoints, left_part.value, axis=1)
    right_point = tf.gather(keypoints, right_part.value, axis=1)
    centre_point = (left_point + right_point)/2
    return centre_point
