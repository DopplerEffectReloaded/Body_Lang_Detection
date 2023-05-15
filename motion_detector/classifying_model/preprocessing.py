# This file Processes the images, detects the poses using movenet API of tensorflow.js,
# and creates CSV files with its coordinates

# All dependencies required:
import tensorflow as tf
from movenet import Movenet
from data import BodyPart
import os
import wget
import tqdm
import pandas as pd
import numpy as np
import csv

# This installs the required file in the current directory
if "movenet_thunder.tflite" not in os.listdir():
    wget.download("https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
                  "movenet_thunder.tflite")  # Downloads the thunder (more precise) model of movenet

movenet = Movenet("movenet_thunder")


def detect(input_tensor, inference_count=2):
    """
    This function runs the detection algorithm on the input image.

    Params:
    input_tensor: This is a input of dimensions [height, width, 3] (3 is the image channel) tensor of type tf.float32
                The height and width can be any given height and width as tensor resizes it according to need.
    inference_count: This is the number of times the model should run on the same image to improve accuracy.
                Default value : 2, Recommended Maximum : 5

    Returns:
        A person entity as detected by Movenet.SinglePose
    """
    im_height, im_width, channel = input_tensor.shape

    # Detecting poses input image

    movenet.detect(input_tensor.numpy(), reset_crop_region=False)  # Set to false since trained on static images

    # Uses prev result to find ROI and increase accuracy

    for _ in range(inference_count):
        detect_person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)

    return detect_person


class PreProcessor:

    def __init__(self, images_in_dir, csv_out_path):
        self._images_in_dir = images_in_dir
        self._csv_out_path = csv_out_path
        self._csv_out_folder_per_pose = "csv_for_each_pose"
        self._message = []

        if self._csv_out_folder_per_pose not in os.listdir():
            os.makedirs(self._csv_out_folder_per_pose)

        self._pose_names = sorted([i for i in os.listdir(images_in_dir)])
