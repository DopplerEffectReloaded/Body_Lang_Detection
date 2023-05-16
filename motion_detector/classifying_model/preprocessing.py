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
        self.images_in_dir = images_in_dir
        self.csv_out_path = csv_out_path
        self.csv_out_folder_per_pose = "csv_for_each_pose"
        self.message = []

        if self.csv_out_folder_per_pose not in os.listdir():
            os.makedirs(self.csv_out_folder_per_pose)

        self.pose_names = sorted([i for i in os.listdir(images_in_dir)])

    def process(self, min_detect_threshold=0.1):
        """
        Processes the images in the folder path specified.

        Params:
        min_detect_threshold: The minimum score the image should pass to be transferred to the csv file
        Void method
        """

        for pose in self.pose_names:
            # Processes the images folder by folder
            pose_folder = os.path.join(self.images_in_dir, pose)
            output_path = os.path.join(self.csv_out_folder_per_pose, pose + ".csv")

            with open(output_path, 'w') as output_file:
                out_write = csv.writer(output_file, quoting=csv.QUOTE_MINIMAL)

                image_names = sorted([i for i in os.listdir(pose_folder)])
                count_valid = 0
                for image in tqdm.tqdm(iterable=image_names, desc="\n-------PROCESSING FILES-----------\n",
                                       total=len(image_names)):
                    path = os.path.join(pose_folder, image)

                    try:
                        image = tf.io.read_file(path)
                        image = tf.io.decode_jpeg(image)
                    except:
                        self.message.append("Skipped" + path + "Invalid Image")
                        continue

                    if image.shape[2] != 3:
                        self.message.append("Skipped" + path + "Image is not RGB")
                        continue

                    person = detect(image)

                    min_score = min([keypoint.score for keypoint in person.keypoints])
                    image_pass = min_score > min_detect_threshold
                    if not image_pass:
                        self.message.append("Skipped" + path + "Pose not detected accurately")
                        continue
                    count_valid += 1
                    # Save coordinates if the keypoints of all poses are above threshold

                    pose_keypoints = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                                               for keypoint in person.keypoints], dtype=np.float32)

                    coordinates = pose_keypoints.flatten().astype(np.str).tolist()
                    out_write.writerow([image] + coordinates)

        print(self.message)

        all_coordinates_df = self.all_coordinates_as_dataframe()
        all_coordinates_df.to_csv(self.csv_out_path, index=False)

    def all_coordinates_as_dataframe(self):
        pass
