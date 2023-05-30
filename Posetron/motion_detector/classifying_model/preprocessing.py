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

    movenet.detect(input_tensor.numpy(), reset_crop_region=False)

    # Uses prev result to find ROI and increase accuracy

    for _ in range(inference_count):
        detect_person = movenet.detect(input_tensor.numpy(), reset_crop_region=True)

    return detect_person


class PreProcessor:

    def __init__(self, images_in_dir, csv_out_path):
        self.images_in_dir = images_in_dir
        self.csv_out_path = csv_out_path
        self.csv_out_folder_per_pose = "csv_for_each_pose"
        self.message = []

        if self.csv_out_folder_per_pose not in os.listdir():
            os.makedirs(self.csv_out_folder_per_pose)

        self.pose_names = np.array(sorted([i for i in os.listdir(images_in_dir)]))

    def processor(self, min_detect_threshold=0.1):
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

                image_names = np.array([i for i in os.listdir(pose_folder)])
                count_valid = 0
                for image in tqdm.tqdm(iterable=image_names):
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
                    if pose_folder == "C:\\Users\\SACHIN\\Body_Lang_Detection\\motion_detector\\classifying_model\\poses\\train\\shoulder":
                        min_detect_threshold = 0.05
                    image_pass = min_score > min_detect_threshold
                    if not image_pass:
                        self.message.append("Skipped" + path + "Pose not detected accurately")
                        continue
                    count_valid += 1
                    # Save coordinates if the keypoints of all poses are above threshold

                    pose_keypoints = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                                               for keypoint in person.keypoints], dtype=np.float32)

                    coordinates = pose_keypoints.flatten().astype(np.str_).tolist()
                    out_write.writerow([image] + coordinates)

        print(self.message)

        all_coordinates_df = self.all_coordinates_as_dataframe()
        all_coordinates_df.to_csv(self.csv_out_path, index=False)

    def all_coordinates_as_dataframe(self):
        df_return = None

        for index, name in enumerate(self.pose_names):
            output_path = os.path.join(self.csv_out_folder_per_pose, name + ".csv")

            df = pd.read_csv(output_path, header=None)

            # Creating labels
            df["pose_no"] = [index] * len(df)
            df["pose_name"] = [name] * len(df)

            df[df.columns[0]] = name + '/' + df[df.columns[0]]

            if df_return is None:
                df_return = df
            else:
                df_return = pd.concat([df_return, df], axis=0)

            bodypoint_list_name = [[bodypart.name + "_x", bodypart.name + "_y" + bodypart.name + "_score"]
                                   for bodypart in BodyPart]

            titles = []

            for cols in bodypoint_list_name:
                titles += cols

            titles = ["filename"] + titles
            titles_map = {df_return.columns[i]: titles[i] for i in range(len(titles))}

            df_return.rename(titles_map, axis=1, inplace=True)

            return df_return


images_path = os.path.join("poses", "train")
csv_output_path = "train_data.csv"

training_processor = PreProcessor(images_in_dir=images_path, csv_out_path=csv_output_path)
training_processor.processor()

images_path = os.path.join('poses', 'test')
csv_output_path = 'test_data.csv'
test_preprocessor = PreProcessor(images_path, csv_output_path)

test_preprocessor.processor()
