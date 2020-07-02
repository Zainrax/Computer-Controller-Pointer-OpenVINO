"""
Tasked with handling CLI interaction & performing the model
parsing to provide the proper output. Takes in OpenVINO models
for face detection to determine boundry, from which the cropped
head is infered to find the head pose & landmarks, fed into
the gaze model to find where a person is looking, controlling
where the point will be placed.

Author: Patrick Baxter
Date: 30/06/2020
License: MIT
"""
import os
import cv2

from argparse import ArgumentParser
from input_feeder import InputFeeder
from model_face_detection import Model_Face_Detection
from model_gaze import Model_Gaze
from model_head_pose import Model_Head_Pose
from model_landmark import Model_Landmark


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd",
                        "--face_detection",
                        required=True,
                        type=str,
                        help="Path to an xml file for the face detection model.")
    parser.add_argument("-fl",
                        "--face_landmark",
                        required=True,
                        type=str,
                        help="Path to an xml file for the face landmark model.")
    parser.add_argument("-g",
                        "--gaze_detection",
                        required=True,
                        type=str,
                        help="Path to an xml file for the gaze estimation model.")
    parser.add_argument("-p",
                        "--pose_detection",
                        required=True,
                        type=str,
                        help="Path to an xml file face pose detection model.")
    parser.add_argument("-i",
                        "--input",
                        required=True,
                        type=str,
                        help="Path to video file, or cam for Web Cam Usage.")
    parser.add_argument("-pt",
                        "--prob_threshold",
                        type=float,
                        default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-l",
                        "--cpu_extension",
                        required=False,
                        type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                        "Absolute path to a shared library with the"
                        "kernels impl.")
    parser.add_argument("-d",
                        "--device",
                        type=str,
                        default="CPU",
                        help="Specify the target device to infer on: "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                        "will look for a suitable plugin for device "
                        "specified (CPU by default)")
    return parser

def main(args):
    device = args.device
    extension = args.cpu_extension
    input_path = args.input
    prob = args.prob_threshold
    if input_path.lower() == "cam":
        input_image = InputFeeder("cam")
    else:
        if os.path.isfile(input_path):
            input_image = InputFeeder("video", input_path)
        else:
            print("Invalid path to file used: {}".format(input_path))
            exit(1)
    fd_model = Model_Face_Detection()
    fl_model = Model_Landmark()
    g_model = Model_Gaze()
    p_model = Model_Head_Pose()

    fd_model.load_model(args.face_detection, extension, device)
    fl_model.load_model(args.face_landmark, extension, device)
    g_model.load_model(args.gaze_detection, extension, device)
    p_model.load_model(args.pose_detection, extension, device)
        
    input_image.load_data()
    for flag, frame in input_image.next_batch():
        if not flag:
            break
        pressed_key = cv2.waitKey(60)
        # Get image crop of image from face detection
        coords = fd_model.predict(frame, prob)


if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(args)
