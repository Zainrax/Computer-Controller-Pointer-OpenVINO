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
from argparse import ArgumentParser

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
    print(args.device)

if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(args)
