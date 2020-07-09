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
import numpy as np

from argparse import ArgumentParser
from input_feeder import InputFeeder
from model_face_detection import Model_Face_Detection
from model_gaze import Model_Gaze
from model_head_pose import Model_Head_Pose
from model_landmark import Model_Landmark
from mouse_controller import MouseController


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
    parser.add_argument("-v",
                        "--visual",
                        required=False,
                        action="store_true",
                        help="Enable visualization on the intermediate models")
    parser.add_argument("-nm",
                        "--no_move",
                        required=False,
                        action="store_false",
                        help="Disables the movement of the mouse")
    return parser

def main(args):
    mouse_controller = MouseController('medium','fast')
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
    frame_count = 0

    for flag, frame in input_image.next_batch():
        if not flag:
            break
        frame_count += 1
        pressed_key = cv2.waitKey(60)
        # Get image crop of image from face detection
        fd_coords = fd_model.predict(frame, prob)
        if len(fd_coords) == 0:
            print("No face found...")
            if pressed_key == 27:
                break
            else:
                continue
        # Get first face available
        fd_coords = fd_coords[0]
        # Crop image [ymin:ymax, xmin:xmax]
        cropped_image = frame[fd_coords[1]:fd_coords[3], fd_coords[0]:fd_coords[2]]
        yaw, pitch, roll = p_model.predict(cropped_image)
        left_eye, right_eye = fl_model.predict(cropped_image)
        left_eye_img = cropped_image[left_eye[1]:left_eye[3], left_eye[0]:left_eye[2]]
        right_eye_img = cropped_image[right_eye[1]:right_eye[3], right_eye[0]:right_eye[2]]
        if left_eye_img.shape != (20,20,3) and right_eye_img.shape != (20,20,3):
            print("Could not find eyes...")
            continue
        if left_eye_img.shape != (20,20,3):
            print("Could not find left eye..")
            left_eye_img = right_eye_img
        elif right_eye_img.shape != (20,20,3):
            print("Could not find right eye..")
            right_eye_img = left_eye_img 
        #Estimate gaze
        mouse_x, mouse_y = g_model.predict(left_eye_img, right_eye_img, [yaw,pitch,roll])
        if args.visual:
            # Face Outline
            cv2.rectangle(frame, (fd_coords[0],fd_coords[1]),(fd_coords[2],fd_coords[3]),(0,255,100))
            # Eye Outlines
            size = 20
            left_cornerx = left_eye[0] + fd_coords[0]
            left_cornery = left_eye[1] + fd_coords[1]
            left_eye = [left_cornerx, left_cornery, left_cornerx + size, left_cornery + size]
            right_cornerx = right_eye[0] + fd_coords[0]
            right_cornery = right_eye[1] + fd_coords[1]
            right_eye = [right_cornerx, right_cornery, right_cornerx + size, right_cornery + size]
            cv2.rectangle(frame, (left_eye[0],left_eye[1]),(left_eye[2],left_eye[3]),(0,10,200), thickness=4)
            cv2.rectangle(frame, (right_eye[0],right_eye[1]),(right_eye[2],right_eye[3]),(0,10,200), thickness=4)

        cv2.imshow("Image",frame)
        # Perfomance Dependacy
        if frame_count % 5 == 0 and args.no_move:
            mouse_controller.move(mouse_x, mouse_y)
        


if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(args)
