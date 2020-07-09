'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
from math import cos, sin, pi
from model import Model

class Model_Gaze(Model):
    def load_model(self, model_xml, cpu_ext, device):
        super().load_model(model_xml,cpu_ext,device)
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        le_img = self.preprocess_input(left_eye_image)
        re_img = self.preprocess_input(right_eye_image)
        outputs = self.plugin_net.infer({
            'left_eye_image':le_img, 
            'right_eye_image':re_img,
            'head_pose_angles': head_pose_angles, 
            })
        return self.preprocess_output(outputs, head_pose_angles[2])


    def preprocess_output(self, outputs, roll):
        # Estimate x, y based on carthesian output of both eye vectors.
        output = outputs[self.output_names[0]][0]
        degree = pi / 180

        # Thetha values
        cos_theta = cos(roll*degree)
        sin_theta = sin(roll*degree)

        # Intermediate values 
        mouse_x = output[0] * cos_theta + output[1] * sin_theta
        mouse_y = output[1] * cos_theta - output[0] * sin_theta
        return mouse_x, mouse_y
