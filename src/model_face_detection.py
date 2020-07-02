'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from model import Model

class Model_Face_Detection(Model):
    '''
    Class for the Face Detection Model.
    '''
    def predict(self, image, prob):
        height, width, _ = image.shape
        processed_image = self.preprocess_input(image)       
        outputs = self.plugin_net.infer({self.input_name:processed_image})
        coords = self.preprocess_output(outputs,prob, height, width)
        return coords

    def preprocess_input(self, image):
        # Add batch dimension shape: [1x3x384x672] - An input image in the format [BxCxHxW]
        resized_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        input_image = np.expand_dims(resized_image, axis=0).transpose((0,3,1,2))
        return input_image

    def preprocess_output(self, outputs, prob, h_scale, w_scale):
        coords = []
        outputs = outputs[self.output_names][0][0]
        for _,_,conf,xmin,ymin,xmax,ymax in outputs:
            if conf > prob:
                coords.append([
                    int(xmin * w_scale),
                    int(ymin * h_scale),
                    int(xmax * w_scale),
                    int(ymax * h_scale)
                    ])
        return coords
