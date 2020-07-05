'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from model import Model


class Model_Landmark(Model):
    def predict(self, image):
        height, width, _ = image.shape
        processed_image = self.preprocess_input(image)
        outputs = self.plugin_net.infer({self.input_name:processed_image})
        coords = self.preprocess_output(outputs, height, width)
        
        return coords['left_eye'], coords['right_eye']

    def preprocess_input(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        input_image = np.expand_dims(resized_image, axis=0).transpose((0,3,1,2))
        return input_image

    def preprocess_output(self, outputs, h_scale, w_scale):
        output = outputs[self.output_names][0]

        left_eye_x = int(output[0][0][0] * w_scale)
        left_eye_y = int(output[1][0][0] * h_scale)
        right_eye_x = int(output[2][0][0] * w_scale)
        right_eye_y = int(output[3][0][0] * h_scale)

        leye_minx = left_eye_x - 10
        leye_miny = left_eye_y - 10
        leye_maxx = left_eye_x + 10
        leye_maxy = left_eye_y + 10

        reye_minx = right_eye_x - 10
        reye_miny = right_eye_y - 10
        reye_maxx = right_eye_x + 10
        reye_maxy = right_eye_y + 10

        return {'left_eye': (leye_minx,leye_miny,leye_maxx, leye_maxy), 
                'right_eye': (reye_minx,reye_miny,reye_maxx, reye_maxy)}
