'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from model import Model

class Model_Head_Pose(Model):
    def predict(self, image):
        processed_image = self.preprocess_input(image)
        outputs = self.plugin_net.infer({self.input_name: processed_image})
        return self.preprocess_output(outputs)

    def preprocess_output(self, outputs):
        yaw = outputs['angle_y_fc'][0][0]
        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]
        return yaw, pitch, roll
