'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from model import Model

class Model_Head_Pose(Model):
    def predict(self, image):
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        raise NotImplementedError

    def preprocess_output(self, outputs):
        raise NotImplementedError
