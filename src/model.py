'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import numpy as np
from openvino.inference_engine import IECore

class Model:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self):
        self.plugin = None
        self.plugin_net = None
        self.network = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None

    def load_model(self, model_xml, cpu_ext, device):
        self.plugin = IECore()
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self.network = self.plugin.read_network(model=model_xml,
                                                weights=model_bin)
        if cpu_ext is not None:
            self.plugin.add_extension(cpu_ext, device)
        layers_map = self.plugin.query_network(network=self.network,
                                               device_name=device)
        layers = self.network.layers.keys()
        unsupported_layers = [
            layer for layer in layers_map if layer not in layers
        ]
        if len(unsupported_layers) != 0:
            print("Found unsupported layers: {}".format(unsupported_layers))
            print("Please check the extension for availability.")
            exit(1)

        self.plugin_net = self.plugin.load_network(self.network,
                                                   device,
                                                   num_requests=1)
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

    def preprocess_input(self, image):
        resized_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        input_image = np.expand_dims(resized_image, axis=0).transpose((0,3,1,2))
        return input_image

