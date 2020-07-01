'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
from openvino.inference_engine import IECore

class Model:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self):
        self.plugin = None
        self.plugin_net = None
        self.network = None

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

    def predict(self, image):
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        raise NotImplementedError

    def preprocess_output(self, outputs):
        raise NotImplementedError
