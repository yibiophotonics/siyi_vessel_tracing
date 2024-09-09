import os
import sys
sys.path.append(os.path.dirname(__file__))
from collections import OrderedDict

from models.InSegNN import InSegNN
from models.unet import UNet
from models.count import CountParameters

class EmbeddingModel():
    def __init__(self, model_type = 'UNet'):
        self.model_type = model_type

    def CreateModel(self, config = None, use_cuda = False):
        if not config['use_temporal']:
            config['time'] = 1
            config['temporal_model'] = 'None'
        if self.model_type == 'UNet':
            self.model = InSegNN( in_channels = config['in_channels'],
                                out_channels_instance = config['out_channels_instance'],
                                init_features = config['init_features'],
                                downsample_time= config['downsample_time'],
                                use_temporal = config['use_temporal'],
                                temporal_model= config['temporal_model'],
                                time = config['time'],
                                semantic_path = config['semantic_path'])

        elif self.model_type == 'InSegNN':
            self.model = InSegNN( in_channels = config['in_channels'],
                                out_channels_instance = config['out_channels_instance'],
                                init_features = config['init_features'],
                                downsample_time= config['downsample_time'],
                                use_temporal = config['use_temporal'],
                                temporal_model= config['temporal_model'],
                                time = config['time'],
                                semantic_path = config['semantic_path'])
        if use_cuda:
            self.model.cuda()

        n_parameters_t = CountParameters(self.model, only_trainable_parameters = True)
        n_parameters = CountParameters(self.model, only_trainable_parameters = False)
        print("Model Parameter:", n_parameters)
        print("Model Trainable Parameter:", n_parameters_t)

        return self.model

    def ForwardModel(self, batch, type = 'train'):

        if type == 'train':
            self.model.train()
            self.model.zero_grad()
        elif type == 'test':
            self.model.eval()
        output = self.model(batch)

        return output
