import torch.nn as nn
from models.count import CountParameters

class TemporalModule(nn.Module):
    def __init__(self, input_features = 128, model_name = 'baseline', time = 5):
        super().__init__()
        self.model_name = model_name
        self.model = TemporalModel(model_name = model_name,
                                                      input_features = input_features,
                                                      time = time).get_model()
        n_parameters_t = CountParameters(self.model, only_trainable_parameters = True)
        n_parameters = CountParameters(self.model, only_trainable_parameters = False)
        print("Temporal Module Parameter:", n_parameters)
        print("Temporal Module Trainable Parameter:", n_parameters_t)

    def forward(self, x):
        # from batch * time * channel * height * width
        # to batch * channel * time * height * width
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x


class TemporalModel:
    def __init__(self, model_name, input_features, time):
        self.model_name = model_name
        self.input_features = input_features
        self.time = time

    def get_model(self):
        input_features = self.input_features

        if self.model_name == 'no_temporal':
            return nn.Sequential()

        elif self.model_name == 'baseline':
            return nn.Sequential(CausalConv3d(input_features, input_features, (2, 3, 3), dilation = (1, 1, 1)),
                                 NormActivation(input_features, dimension = '3d', activation = 'leaky_relu'),
                                 CausalConv3d(input_features, input_features, (1, 3, 3), dilation = (1, 1, 1)),
                                 NormActivation(input_features, dimension = '3d', activation = 'leaky_relu'),
                                 CausalConv3d(input_features, input_features, (1, 3, 3), dilation = (1, 1, 1)),
                                 NormActivation(input_features, dimension = '3d', activation = 'leaky_relu'),
                                 )

        elif self.model_name == 'improved':
            model = []
            for i in range((self.time - 1) // 2):
                model.append(CausalConv3d(input_features, input_features, (3, 3, 3), dilation = (1, 1, 1)))
                model.append(NormActivation(input_features, dimension = '3d', activation = 'leaky_relu'))
                model.append(CausalConv3d(input_features, input_features, (1, 3, 3), dilation = (1, 1, 1)))
                model.append(NormActivation(input_features, dimension = '3d', activation='leaky_relu'))
                model.append(CausalConv3d(input_features, input_features, (1, 3, 3), dilation = (1, 1, 1)))
                model.append(NormActivation(input_features, dimension = '3d', activation = 'leaky_relu'))

            return nn.Sequential(*model)

        else:
            return nn.Sequential()


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = (2, 3, 3), dilation = (1, 1, 1), bias = False):
        super().__init__()
        # kernel_size must be a 3-tuple
        time_pad = (kernel_size[0] - 1) * dilation[0]
        height_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        width_pad = ((kernel_size[2] - 1) * dilation[2]) // 2

        # Pad temporally on the left (left is past)
        self.pad = nn.ConstantPad3d(padding=(width_pad, width_pad, height_pad, height_pad, time_pad, 0), value = 0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, dilation = dilation, stride = 1, padding = 0, bias = bias)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)

        return x

class NormActivation(nn.Module):
    def __init__(self, num_features, dimension='2d', activation='none', momentum=0.05, slope=0.01):
        super().__init__()

        if dimension == '1d':
            self.norm = nn.BatchNorm1d(num_features=num_features, momentum=momentum)
        elif dimension =='2d':
            self.norm = nn.BatchNorm2d(num_features=num_features, momentum=momentum)
        elif dimension == '3d':
            self.norm = nn.BatchNorm3d(num_features=num_features, momentum=momentum)

        if activation == "relu":
            self.activation_fn = lambda x: nn.functional.relu(x, inplace=True)
        elif activation == "leaky_relu":
            self.activation_fn = lambda x: nn.functional.leaky_relu(x, negative_slope=slope, inplace=True)
        elif activation == "elu":
            self.activation_fn = lambda x: nn.functional.elu(x, inplace=True)
        elif activation == "none":
            self.activation_fn = lambda x: x

    def forward(self, x):
        x = self.norm(x)
        x = self.activation_fn(x)
        return x