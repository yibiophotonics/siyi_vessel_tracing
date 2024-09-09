from __future__ import division
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.utlis import ResEncoder, Decoder, Deconv, DownSample, InitializeWeights
from models.temporalModule import TemporalModule

class InSegNN(nn.Module):
    def __init__(self, out_channels_instance = 12, in_channels = 3, init_features = 32, downsample_time = 4,
                 use_temporal = True, temporal_model = 'improved', time = 5,
                 semantic_path = False):

        """
        :param ins_dimension: the instance embedding dimension.
        :param channels: the channels of the input image.
        """
        super(InSegNN, self).__init__()
        self.downsample_time = downsample_time
        self.use_temporal = use_temporal
        self.temporal_model = temporal_model
        self.time = time
        self.semantic_path = semantic_path

        features = init_features
        self.enc_input = ResEncoder(in_channels, features)
        self.encoder1 = ResEncoder(features, features * 2)
        self.encoder2 = ResEncoder(features * 2, features * 4)
        self.encoder3 = ResEncoder(features * 4, features * 8)

        if self.downsample_time == 4:
            self.encoder4 = ResEncoder(features * 8, features * 16)
            bottleneck_channels = features * 16
        else:
            bottleneck_channels = features * 8

        self.downsample = DownSample()

        if self.use_temporal:
            # temporal_module: torch.tensor(B, T, C, H, W)
            self.temporal_module = TemporalModule(input_features=bottleneck_channels,
                                                  model_name=self.temporal_model, time=self.time)
        if self.downsample_time == 4:
            self.decoder4 = Decoder(features * 16, features * 8)
        self.decoder3 = Decoder(features * 8, features * 4)
        self.decoder2 = Decoder(features * 4, features * 2)
        self.decoder1 = Decoder(features * 2, features)

        if self.downsample_time == 4:
            self.deconv4 = Deconv(features * 16, features * 8)
        self.deconv3 = Deconv(features * 8, features * 4)
        self.deconv2 = Deconv(features * 4, features * 2)
        self.deconv1 = Deconv(features * 2, features)

        self.final_ins = nn.Conv2d(features, out_channels_instance, kernel_size = 1)

        if self.semantic_path:
            self.final_sem = nn.Conv2d(features, 2, kernel_size = 1)

        InitializeWeights(self)

    def forward(self, x):

        b, seq_len, c, h, w = x.shape
        x = x.view(b * seq_len, c, h, w)

        # Do Encoder operations here
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        if self.downsample_time == 4:
            enc3 = self.encoder3(down3)
            down4 = self.downsample(enc3)
            input_feature = self.encoder4(down4)
        else:
            input_feature = self.encoder3(down3)

        # Do Temporal operations here
        if self.use_temporal:
            new_batch, c, h, w = input_feature.shape
            input_feature_t = input_feature.view(b, seq_len, c, h, w)
            temporal = self.temporal_module(input_feature_t)
            temporal = temporal.view(b * seq_len, c, h, w)
            input_feature = temporal

        # Do decoder operations here
        if self.downsample_time == 4:
            up4 = self.deconv4(input_feature)
            up4 = torch.cat((enc3, up4), dim=1)
            dec4 = self.decoder4(up4)
            up3 = self.deconv3(dec4)
        else:
            up3 = self.deconv3(input_feature)

        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        final_ins = self.final_ins(dec1)

        new_batch, c, h, w = final_ins.shape
        final_ins = final_ins.view(b, seq_len, c, h, w)

        if self.semantic_path:
            final_sem = self.final_sem(dec1)
            final_sem = torch.sigmoid(final_sem)
            return final_sem, final_ins

        return final_ins