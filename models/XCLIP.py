from transformers import XCLIPVisionModel
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from transformers import XCLIPVisionModel
class XCLIP(nn.Module):
    def __init__(
        self, channel_size=512, dropout=0.2, class_num=1
    ):
        super(XCLIP, self).__init__()
      
        self.backbone = XCLIPVisionModel.from_pretrained("GenVideo/pretrained_weights/xclip")
        self.fc_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, 1)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.backbone(images, output_hidden_states=True)
        sequence_output = outputs['pooler_output'].reshape(b, t, -1)
        video_level_features = self.fc_norm(sequence_output.mean(1))
        pred = self.head(video_level_features)

        return pred


