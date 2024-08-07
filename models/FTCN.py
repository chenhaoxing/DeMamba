'''
Exploring Temporal Coherence for More General Video Face Forgery Detection @ ICCV'2021
Copyright (c) Xiamen University and its affiliates.
Modified by Yinglin Zheng from https://github.com/yinglinzheng/FTCN
'''

import torch
from torch import nn
from .time_transformer import TimeTransformer
from .clip import clip

class RandomPatchPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        if self.training and my_cfg.model.transformer.random_select:
            while True:
                idx = random.randint(0, h * w - 1)
                i = idx // h
                j = idx % h
                if j == 0 or i == h - 1 or j == h - 1:
                    continue
                else:
                    break
        else:
            idx = h * w // 2
        x = x[..., idx]
        return x


def valid_idx(idx, h):
    i = idx // h
    j = idx % h
    if j == 0 or i == h - 1 or j == h - 1:
        return False
    else:
        return True


class RandomAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        candidates = list(range(h * w))
        candidates = [idx for idx in candidates if valid_idx(idx, h)]
        max_k = len(candidates)
        if self.training and my_cfg.model.transformer.random_select:
            k = my_cfg.model.transformer.k
        else:
            k = max_k
        candidates = random.sample(candidates, k)
        x = x[..., candidates].mean(-1)
        return x

class TransformerHead(nn.Module):
    def __init__(self, spatial_size=7, time_size=8, in_channels=2048):
        super().__init__()
        # if my_cfg.model.inco.no_time_pool:
        #     time_size = time_size * 2
        patch_type = 'time'
        if patch_type == "time":
            self.pool = nn.AvgPool3d((1, spatial_size, spatial_size))
            self.num_patches = time_size
        elif patch_type == "spatial":
            self.pool = nn.AvgPool3d((time_size, 1, 1))
            self.num_patches = spatial_size ** 2
        elif patch_type == "random":
            self.pool = RandomPatchPool()
            self.num_patches = time_size
        elif patch_type == "random_avg":
            self.pool = RandomAvgPool()
            self.num_patches = time_size
        elif patch_type == "all":
            self.pool = nn.Identity()
            self.num_patches = time_size * spatial_size * spatial_size
        else:
            raise NotImplementedError(patch_type)

        self.dim = -1
        if self.dim == -1:
            self.dim = in_channels

        self.in_channels = in_channels

        if self.dim != self.in_channels:
            self.fc = nn.Linear(self.in_channels, self.dim)

        default_params = dict(
            dim=self.dim, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
        )
        
        self.time_T = TimeTransformer(
            num_patches=self.num_patches, num_classes=1, **default_params
        )


    def forward(self, x):
        x = self.pool(x)
        x = x.reshape(-1, self.in_channels, self.num_patches)
        x = x.permute(0, 2, 1)
        if self.dim != self.in_channels:
            x = self.fc(x.reshape(-1, self.in_channels))
            x = x.reshape(-1, self.num_patches, self.dim)
        x = self.time_T(x)

        return x


class ViT_B_FTCN(nn.Module):
    def __init__(
        self, channel_size=512, class_num=1
    ):
        super(ViT_B_FTCN, self).__init__()
        self.clip_model, preprocess = clip.load('ViT-B-16')
        self.clip_model = self.clip_model.float()

        self.head = TransformerHead(spatial_size=14, time_size=8, in_channels=512)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        sequence_output = self.clip_model.encode_image(images)
        _, _, c = sequence_output.shape
        sequence_output = sequence_output.view(b, t, 14, 14, c)
        sequence_output = sequence_output.permute(0, 4, 1, 2, 3)

        res = self.head(sequence_output)

        return res
if __name__ == '__main__':
    model = ViT_B_FTCN()
    model = model.cuda()
    dummy_input = torch.randn(4,8,3,224,224)
    dummy_input = dummy_input.cuda()
    model(dummy_input)
