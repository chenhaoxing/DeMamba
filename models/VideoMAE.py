import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch.nn as nn
import torchvision
import time
from .mamba_base import MambaConfig, ResidualBlock

def create_reorder_index(N, device):
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N*N, N))
        else:
            new_order.extend(range(col + N*(N-1), col-1, -N))
    return torch.tensor(new_order, device=device)

def reorder_data(data, N):
    assert isinstance(data, torch.Tensor), "data should be a torch.Tensor"
    device = data.device
    new_order = create_reorder_index(N, device)
    B, t, _, _ = data.shape
    index = new_order.repeat(B, t, 1).unsqueeze(-1)
    reordered_data = torch.gather(data, 2, index.expand_as(data))
    return reordered_data

class Videomae_Net(nn.Module):
    def __init__(
        self, channel_size=512, dropout=0.2, class_num=1
    ):
        super(Videomae_Net, self).__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained("/ossfs/workspace/GenVideo/pretrained_weights/videomae")
        self.fc1 = nn.Linear(768, class_num)
        self.bn1 = nn.BatchNorm1d(768)

        self._init_params()
    
    def _init_params(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x):
        x = self.model.videomae(x)
        sequence_output = x[0]
        print(sequence_output.shape)
        if self.model.fc_norm is not None:
            sequence_output = self.model.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]
        x = self.bn1(sequence_output)
        x = self.fc1(x)
        return x



if __name__ == '__main__':

    model = Videomae_Net()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()


    input_data = torch.randn(1, 16, 3, 224, 224).to(device)

    model(input_data)

