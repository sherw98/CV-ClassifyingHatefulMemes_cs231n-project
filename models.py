"""
Models class

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Baseline_model(nn.Module):
    """
    Baseline model for the Hateful Memes classification

    Uses pre-trained vision and text embeddings
    from googlenet (torchvision) and fastText respectively

    Args:
        hidden_size,
        drop_prob
    """
    
    def __init__(self, hidden_size, drop_prob = 0.1):
        super(Baseline_model, self).__init__()
        
        self.vision_pretrain = torchvision.models.googlenet(pretrained=True)

        self.fc1 = nn.Linear(1300, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(drop_prob)


    def flatten(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    def forward(self, image, text):

        # concat 
        image = self.vision_pretrain(image)

        image = self.flatten(F.relu(image))
        text = self.flatten(F.relu(text))
        combined_feat = torch.cat((image, text), dim = 1)

        # forward through linear layers
        fc1_out = self.fc1(combined_feat)
        fc1_out = self.dropout(fc1_out)
        relu_out = self.relu(fc1_out)
        fc2_out = self.fc2(relu_out)

        return F.log_softmax(fc2_out, dim = -1)


