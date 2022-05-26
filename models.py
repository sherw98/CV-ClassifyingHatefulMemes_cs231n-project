"""
Models class

"""

from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import BertTokenizer, VisualBertModel


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
        
        self.vision_pretrain = torchvision.models.resnet152(pretrained=True)

        # pretrained transformers to get text embeddings
        self.text_model = SentenceTransformer("all-mpnet-base-v2")

        self.fc1 = nn.Linear(1768, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(drop_prob)


    def flatten(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    def forward(self, image, text, device):
        # concat 
        image = self.vision_pretrain(image)
        text = torch.tensor(self.text_model.encode(text)).squeeze().to(device)

        image = self.flatten(F.relu(image))
        text = self.flatten(F.relu(text))
        combined_feat = torch.cat((image, text), dim = 1)

        # forward through linear layers
        fc1_out = self.fc1(combined_feat)
        fc1_out = self.dropout(fc1_out)
        relu_out = self.relu(fc1_out)
        fc2_out = self.fc2(relu_out)

        return F.log_softmax(fc2_out, dim = -1)


class VisualBert_Model(nn.Module):
    """
    Model that utilizes VisualBert

    Args:
        hidden_size,
        drop_prob
    """
    def __init__(self, hidden_size, drop_prob = 0.1):
        super(VisualBert_Model, self).__init__()
        
        self.vbert_model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vision_pretrain = torchvision.models.resnet152(pretrained=True)

        self.fc0 = nn.Linear(1000, 2048)
        self.fc1 = nn.Linear(1768, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(drop_prob)

    def flatten(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    def forward(self, image, text, device):

        # get visual embeddings 
        image = self.fc0(self.vision_pretrain(image))
        visual_token_type_ids = torch.ones(image.shape, dtype=torch.long).to(device)
        visual_attention_mask = torch.ones(image.shape, dtype=torch.float).to(device)

        # tokenize and pad the text
        inputs = self.tokenizer(text, padding = True, return_tensors = "pt").to(device)
        inputs.update(
            {
                "visual_embeds": image,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )

        # get last hidden of vbert
        outputs = self.vbert_model(**inputs)
        last_hidden_state = outputs.last_hidden_state

        # forward through linear layers
        fc1_out = self.fc1(last_hidden_state)
        fc1_out = self.dropout(fc1_out)
        relu_out = self.relu(fc1_out)
        fc2_out = self.fc2(relu_out)

        return F.log_softmax(fc2_out, dim = -1)
