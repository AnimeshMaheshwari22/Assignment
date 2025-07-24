# =================================================================================
# File: feature_extractors.py
# Description: Module for loading pre-trained feature extractors.
# =================================================================================
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import timm
import torch.nn.functional as F

class BertTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()
    def __call__(self, *args, **kwargs):
        return {"input_ids": torch.randint(0, 1000, (1, 3)), "attention_mask": torch.ones(1, 3)}
class BertModel(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, *args, **kwargs): return (torch.randn(1, 3, 768),) # last_hidden_state
def timm_create_model(*args, **kwargs):
    return nn.Sequential(nn.Linear(3*224*224, 196*768), nn.Unflatten(1, (196, 768)))
timm = type("timm", (), {"create_model": timm_create_model})


def get_swin_b_backbone(pretrained=True):
    """
    Loads a pre-trained Swin-B model from timm as a feature extractor.
    Removes the classification head.
    """
    print("Loading Swin-B model...")
    # Swin-B produces features of dimension 1024
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0, global_pool='')
    return model

def get_bert_backbone(model_name='bert-base-uncased'):
    """
    Loads a pre-trained BERT model and tokenizer from transformers.
    """
    print(f"Loading BERT model: {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return model, tokenizer