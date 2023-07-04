import pandas as pd
import transformers
from transformers import BertModel, BertTokenizer
import torch
from torch import nn, optim

class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=1):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = out.pooler_output
        out = self.dropout(out)
        out = self.fc(out)
        return out