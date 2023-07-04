import pandas as pd
import transformers
from transformers import BertModel, BertTokenizer
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def preprocess_data(df):
    print("Preprocessing data...")
    encoded_reviews = tokenizer(
        df['text'].tolist(),
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_reviews['input_ids']
    print(input_ids.shape)
    attention_masks = encoded_reviews['attention_mask']
    print(attention_masks.shape)
    labels = torch.tensor(df['label'].values)
    print(labels.shape)
    return input_ids, attention_masks, labels


class AmazonReviewDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }