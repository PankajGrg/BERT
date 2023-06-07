import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertModel, BertTokenizer
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

print("Loading the data...")

train_df = pd.read_csv('train_df.csv', engine='pyarrow', dtype_backend='pyarrow')
df_label_0 = train_df[train_df['label'] == 0]
df_label_1 = train_df[train_df['label'] == 1]
df_label_0_subset = df_label_0.sample(n=5000, random_state=42)
df_label_1_subset = df_label_1.sample(n=5000, random_state=42)
df_train = pd.concat([df_label_0_subset, df_label_1_subset])
print(f"Train shape: {df_train.shape}")

test_df = pd.read_csv('test_df.csv', engine='pyarrow', dtype_backend='pyarrow')
df_label_0 = test_df[test_df['label'] == 0]
df_label_1 = test_df[test_df['label'] == 1]
df_label_0_subset = df_label_0.sample(n=1000, random_state=42)
df_label_1_subset = df_label_1.sample(n=1000, random_state=42)
test_df = pd.concat([df_label_0_subset, df_label_1_subset])
df_val, df_test = train_test_split(test_df, test_size=0.5, random_state=42)
print(f"Validation shape: {df_val.shape}")
print(f"Test shape: {df_test.shape}")

print("Initializing the tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
    attention_masks = encoded_reviews['attention_mask']
    labels = torch.tensor(df['label'].values)
    return input_ids, attention_masks, labels

train_input_ids, train_attention_mask, train_label = preprocess_data(df_train)
test_input_ids, test_attention_mask, test_label = preprocess_data(df_test)
val_input_ids, val_attention_mask, val_label = preprocess_data(df_val)

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

train_dataset = AmazonReviewDataset(train_input_ids, train_attention_mask, train_label)
test_dataset = AmazonReviewDataset(test_input_ids,test_attention_mask,test_label)
val_dataset = AmazonReviewDataset(val_input_ids,val_attention_mask,val_label)

print("Creating data loaders...")
batch_size = 32
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=4,pin_memory=True)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,num_workers=4,pin_memory=True)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,num_workers=4,pin_memory=True)

print("Checking if GPU is available...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device: ",device)

print("Initializing model...")
model = SentimentClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.BCEWithLogitsLoss()
num_epochs = 20

warmup_steps = int(0.1 * num_epochs)
def lr_lambda(current_iteration):
    if current_iteration < warmup_steps:
        return 2e-5 * current_iteration / warmup_steps
    else:
        return 2e-5 * (1 - (current_iteration - warmup_steps) / (num_epochs * len(train_dataloader) - warmup_steps))
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

train_losses = []
val_losses = []

print("Starting training loop...")
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    model.train()
    total_loss = 0 
    total_steps = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)
        output=model(input_ids=input_ids,attention_mask=attention_mask)
        output = output.float().squeeze()
        labels = labels.float().squeeze()
        loss=criterion(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss+=loss.item()
        total_steps+=1
    average_train_loss=total_loss/total_steps
    train_losses.append(average_train_loss)
    print(f'Training Loss: {average_train_loss}')
 
    # Validation 
    model.eval()
    total_val_loss=0
    total_val_steps=0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)
            output=model(input_ids=input_ids,attention_mask=attention_mask)
            output = output.float().squeeze()
            labels = labels.float().squeeze()
            val_loss=criterion(output, labels)
            total_val_loss+=val_loss.item()
            total_val_steps+=1
        average_val_loss=total_val_loss/total_val_steps
        val_losses.append(average_val_loss)
        print(f'Validation Loss: {average_val_loss}')


print("Starting testing loop...")
model.eval()
total_test_loss = 0
total_test_steps = 0
total_correct = 0
total_samples = 0
activation = nn.Sigmoid()
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        output = output.float().squeeze()
        labels = labels.float().squeeze()
        test_loss = criterion(output, labels)
        pred_label = torch.round(activation(output))
        total_correct += (pred_label == labels).sum().item()
        total_samples += labels.size(0)
        total_test_loss += test_loss.item()
        total_test_steps += 1

average_test_loss = total_test_loss / total_test_steps
accuracy = total_correct / total_samples
print(f'total_correct: {total_correct}')
print(f'total_samples: {total_samples}')
print(f'Testing Loss: {average_test_loss}')
print(f'Accuracy: {accuracy}')


