import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

from preprocess import preprocess_data, AmazonReviewDataset
from model import SentimentClassifier


df_train = pd.read_csv('train.csv', engine='pyarrow', dtype_backend='pyarrow')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_input_ids, train_attention_mask, train_label = preprocess_data(df_train)
train_dataset = AmazonReviewDataset(train_input_ids, train_attention_mask, train_label)
train_dataloader = DataLoader(train_dataset,batch_size=32,num_workers=4,pin_memory=True,shuffle=True)

model = SentimentClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 15

warmup_steps = int(0.1 * num_epochs * len(train_dataloader))
def lr_lambda(current_iteration):
    if current_iteration < warmup_steps:
        lr = current_iteration / warmup_steps
    else:
        lr = max(0.0, float(num_epochs * len(train_dataloader) - current_iteration) / float(max(1, num_epochs * len(train_dataloader) - warmup_steps)))
    return lr

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
train_losses = []
print("Starting training loop...")
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    model.train()
    total_loss = 0 
    total_steps = 0
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)
        output=model(input_ids=input_ids,attention_mask=attention_mask)
        loss=criterion(output.squeeze(), labels.float().squeeze())
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss+=loss.item()
        total_steps+=1
    average_train_loss=total_loss/total_steps
    train_losses.append(average_train_loss)
    print(f'Training Loss: {average_train_loss}')

model_checkpoint = r'result/final_model.pt'
torch.save(model.state_dict(), model_checkpoint)

import matplotlib.pyplot as plt
import os

plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch')
plt.savefig('result/train_losses.png')
plt.close()

print("Training Completed")


