import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from preprocess import preprocess_data, AmazonReviewDataset
from model import SentimentClassifier

test_df = pd.read_csv('test.csv', engine='pyarrow', dtype_backend='pyarrow')
print(f"Test shape: {test_df.shape}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_input_ids, test_attention_mask, test_label = preprocess_data(test_df)
test_dataset = AmazonReviewDataset(test_input_ids, test_attention_mask, test_label)
test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=False)

model = SentimentClassifier().to(device)
model_checkpoint = 'result/final_model.pt'
model.load_state_dict(torch.load(model_checkpoint))
criterion = nn.BCEWithLogitsLoss()
model.eval()

print("Starting testing loop...")

total_test_loss = 0
total_correct = 0
total_samples = 0
test_losses = []
activation = nn.Sigmoid()

y_true = []
y_pred = []

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
        test_losses.append(test_loss.item())

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred_label.cpu().numpy())

average_test_loss = sum(test_losses) / len(test_losses)
accuracy = total_correct / total_samples


cf = confusion_matrix(y_true, y_pred)
classification_rep = classification_report(y_true, y_pred)

print(f'Testing Loss: {average_test_loss}')
print(f'Total Correct: {total_correct}')
print(f'Total Samples: {total_samples}')
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:',cf)
print('Classification Report:',classification_rep)


import matplotlib.pyplot as plt
import seaborn as sn

plt.figure(figsize=(8, 6))
sn.heatmap(cf, annot=True, cmap='Blues', fmt='.0f', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.text(0.5, 0.2, 'True Positives (TP): {}'.format(cf[0, 0]), ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.text(1.5, 0.2, 'False Positives (FP): {}'.format(cf[0, 1]), ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.text(0.5, 1.2,'False Negatives (FN): {}'.format(cf[1, 0]), ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.text(1.5, 1.2, 'True Negatives (TN): {}'.format(cf[1, 1]), ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.savefig('result/cf.png')
plt.close()


plt.plot(test_losses)
plt.xlabel('Sample')
plt.ylabel('Testing Loss')
plt.title('Test Loss vs. Sample')
plt.savefig('result/test_losses.png')
plt.close()

print("Testing Completed")

