# from sklearn.utils import shuffle
# from transformers import BertTokenizer, BertForSequenceClassification
# from torch.utils.data import DataLoader
# import torch
# import pandas as pd

# df = pd.read_csv('ML\\shuffled_rus_words.csv')
# df = shuffle(df)
# # Load the BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
# model = BertForSequenceClassification.from_pretrained(
#     'DeepPavlov/rubert-base-cased', num_labels=2)

# # Tokenize your texts
# inputs = tokenizer(texts_train, return_tensors='pt',
#                    padding=True, truncation=True)

# # Move your data to the GPU if available
# device = torch.device(
#     'cuda') if torch.cuda.is_available() else torch.device('cpu')
# inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
# labels_train = torch.tensor(labels_train).to(device)

# # Train the model
# model.to(device)
# model.train()
# optimizer = torch.optim.Adam(model.parameters())
# loss_fn = torch.nn.CrossEntropyLoss()

# for epoch in range(10):
#     optimizer.zero_grad()
#     outputs = model(**inputs)
#     loss = loss_fn(outputs.logits, labels_train)
#     loss.backward()
#     optimizer.step()
#     print(f'Epoch {epoch+1}/{10} Loss: {loss.item()}')

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('shuffled_rus_words.csv')

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Preprocess the data


class LaughterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Split the dataset into training and validation sets
df_train, df_val = train_test_split(df, test_size=0.1)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
train_dataset = LaughterDataset(
    df_train['word'], df_train['is_laugh'], tokenizer, max_len=128)
val_dataset = LaughterDataset(
    df_val['word'], df_val['is_laugh'], tokenizer, max_len=128)

# Create a BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased', num_labels=2)

# Define the training function


def train_model(model, data_loader, loss_fn, optimizer, device,
                scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        _, preds = torch.max(outputs[1], dim=1)
        loss = loss_fn(outputs[1], labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


EPOCHS = 10
# Train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

for epoch in range(EPOCHS):
    print(epoch)
    train_acc, train_loss = train_model(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )

# Save the trained model
torch.save(model.state_dict(), 'laughter_classifier_model.pth')

# Define the prediction function


def predict_laughter(text):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased', num_labels=2)
    model.load_state_dict(torch.load('laughter_classifier_model.pth'))
    model = model.to(device)
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _, preds = torch.max(outputs[1], dim=1)
    return 'Laughter' if preds[0].item() == 1 else 'Not laughter'


while 1:
    text = input("Enter a text: ")
    print(predict_laughter(text))
