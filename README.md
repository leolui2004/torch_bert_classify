# A Full Guide of PyTorch to do NLP (Japanese) Classification using BERT
This is a PyTorch full guide of building a NLP model for 2-class classification using BERT. I did a similar one 2 years ago using TensorFlow but then the latest huggingface/transformers did work well with PyTorch so I switched to it and also further added some functions on it to make it a full program. Feel free to provide comments, I just started learning Python last year and I am now concentrating on data analysis, visualization and deep learning applications.

## Simple Introduction
Again this is a NLP classification model to classify the positive and negative reviews (2-class) on some tweets on Twitter. I wrote it in Python using PyTorch and utilized huggingface/transformers with pre-trained tokenizer and also the model cl-tohoku/bert-base-japanese.

## Background
I did a similar project using Tensorflow around 2 years ago with scripts from gathering data to visualization, and you can find the source here: https://github.com/leolui2004/bert_classify_visualize. That one is just like a proof of concept without going deep into the deep learning stuff, so this time I want to focus on PyTorch with not just training but also some advanced functions like validation and testing, learning rate scheduler, checkpoints saving and loading, etc. There are the features that we actually need to use in real life cases and should make our model have higher accuracy.

Note that this is a guide on PyTorch thus I will not explain the concepts of deep learning and I assume that readers may know the basic Python scripts and have some knowledge on deep learning and NLP. I will also use some low level techniques like zero_grad(), loss.backward() which is different from the script I wrote 2 years ago using Tensorflow without calling low level functions but just straightly run the train and fit. And this would give you more flexibility to customize the training loop.

## Process
Here I will skip the data gathering process and go straight into the deep learning model part. Imagine we already have a csv file containing labels and texts and are ready for training.

We need to use these libraries

```
import gc
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import metrics
import transformers
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers.modeling_bert import BertModel
from torch.utils.tensorboard import SummaryWriter
```

And here are some hyperparameters and basic settings we need to define
The first part is the hyperparameters for training that you may want to change for testing efficiency and accuracy.
For MAX_LEN because here we are using tweets on Twitter to learn and Twitter itself have length limit on tweets so it should not be a big problem, but if you want to do a training on paragraphs like posts on Instagram, Facebook, you may think of using a longer length or you just pick the initial part of the texts for training.
Note that as we are going to use a learning rate scheduler, thus the number set here would be the initial learning rate (Epoch 0) for the training. If you set it too high, it may take a long time to show an obvious converge, reversely if you set it too low, it cannot learn effectively start from the beginning.
While train_size is the ratio for cutting out training data from a full dataset, valid_size is the ratio for cutting out validation data from the remaining dataset, thus the below setting will cut the dataset of training : validation : testing into 8 : 1 : 1.

```
MAX_LEN = 200
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-05

tokenizer = transformers.BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
train_size = 0.8
valid_size = 0.5
```

First let's start to import the csv file into a data frame format

```
max_length = 0
train_texts = []
train_labels = []

csv_counter = 0
with open('training_data.csv', 'r', encoding='utf-8') as csv_read:
    csv_reader = csv.reader(csv_read)
    for csv_read_row in csv_reader:
        csv_counter += 1
        if csv_counter > 1:
            train_texts.append(csv_read_row[1])
            train_labels.append(int(csv_read_row[0]))
            if len(csv_read_row[1]) > max_length:
                max_length = len(csv_read_row[1])

data = {'texts':train_texts, 'labels':train_labels}
new_df = pd.DataFrame(data)
```

Then we need to encode the texts using a pre-trained tokenizer and divide them into training, validation, testing dataset, and put it into a DataLoader for PyTorch to load later
If your data is not shuffled before loading into the data frame, you need to do the shuffle here or else you will get 3 super uneven distributed datasets which sometimes you even miss some of the labels in the training dataset. It may be also better to get one to two more times shuffling your data even if you prepared a shuffled csv.

```
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe.texts
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
       texts = str(self.texts[index])
        texts = " ".join(texts.split())

        inputs = self.tokenizer.encode_plus(
            texts,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

sample_df=new_df.sample(frac=1,random_state=200)
train_dataset = sample_df.sample(frac=train_size,random_state=200)
test_df = sample_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

valid_dataset = test_df.sample(frac=valid_size,random_state=200)
test_dataset = test_df.drop(valid_dataset.index).reset_index(drop=True)
valid_dataset = valid_dataset.reset_index(drop=True)

print(f'Dataset: {sample_df.shape} Train: {train_dataset.shape} Validation: {valid_dataset.shape} Test: {test_dataset.shape}')

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
valid_params  = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **valid_params)
testing_loader = DataLoader(testing_set, **test_params)
```

After that it comes to the model building part. For me I am going to add few layers after the last layer of BERT to make the accuracy higher, it would be also a part that you can try different kinds of structure to see which architecture is most effective.
A simple tricks in here would be try to add Dropout layer in-between each layer as to make the model become more generalize
Remember the last layer must have the same output as our task so if we are doing a 2-class classification then the last layer output part of the model must be 2, and do not forget we have to put the model into our device (GPU) if you are going to train the model in GPU

```
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
        self.l2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(768, 384)
        self.l4 = torch.nn.BatchNorm1d(384)
        self.l5 = torch.nn.Linear(384, 96)
        self.l6 = torch.nn.Dropout(0.2)
        self.l7 = torch.nn.Linear(96, 24)
        self.l8 = torch.nn.Dropout(0.2)
        self.l9 = torch.nn.Linear(24, 2)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output_5 = F.relu(self.l5(output_4))
        output_6 = self.l6(output_5)
        output_7 = F.relu(self.l7(output_6))
        output_8 = self.l8(output_7)
        output = self.l9(output_8)
        return output

model = BERTClass()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

We then define the loss function, again it depends on the task you are doing, for multi-class or multi-label classification, you may want to use other loss functions 

```
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
```

Next we set up a learning rate scheduler, the STEP_SIZE is the gamma decay step, this is different from epoch. You can also change the gamma rate, type of optimizer to try out different results.

```
optimizer = torch.optim.AdamW(params= model.parameters(), lr=LEARNING_RATE)
STEP_SIZE = int(TRAIN_SIZE / TRAIN_BATCH_SIZE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.75)
```

Last thing before our training, we want to setup TensorBoard to record and visualize our result

```
def images_to_probs(model, ids, mask, token_type_ids):
    output = model(ids, mask, token_type_ids)
    _, preds_tensor = torch.max(output.cpu(), 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(model, ids, mask, token_type_ids, labels):
    classes = ['Label 1', 'Label 2']
    
    preds, probs = images_to_probs(model, ids, mask, token_type_ids)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(2):
        ax = fig.add_subplot(1, 2, idx+1, xticks=[], yticks=[])
        labels_idx = np.argmax(labels[idx].cpu())
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[int(preds[idx])],
            probs[idx] * 100.0,
            classes[labels_idx]),
                    color=("green" if int(preds[idx])==labels_idx.item() else "red"))
    return fig
```

We now finish all the preparation and we can start the training
I divided the function into training and validation parts, with finishing one epoch, it will print out the loss, current learning rate and validation accuracy, the validation accuracy will help to prove if the model is overfitting or doing a good result on validation dataset.
Remember to switch modes between training and validation as we are just going to validate the model when we are in validation mode
For the first few lines below, that would be initializing TensorBoard and also cleaning up memory and caches

```
%load_ext tensorboard
writer = SummaryWriter('tensorboard')
gc.collect()
torch.cuda.empty_cache()

def train():
    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()
        for _,data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            running_loss += loss.item()
            if _ == len(training_loader) - 1:
                average_loss = running_loss / ( len(training_loader) - 1)
                print(f'Epoch: {epoch + 1}, Loss: {average_loss}, lr: {scheduler.get_last_lr()[0]}')
                writer.add_scalar('training loss', running_loss / 1000, epoch * len(training_loader) + _)
                writer.add_figure('predictions vs. actuals', plot_classes_preds(model, ids, mask, token_type_ids, targets), global_step=epoch * len(training_loader) + _)
                running_loss = 0.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        fin_valid_targets=[]
        fin_valid_outputs=[]
        
        with torch.no_grad():
            for _,valid_data in enumerate(validation_loader, 0):
                valid_ids = valid_data['ids'].to(device, dtype = torch.long)
                valid_mask = valid_data['mask'].to(device, dtype = torch.long)
                valid_token_type_ids = valid_data['token_type_ids'].to(device, dtype = torch.long)
                valid_targets = valid_data['targets'].to(device, dtype = torch.float)

                valid_outputs = model(valid_ids, valid_mask, valid_token_type_ids)
                
                fin_valid_targets.extend(valid_targets.cpu().detach().numpy().tolist())
                fin_valid_outputs.extend(torch.sigmoid(valid_outputs).cpu().detach().numpy().tolist())
        
        arg_valid_outputs = np.array(fin_valid_outputs).argmax(axis=1)[:,None] == range(np.array(fin_valid_outputs).shape[1])
        
        valid_accuracy = metrics.accuracy_score(fin_valid_targets, arg_valid_outputs)
        print(f"Validation Accuracy = {valid_accuracy}")
    
    writer.add_graph(model, (ids, mask, token_type_ids))
    writer.close()

train()
```

After finishing the training, we can see the result on TensorBoard, you may need to set something related to the network if you have difficulties when accessing to the TensorBoard web application

```
tensorboard --port=7000 --logdir=runs
```

Finally we want to use our testing dataset to do a batch test, the techniques is basically same as validation

```
def testing():
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

test_outputs, test_targets = testing()
```

At the end we print out the testing result

```
outputs = np.array(test_outputs).argmax(axis=1)[:,None] == range(np.array(test_outputs).shape[1])

accuracy = metrics.accuracy_score(test_targets, outputs)
f1_score_micro = metrics.f1_score(test_targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(test_targets, outputs, average='macro')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")
```

## Reference
BERT with SentencePiece を日本語 Wikipedia で学習してモデルを公開しました 
https://yoheikikuta.github.io/bert-japanese/
