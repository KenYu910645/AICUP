from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.models as models
from collections import defaultdict

# This is for the progress bar.
from tqdm.auto import tqdm
import argparse

CROP_NAME = ['banana', 'carrot', 'corn', 'dragonfruit', 'garlic',
             'guava', 'peanut', 'pineapple', 'pumpkin',
             'rice', 'soybean', 'sugarcane', 'tomato', 'bareland']

EFF_TRAIN_CSV = "/home/lab530/KenYu/aicpu/eff_training.csv"
CON_TRAIN_CSV = "/home/lab530/KenYu/aicpu/con_training.csv"

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--n_epochs', type = int, default = 30)
parser.add_argument('--patience', type = int, default = 100)
parser.add_argument('--learning_rate', type = float, default = 0.0003)
parser.add_argument('--input_size', type = int, default = 224)
parser.add_argument('--ckpt', type = str, default = "")
parser.add_argument('--device', type = str, default = "cuda:0")
parser.add_argument('--model_type', type = str, default = "con")

config = parser.parse_args()
print(config)
batch_size    = config.batch_size
n_epochs      = config.n_epochs
patience      = config.patience
learning_rate = config.learning_rate
input_size    = config.input_size
ckpt_path     = config.ckpt
device        = config.device

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CropDataset(Dataset):
    def __init__(self, path, tfm=None, files = None):
        super(CropDataset).__init__()
        df_eff = pd.read_csv(EFF_TRAIN_CSV)
        df_con = pd.read_csv(CON_TRAIN_CSV)
        self.f_name = df_eff['Id'].tolist()
        eff_f = df_eff['last_layer'].tolist()
        con_f = df_con['last_layer'].tolist()

        feat_list = []
        for i in range(len(eff_f)):
            eff_f[i]
            con_f[i]
            feat_list.append( [float(i) for i in eff_f[i][1:-1].split(',')] +\
                              [float(i) for i in con_f[i][1:-1].split(',')]  )

        self.feat_tensor = torch.tensor(feat_list)

    def __len__(self):
        return len(self.f_name)
  
    def __getitem__(self, idx):
        try:
            label = int(self.f_name[idx].split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return self.feat_tensor[idx], label

class My_Classifier(nn.Module):
    def __init__(self):
        super(My_Classifier, self).__init__()
        self.ensemble = nn.Sequential(
            nn.Linear(28, 14)
        )

    def forward(self, x):
        return self.ensemble(x)

_dataset_dir = "./data"
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = CropDataset(os.path.join(_dataset_dir,"training"), tfm=None)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = CropDataset(os.path.join(_dataset_dir,"validation"), tfm=None)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

model = My_Classifier().to(device)
# load checkpoint 
if ckpt_path != "":
    print(f"Loading Checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):
    model.train()
    train_loss = []
    train_accs = []
    for batch in tqdm(train_loader):
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc  = sum(train_accs) / len(train_accs)

    model.eval()

    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    print(f"Epoch({epoch + 1:03d}/{n_epochs:03d}) train_loss {train_loss:.5f} | train_acc {train_acc:.5f} | valid_loss {valid_loss:.5f} | valid_acc {valid_acc:.5f}")

    # Print the information.
    # print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # update logs
    if valid_acc > best_acc:
        with open("./ensemble_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open("./ensemble_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    
    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), "ensemble_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

