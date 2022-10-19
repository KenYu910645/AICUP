_exp_name = "convnet"
# Import necessary packages.
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
import random
import argparse

CROP_NAME = ['banana', 'carrot', 'corn', 'dragonfruit', 'garlic',
             'guava', 'peanut', 'pineapple', 'pumpkin',
             'rice', 'soybean', 'sugarcane', 'tomato', 'bareland']

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

myseed = 5278  # set a random seed for reproducibility
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)
# NUM_TTA_SAMPLE = 200
# NUM_VALIDE_TTA = 3

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.
# 
# Please refer to PyTorch official website for details about different transforms.

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Shape shift
    transforms.Resize((input_size, input_size)),
    # transforms.RandomChoice([transforms.RandomCrop((450, 450)), transforms.Resize((input_size, input_size))]),
    # transforms.Resize((input_size, input_size)),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomChoice([transforms.RandomRotation(degrees=90), transforms.Resize((input_size, input_size))]),
    
    # Color
    # transforms.ColorJitter(),
    # transforms.RandomGrayscale(0.1),
    # transforms.RandomAutocontrast(0.5),
    # transforms.RandomEqualize(0.5),
    # transforms.RandomChoice([transforms.RandomAdjustSharpness(3, 1.0), transforms.GaussianBlur(31)]),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

class CropDataset(Dataset):
    def __init__(self,path,tfm=test_tfm,files = None):
        super(CropDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label

# There are tons of model in Pytoch lib https://pytorch.org/vision/stable/models.html
class My_Classifier(nn.Module):
    def __init__(self):
        super(My_Classifier, self).__init__()
        # VGG16
        # self.model = models.vgg16_bn(pretrained=False)
        # self.model.classifier[6] = nn.Linear(4096, 11)

        # Resnet101
        # self.model = models.resnet101(pretrained=False)
        # self.model.fc = nn.Linear(2048, 11)

        # Efficient Net
        if config.model_type == 'eff':
            self.model = models.efficientnet_b4(pretrained=True)
            self.model.classifier[1] = nn.Linear(1792, 14)
        elif config.model_type == 'con':
            # ConvNext
            self.model = models.convnext_base(pretrained=True)
            self.model.classifier[2] = nn.Linear(1024, 14)
        elif config.model_type == 'vit':
            self.model = models.vit_b_16(pretrained=True)
            print(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

_dataset_dir = "./data"
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = CropDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = CropDataset(os.path.join(_dataset_dir,"validation"), tfm=train_tfm)
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
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    
    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

