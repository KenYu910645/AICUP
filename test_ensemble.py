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

EFF_TEST_CSV = "/home/lab530/KenYu/aicpu/eff_testing.csv"
CON_TEST_CSV = "/home/lab530/KenYu/aicpu/con_testing.csv"

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--n_epochs', type = int, default = 30)
parser.add_argument('--patience', type = int, default = 100)
parser.add_argument('--learning_rate', type = float, default = 0.0003)
parser.add_argument('--input_size', type = int, default = 224)
parser.add_argument('--ckpt', type = str, default = "/home/lab530/KenYu/aicpu/ensemble_best.ckpt")
parser.add_argument('--device', type = str, default = "cuda:0")

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
        df_eff = pd.read_csv(EFF_TEST_CSV)
        df_con = pd.read_csv(CON_TEST_CSV)
        # 
        self.f_name = df_eff['Id'].tolist()
        eff_f = df_eff['last_layer'].tolist()
        con_f = df_con['last_layer'].tolist()
        # 
        feat_list = []
        for i in range(len(eff_f)):
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
        return self.feat_tensor[idx], label, self.f_name[idx]

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
test_set = CropDataset(os.path.join(_dataset_dir,"testing"), tfm=None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# load checkpoint 
model = My_Classifier().to(device)
model.load_state_dict(torch.load(ckpt_path))
model.eval()

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

valid_accs = []
predict_label = []
predict_fn = []
last_layer = []
for batch in test_loader:
    imgs, labels, fn = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
    # Calcalate Validation
    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    valid_accs.append(acc)
    # Output predict
    test_label = np.argmax(logits.cpu().data.numpy(), axis=1)
    for i, y_pred in enumerate(test_label.squeeze().tolist()):
        predict_fn.append(fn[i])
        predict_label.append(CROP_NAME[int(y_pred)])
        last_layer.append(logits[i].tolist())
    print(f"{len(predict_fn)}/{len(test_loader)}")

valid_acc = sum(valid_accs) / len(valid_accs)

# Print the information.
print(f"Validation accuracy = {valid_acc:.5f}")

df = pd.DataFrame()
for i in range(len(predict_fn)):
    df["image_filename"] = predict_fn
    df["label"] = predict_label
df.to_csv("ensemble_summit.csv", index = False)