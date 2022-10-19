# Import necessary packages.
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.models as models
from collections import defaultdict

CROP_NAME = ['banana', 'carrot', 'corn', 'dragonfruit', 'garlic',
             'guava', 'peanut', 'pineapple', 'pumpkin',
             'rice', 'soybean', 'sugarcane', 'tomato', 'bareland']

CKPT = "/home/lab530/KenYu/aicpu/con_best.ckpt"
model_type = 'con' # 'con'
target_dir = "testing"
input_size = 224
batch_size = 64
device = "cuda:1"
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# This transform should be same as train_tfm
test_tfm = transforms.Compose([
    transforms.Resize((input_size, input_size)),
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
        return im, label, fname.split("/")[-1]



class My_Classifier(nn.Module):
    def __init__(self):
        super(My_Classifier, self).__init__()       
        # Efficient Net
        if model_type == 'eff':
            self.model = models.efficientnet_b4(pretrained=True)
            self.model.classifier[1] = nn.Linear(1792, 14)
        elif model_type == 'con':
            # ConvNext
            self.model = models.convnext_base(pretrained=True)
            self.model.classifier[2] = nn.Linear(1024, 14)
        elif model_type == 'vit':
            self.model = models.vit_b_16(pretrained=True)
            print(self.model)
        
    def forward(self, x):
        x = self.model(x)
        return x

###################
### Testing Set ###
###################
_dataset_dir = "./data"
test_set = CropDataset(os.path.join(_dataset_dir, target_dir), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Testing and generate prediction CSV
model_best = My_Classifier().to(device)
model_best.load_state_dict(torch.load(CKPT))
model_best.eval()

file_names = sorted(os.listdir(os.path.join(_dataset_dir, target_dir)))

valid_accs = []
predict_label = []
predict_fn = []
last_layer = []
for batch in test_loader:
    imgs, labels, fn = batch
    with torch.no_grad():
        logits = model_best(imgs.to(device))
    # Calcalate Validation
    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    valid_accs.append(acc)
    # Output predict
    test_label = np.argmax(logits.cpu().data.numpy(), axis=1)
    for i, y_pred in enumerate(test_label.squeeze().tolist()):
        predict_fn.append(fn[i])
        predict_label.append(CROP_NAME[int(y_pred)])
        last_layer.append(logits[i].tolist())
    print(f"{len(predict_fn)}/{len(file_names)}")
    
valid_acc = sum(valid_accs) / len(valid_accs)

# Print the information.
print(f"Validation accuracy = {valid_acc:.5f}")

df = pd.DataFrame()
for i in range(len(predict_fn)):
    df["Id"] = predict_fn
    df["Category"] = predict_label
    df["last_layer"] = last_layer
df.to_csv(f"{model_type}_{target_dir}.csv", index = False)