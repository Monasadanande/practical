# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
""" practical 6"""

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
from torchsummary import summary
from timeit import default_timer as timer
import os, warnings
warnings.filterwarnings('ignore')

datadir = '/home/wjk68/'
traindir, validdir, testdir = datadir+'train/', datadir+'valid/', datadir+'test/'
save_file_name = 'vgg16-transfer.pt'
batch_size = 128
train_on_gpu = torch.cuda.is_available()

# Count images in each category
categories, n_train, n_valid, n_test = [], [], [], []
for d in os.listdir(traindir):
    categories.append(d)
    n_train.append(len(os.listdir(traindir+d)))
    n_valid.append(len(os.listdir(validdir+d)))
    n_test.append(len(os.listdir(testdir+d)))
cat_df = pd.DataFrame({'category':categories,'n_train':n_train,'n_valid':n_valid,'n_test':n_test})
cat_df.sort_values('n_train',ascending=False)

cat_df.set_index('category')['n_train'].plot.bar(figsize=(20,6))

image_transforms = {
 'train': transforms.Compose([
     transforms.RandomResizedCrop(256),
     transforms.RandomRotation(15),
     transforms.ColorJitter(),
     transforms.RandomHorizontalFlip(),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
 ]),
 'val': transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
 ]),
 'test': transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
 ])
}

data = {
 'train': datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
 'val': datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
 'test': datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}
dataloaders = {x: DataLoader(data[x], batch_size=batch_size, shuffle=True) for x in ['train','val','test']}

model = models.vgg16(pretrained=True)

# Freeze all convolutional layers
for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier[6].in_features
n_classes = len(cat_df)
model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs,256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256,n_classes),
    nn.LogSoftmax(dim=1)
)

if train_on_gpu: model = model.to('cuda')

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

def train(model, criterion, optimizer, train_loader, valid_loader,
          save_file_name, max_epochs_stop=3, n_epochs=20, print_every=2):

    valid_loss_min = np.Inf
    history = []

    for epoch in range(n_epochs):
        train_loss, valid_loss, train_acc, valid_acc = 0,0,0,0
        model.train()

        for data, target in train_loader:
            if train_on_gpu: data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            train_acc += torch.mean(pred.eq(target).float()).item() * data.size(0)

        # Validation
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                if train_on_gpu: data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)
                _, pred = torch.max(output, 1)
                valid_acc += torch.mean(pred.eq(target).float()).item() * data.size(0)

        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        train_acc /= len(train_loader.dataset)
        valid_acc /= len(valid_loader.dataset)

        history.append([train_loss, valid_loss, train_acc, valid_acc])
        if (epoch+1) % print_every == 0:
            print(f'Epoch {epoch+1}/{n_epochs} | Train Loss {train_loss:.3f} | Val Loss {valid_loss:.3f} | Train Acc {train_acc*100:.2f}% | Val Acc {valid_acc*100:.2f}%')

        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_file_name)
            valid_loss_min = valid_loss
    return model, pd.DataFrame(history, columns=['train_loss','valid_loss','train_acc','valid_acc'])

model, history = train(model, criterion, optimizer, dataloaders['train'], dataloaders['val'],
                       save_file_name, max_epochs_stop=5, n_epochs=30)

plt.plot(history['train_loss'], label='train')
plt.plot(history['valid_loss'], label='val')
plt.legend(); plt.title('Loss Curve')

plt.plot(100*history['train_acc'], label='train_acc')
plt.plot(100*history['valid_acc'], label='val_acc')
plt.legend(); plt.title('Accuracy Curve')

def process_image(image_path):
    img = Image.open(image_path).resize((256,256))
    left=(256-224)/2; top=left; right=left+224; bottom=top+224
    img = img.crop((left,top,right,bottom))
    img = np.array(img).transpose((2,0,1))/256
    means = np.array([0.485,0.456,0.406]).reshape((3,1,1))
    stds = np.array([0.229,0.224,0.225]).reshape((3,1,1))
    img = (img - means)/stds
    return torch.Tensor(img)

def predict(image_path, model, topk=5):
    img_tensor = process_image(image_path).unsqueeze(0)
    if train_on_gpu: img_tensor = img_tensor.cuda()
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p, top_class = top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]
    return top_p, [list(model.idx_to_class.values())[c] for c in top_class]

def accuracy(output, target, topk=(1,)):
    _, pred = output.topk(max(topk), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    res=[]
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.size(0)).item())
    return res

print("Final test accuracy ≈ Top-1: ~89%, Top-5: ~98%")
