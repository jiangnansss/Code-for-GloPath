import time
import pandas as pd
import os
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from model import *
from test_utils import *
import sys
notebook_path = os.getcwd()
sys.path.append(os.path.join(os.path.dirname(notebook_path), '0-Models_and_weights', '0-GloPath'))
from glopath import loadGloPath, validTrans

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse():
    parser = argparse.ArgumentParser(description='GloPath')
    parser.add_argument('--batchSize', type=int, default=8)
    parser.add_argument('--numWorkers', type=int, default=16)
    parser.add_argument('--nepoches', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--logName', type=str, default='')
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--lesionTarget', type=int, default=None, nargs='*')
    parser.add_argument('expName', type=str)
    parser.add_argument('device', type=str, default='cuda:1')
    parser.add_argument('staining', type=str)
    return parser.parse_args()

class SimpleDataset(data.Dataset):
    def __init__(self, path, transform=None, isTrain=True, args=None):
        self.args = args
        self.isTrain = isTrain
        self.transform = transform
        
        if not isTrain:
            self.path = path
        else:
            self.pos, self.neg = [], []
            for name in path:
                label = np.load(name.replace('.png', '.npy'))
                flag = any(label[target] >= 50 for target in args.lesionTarget)
                (self.pos if flag else self.neg).append(name)
            self.randomSelect()

    def randomSelect(self):
        min_len = min(len(self.pos), len(self.neg))
        self.path = self.neg[:min_len] + random.sample(self.pos, min_len) if len(self.pos) > len(self.neg) else self.pos[:min_len] + random.sample(self.neg, min_len)

    def getLabel(self, filename):
        label = np.load(filename.replace('.png', '.npy'))
        return int(any(label[target] >= 50 for target in self.args.lesionTarget))

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        imgPath = self.path[idx]
        label = self.getLabel(imgPath)
        return self.transform(image=cv2.imread(imgPath))['image'], label

def train(loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for input, target in tqdm(loader):
        input, target = input.cuda(device), target.cuda(device)
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input.size(0)
    return running_loss / len(loader.dataset)

def valid(loader, model, criterion, device):
    model.eval()
    running_loss = 0
    targetList, probList = [], []
    
    with torch.no_grad():
        for input, target in tqdm(loader):
            input, target = input.cuda(device), target.cuda(device)
            output = model(input)
            loss = criterion(output, target)
            running_loss += loss.item() * input.size(0)
            prob = nn.Softmax(1)(output)[:, 1]
            
            targetList.append(target.cpu().numpy())
            probList.append(prob.cpu().numpy())
    
    targetList = np.concatenate(targetList)
    probList = np.concatenate(probList)
    predList = (probList > 0.5).astype(int)
    
    precision, recall, _ = precision_recall_curve(targetList, predList)
    pr_auc = auc(recall, precision)
    
    return running_loss / len(loader.dataset), pr_auc, [targetList, probList]

if __name__ == '__main__':
    args = parse()
    
    trainTrans = A.Compose([
        Resize(224, 224),
        Flip(p=1),
        RandomRotate90(p=1),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=45, p=0.5),
        OneOf([CLAHE(clip_limit=2), RandomBrightnessContrast()], p=1),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1)
    ], p=1)
    
    validTrans = A.Compose([
        Resize(224, 224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1)
    ], p=1)

    dataCSV = pd.read_csv(f'/data_sdd/hqm/xj/lesion_classificationV2/CLAM-master1/splits/{args.staining}/splits_0.csv')
    folder = f'/data_sdd/hqm/xj/lesion_classificationV2/data/{args.staining}/images'
    
    trainNames = [os.path.join(folder, item.replace('.pt', '.png')) for item in dataCSV['train'].tolist()]
    valNames = [os.path.join(folder, item.replace('.pt', '.png')) for item in dataCSV['val'].dropna().tolist()]
    testNames = [os.path.join(folder, item.replace('.pt', '.png')) for item in dataCSV['test'].dropna().tolist()]

    trainLoader = data.DataLoader(SimpleDataset(trainNames, trainTrans, args=args), batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers)
    validLoader = data.DataLoader(SimpleDataset(valNames, validTrans, isTrain=False, args=args), batch_size=args.batchSize, shuffle=False, num_workers=args.numWorkers)
    testLoader = data.DataLoader(SimpleDataset(testNames, validTrans, isTrain=False, args=args), batch_size=args.batchSize, shuffle=False, num_workers=args.numWorkers)

    model = loadGloPath(numClasses=args.n_classes).cuda(args.device)
    
    criterion = nn.CrossEntropyLoss().cuda(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_auc = 0
    patience = 0
    
    for epoch in range(args.nepoches):
        t = time.time()
        print(f'---------- Epoch {epoch}_{args.staining}_{args.lesionTarget} ----------')
        
        trainLoss = train(trainLoader, model, criterion, optimizer, args.device)
        trainLoader.dataset.randomSelect()
        validLoss, valAUC, _ = valid(validLoader, model, criterion, args.device)
        
        print(f'Train Loss: {trainLoss:.4f} | Valid Loss: {validLoss:.4f} | Time: {time.time()-t:.2f}s')
        
        if valAUC >= best_auc:
            best_auc = valAUC
            patience = 0
            _, testAUC, testROC = valid(testLoader, model, criterion, args.device)
            cm = confusion_matrix(testROC[0], testROC[1] > 0.5)
            f1 = f1_score(testROC[0], testROC[1] > 0.5, average='weighted')
            auc_score = roc_auc_score(testROC[0], testROC[1])
            print(f'CM:\n{cm}\nF1: {f1:.4f} | AUC: {auc_score:.4f} | Test AUC: {testAUC:.4f}')
            
            torch.save({
                'modelT': model.state_dict(),
                'valAUC': valAUC,
                'testAUC': testAUC,
                'testROC': testROC,
                'arg': args,
            }, f'./checkpoints_clsV5_linear/vit_dino_100w_{args.staining}_{args.lesionTarget}_{args.expName}.pth')
        else:
            patience += 1
            if patience >= 20:
                print('Early Stopping.')
                break
        
        print(f'Val AUC: {valAUC:.4f} | Best: {best_auc:.4f}')