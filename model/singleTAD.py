

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import trange, tqdm
import copy
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score, precision_recall_curve

class MyDataset(TensorDataset):    
    def __init__(self, feature_inter, label):
        assert feature_inter.size(0) == label.size(0) 
        self.feature_inter = feature_inter

        self.label = label

    def __getitem__(self, index): 
        feature_inter = self.feature_inter[index]

        label = self.label[index]

        return (feature_inter, label)

    def __len__(self):
        return self.label.size(0)

    

def pred_model(val_loader, net, loss, device):
    total_loss = 0
    final_score = []
    net.eval()
    total = 0
    correct = 0
    att = []
    for batch_idx, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        
        pred_y,att = net(x)
        
        test_loss = loss(pred_y, y)
        total_loss += test_loss.item()*y.size(0)
        
        pred_y = F.softmax(pred_y, dim=1)
        pred_y = pred_y[:,1]
    
        predictions = (pred_y > 0.5).long()
        
        total += y.size(0)
        correct += (predictions == y).sum().item()
        final_score += pred_y.cpu().detach().numpy().tolist()
    
    net.train()
    final_score = torch.from_numpy(np.array(final_score)).float()
    
    accuracy = correct / total

    return total_loss, final_score, accuracy, att


def eval_results(pred_score, true_label):
    pred_label = (pred_score>0.5).long().cpu().detach().numpy()
    pred_score = pred_score.cpu().detach().numpy()
    true_label = true_label.cpu().detach().numpy()
    
    precision, recall, fmeasure, _ = precision_recall_fscore_support(true_label, pred_label, average='binary')

    AUC = roc_auc_score(true_label, pred_score)
    
    ACC = accuracy_score(true_label, pred_label)
    
    precisions, recalls, thresholds = precision_recall_curve(true_label, pred_score)
    AUPR = auc(recalls, precisions)
    
    return precision, recall, fmeasure, ACC, AUC, AUPR


def extract(inputpath,classLabel):
    chipdata = ['CTCF','DNase','H2AFZ','H3K4me2',
                'H3K4me3','H3K9ac','H3K27ac',
                'H3K36me3', 'H3K79me2','RAD21', 
                'SMC3','POLR2A']
    
    if classLabel == 1:
        flag = 'pos'
    else:
        flag = 'neg'
    
    for chipname in chipdata:
        chipfile = inputpath+chipname+'.'+flag+'.tsv'
        temp_data = pd.read_csv(chipfile, sep='\t', header=None)
        temp_data = temp_data.values
        sample_num = temp_data.shape[0]
        feature_num = temp_data.shape[1]
        break
    x = np.zeros((sample_num, feature_num, len(chipdata)), dtype=float)
    
    if classLabel == 1:
        y = np.ones((sample_num, ), dtype=np.int_)
    else:
        y = np.zeros((sample_num, ), dtype=np.int_)
    
    for idx, chipname in enumerate(chipdata):
        chipfile = inputpath+chipname+'.'+flag+'.tsv'
        temp_data = pd.read_csv(chipfile, sep='\t', header=None)
        temp_data = temp_data.values
        temp_data = temp_data.astype(str)
        temp_data[temp_data=='None'] = '0.0'
        temp_data = temp_data.astype(float)
        x[:,:,idx] = temp_data[:]
    return x, y



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        attention_value = self.sigmoid(out)
        weighted_x = x.mul(attention_value)
        return weighted_x, attention_value  
    

class Net_cob(nn.Module):
    def __init__(self):
        super(Net_cob, self).__init__()
        self.cnn_block = nn.Sequential(nn.Conv1d(in_channels = 12, 
                                                 out_channels = 64, 
                                                 kernel_size = 3,
                                                 padding=1), 
                                       nn.ReLU(), 
                                       nn.Dropout(0.25), 
                                       nn.MaxPool1d(2),
                                       nn.Conv1d(in_channels = 64, 
                                                 out_channels = 64, 
                                                 kernel_size = 3), 
                                       nn.ReLU(),  
                                       nn.MaxPool1d(2))
        self.flat = nn.Flatten()
        self.fc_block = nn.Sequential(nn.Dropout(0.25),
                                      nn.Linear(1536, 128),
                                      nn.ReLU(), 
                                      nn.Dropout(0.5))
        
        self.pred = nn.Linear(128, 2)
        self.channel_attention = ChannelAttention(12)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x,att =self.channel_attention(x)
        x = self.cnn_block(x)
        x = self.flat(x)
        #print("mid:",x.size())
        x = self.fc_block(x)
        x= self.pred(x)
        return x,att   


def main(cell_line, inter_file):
    '''
    load data
    '''
    (x1,y1) = extract(inter_file,1)
    (x2,y2) = extract(inter_file,0)
    x_inter = np.concatenate((x1,x2))
    y_inter = np.concatenate((y1,y2))
#     print(x_inter.shape)
    
    '''
    split train/val/test data
    '''
    batch_size = 128
    epoch = 120

    device = torch.device('cuda:1')

    x_train_inter, x_test_inter, y_train_inter, y_test_inter = train_test_split(x_inter, y_inter, 
                                                                                train_size=0.8, random_state=49)
    x_train_inter, x_val_inter, y_train_inter, y_val_inter = train_test_split(x_train_inter, y_train_inter, 
                                                                              train_size = 0.9, random_state=49)

    
    x_train_inter = torch.from_numpy(x_train_inter).float()
    x_val_inter = torch.from_numpy(x_val_inter).float()
    x_test_inter = torch.from_numpy(x_test_inter).float()
    
    y_train = torch.from_numpy(y_train_inter).long()
    y_val = torch.from_numpy(y_val_inter).long()
    y_test = torch.from_numpy(y_test_inter).long()

    '''
    generate train/val/test dataset
    '''
    train_set = MyDataset(x_train_inter, y_train)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    val_set = MyDataset(x_val_inter, y_val)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    test_set = MyDataset(x_test_inter, y_test)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    '''
    model set
    '''
    net = Net_cob().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = 3e-4) #, weight_decay=1e-6
    # class_weights = torch.tensor([1.0, 2.0]).to(device)
    loss = nn.CrossEntropyLoss()
    # print(net)
    '''
    model train
    '''
    patient = 0
    best_val = 0
    for e in trange(epoch):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):

            x = x.to(device)
            y = y.to(device)

            pred_y, _ = net(x)
            train_loss = loss(pred_y, y)
            total_loss += train_loss.item()*y.size(0)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        total_loss = total_loss/y_train.size(0)
        val_loss, y_pred, val_acc, att = pred_model(val_loader, net, loss, device)
        val_loss = val_loss/y_val.size(0)

        fpr_mlp, tpr_mlp, _ = roc_curve(y_val, y_pred)
        val_auc = auc(fpr_mlp, tpr_mlp)

        if best_val<val_auc:
            patient = 0
            best_val = val_auc
            torch.save({'epoch': e, 'model_state_dict': net.state_dict(),
                        'loss': total_loss, 
                        'optimizer_state_dict': optimizer.state_dict()}, 
                       './best_{0}_model.pt'.format(cell_line))
        else:
            patient += 1

        # print(patient)
        if patient>=8:
            break



    checkpoint = torch.load('./best_{0}_model.pt'.format(cell_line))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    test_loss, y_pred, test_acc, _ = pred_model(test_loader, net, loss, device)
    test_loss = test_loss/y_test.size(0)

    precision, recall, fmeasure, ACC, AUC, AUPR = eval_results(y_pred, y_test)
    
    print('*******' * 3)
    print('\tloss: {:.5f}\n\tPrecision: {:.5f}\n\tRecall: {:.5f}\n\tF-measure: {:.5f}\n\tACC: {:.5f}\n\tAUC: {:.5f}\n\tAUPR: {:.5f}'.format(test_loss, precision, recall, fmeasure, ACC, AUC, AUPR))
    print('*******' * 3)
    
    
    
    return y_pred.cpu().detach().numpy(), y_test.cpu().detach().numpy(), AUC,ACC,AUPR,test_loss,precision,recall,fmeasure
    
    
import sys
import pickle as pkl
if __name__ == '__main__':
    size=[50000,100000,150000,200000,250000,300000]
    cell_lines= ['GM12878','IMR90','Hesc','K562']
    result_table = pd.DataFrame(columns=['Cell Line', 'size','type', 'AUC', 'ACC','AUPR','test_loss','precision','recall','fmeasure'])
    for cell_line in cell_lines:
        for i in range(len(size)):
            for j in range(0,10):
                name=str(size[i]//1000)+"kb"
                inter_file="/input/"
                inter_file+="{0}/".format(cell_line)+"3d5_"+name+"/"
                types=str(j)
                y_pred, y_test,AUC,ACC,AUPR,test_loss,precision,recall,fmeasure = main(cell_line, inter_file)
                new_row = pd.DataFrame({
                    'Cell Line': [cell_line],
                    'size': [name],
                    'type': [types],
                    'AUC': [AUC],
                    'ACC': [ACC],
                    'AUPR': [AUPR],
                    'test_loss': [test_loss],
                    'precision': [precision],
                    'recall': [recall],
                    'fmeasure': [fmeasure],
                })
                result_table = pd.concat([result_table, new_row], ignore_index=True)

    print(result_table)