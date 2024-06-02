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
    def __init__(self, feature_inter, feature_nb, feature_mid, label):
        assert feature_inter.size(0) == label.size(0)
        assert feature_nb.size(0) == label.size(0)
        assert feature_mid.size(0) == label.size(0)
        
        self.feature_inter = feature_inter
        self.feature_nb = feature_nb
        self.feature_mid = feature_mid
        self.label = label

    def __getitem__(self, index): 
        feature_inter = self.feature_inter[index]
        feature_nb = self.feature_nb[index]
        feature_mid = self.feature_mid[index]
        label = self.label[index]

        return (feature_inter, feature_nb, feature_mid, label)

    def __len__(self):
        return self.label.size(0)

    

def pred_model(val_loader, net_inter, net_nb, net_mid, loss, device):
    total_loss = 0
    final_score = []
    net_inter.eval()
    net_nb.eval()
    net_mid.eval()
    total = 0
    correct = 0
    for batch_idx, (x_inter, x_nb, x_mid, y) in enumerate(val_loader):
        x_inter = x_inter.to(device)
        x_nb = x_nb.to(device)
        x_mid = x_mid.to(device)
        y = y.to(device)
        

        pred_y_inter = net_inter(x_inter)
        pred_y_nb = net_nb(x_nb)
        pred_y_mid = net_mid(x_mid)
        

        test_loss_inter = loss(pred_y_inter, y)
        test_loss_nb = loss(pred_y_nb, y)
        test_loss_mid = loss(pred_y_mid, y)
        

        total_test_loss = test_loss_inter + test_loss_nb + test_loss_mid
        total_loss += total_test_loss.item() * y.size(0)

        pred_y_inter = F.softmax(pred_y_inter, dim=1)
        pred_y_nb = F.softmax(pred_y_nb, dim=1)
        pred_y_mid = F.softmax(pred_y_mid, dim=1)
        

        pred_y = (pred_y_inter + pred_y_nb + pred_y_mid) / 3
        pred_y = pred_y[:, 1]
        

        predictions = (pred_y > 0.5).long()
        total += y.size(0)
        correct += (predictions == y).sum().item()
        
        final_score += pred_y.cpu().detach().numpy().tolist()
    

    net_inter.train()
    net_nb.train()
    net_mid.train()
    
    final_score = torch.from_numpy(np.array(final_score)).float()
    accuracy = correct / total

    return total_loss, final_score, accuracy

   


def eval_results(pred_score, true_label):
    pred_label = (pred_score>0.5).long().cpu().detach().numpy()
    pred_score = pred_score.cpu().detach().numpy()
    true_label = true_label.cpu().detach().numpy()
    
    precision, recall, fmeasure, _ = precision_recall_fscore_support(true_label, pred_label, average='binary')
    
    # fpr_mlp, tpr_mlp, _ = roc_curve(true_label, pred_score)
    # AUC = auc(fpr_mlp, tpr_mlp)
    AUC = roc_auc_score(true_label, pred_score)
    
    ACC = accuracy_score(true_label, pred_label)
    
    precisions, recalls, thresholds = precision_recall_curve(true_label, pred_score)
    AUPR = auc(recalls, precisions)
    
    return precision, recall, fmeasure, ACC, AUC, AUPR


def extract(inputpath,classLabel):
    chipdata = ['CTCF''RAD21', 'SMC3']
#     chipdata = ['DNase']
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

class Net_inter(nn.Module):
    def __init__(self):
        super(Net_inter, self).__init__()
        self.cnn_block = nn.Sequential(nn.Conv1d(in_channels = 3,
                                                 out_channels = 64, 
                                                 kernel_size = 3,padding=1), 
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
        self.channel_attention = ChannelAttention(3)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x,att =self.channel_attention(x)
        x = self.cnn_block(x)
        x = self.flat(x)
        x = self.fc_block(x)
        x= self.pred(x)
        return x
    
class Net_mid(nn.Module):
    def __init__(self):
        super(Net_mid, self).__init__()
        self.cnn_block = nn.Sequential(nn.Conv1d(in_channels = 3,
                                                 out_channels = 64, 
                                                 kernel_size = 3,padding=1), 
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
        self.channel_attention = ChannelAttention(3)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x,att =self.channel_attention(x)
        x = self.cnn_block(x)
        x = self.flat(x)
        x = self.fc_block(x)
        x= self.pred(x)
        return x

class Net_nb(nn.Module):
    def __init__(self):
        super(Net_nb, self).__init__()
        self.cnn_block = nn.Sequential(nn.Conv1d(in_channels = 3,
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
        self.channel_attention = ChannelAttention(3)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x,att =self.channel_attention(x)
        x = self.cnn_block(x)
        x = self.flat(x)
        x = self.fc_block(x)
        x= self.pred(x)
        return x    

    

def main(cell_line, inter_file, mid_file, nb_file):
    '''
    load data
    '''
    (x1,y1) = extract(inter_file,1)
    (x2,y2) = extract(inter_file,0)
    x_inter = np.concatenate((x1,x2))
    y_inter = np.concatenate((y1,y2))
    
    (x1,y1) = extract(mid_file,1)
    (x2,y2) = extract(mid_file,0)
    x_mid = np.concatenate((x1,x2))
    y_mid = np.concatenate((y1,y2))
    
    (x1,y1) = extract(nb_file,1)
    (x2,y2) = extract(nb_file,0)
    x_nb = np.concatenate((x1,x2))
    y_nb = np.concatenate((y1,y2))
    
    '''
    split train/val/test data
    '''
    height = x_nb.shape[1]
    width = x_nb.shape[2]
    batch_size = 128
    epoch = 120
    device = torch.device('cuda:1')

    x_train_inter, x_test_inter, y_train_inter, y_test_inter = train_test_split(x_inter, y_inter, 
                                                                                train_size=0.8, random_state=49)
    x_train_inter, x_val_inter, y_train_inter, y_val_inter = train_test_split(x_train_inter, y_train_inter, 
                                                                              train_size = 0.9, random_state=49)

    x_train_nb, x_test_nb, y_train_nb, y_test_nb = train_test_split(x_nb, y_nb, train_size=0.8, random_state=49)
    x_train_nb, x_val_nb, y_train_nb, y_val_nb = train_test_split(x_train_nb, y_train_nb, train_size = 0.9, random_state=49)

    x_train_mid, x_test_mid, y_train_mid, y_test_mid = train_test_split(x_mid, y_mid, train_size=0.8, random_state=49)
    x_train_mid, x_val_mid, y_train_mid, y_val_mid = train_test_split(x_train_mid, y_train_mid, train_size = 0.9, random_state=49)
    
    x_train_inter = torch.from_numpy(x_train_inter).float()
    x_val_inter = torch.from_numpy(x_val_inter).float()
    x_test_inter = torch.from_numpy(x_test_inter).float()

    x_train_mid = torch.from_numpy(x_train_mid).float()
    x_val_mid = torch.from_numpy(x_val_mid).float()
    x_test_mid = torch.from_numpy(x_test_mid).float()

    x_train_nb = torch.from_numpy(x_train_nb).float()
    y_train = torch.from_numpy(y_train_nb).long()

    x_val_nb = torch.from_numpy(x_val_nb).float()
    y_val = torch.from_numpy(y_val_nb).long()

    x_test_nb = torch.from_numpy(x_test_nb).float()
    y_test = torch.from_numpy(y_test_nb).long()
    '''
    generate train/val/test dataset
    '''
    train_set = MyDataset(x_train_inter, x_train_nb, x_train_mid, y_train)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    val_set = MyDataset(x_val_inter, x_val_nb, x_val_mid, y_val)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    test_set = MyDataset(x_test_inter, x_test_nb, x_test_mid, y_test)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    '''
    model set
    '''
    net_inter=Net_inter().to(device)
    net_nb=Net_nb().to(device)
    net_mid=Net_mid().to(device)
    
    optimizer = torch.optim.Adam(
        list(net_inter.parameters()) + list(net_nb.parameters()) + list(net_mid.parameters()),
        lr=3e-4
    )

    loss = nn.CrossEntropyLoss()
    
    '''
    model train
    '''
    patient = 0
    best_val = 0
    accumulated_loss = 0
    for e in trange(epoch):
        total_loss = 0
        for batch_idx, (x_inter, x_nb, x_mid, y) in enumerate(train_loader):
            x_inter = x_inter.to(device)
            x_nb = x_nb.to(device)
            x_mid = x_mid.to(device)
            y = y.to(device)

            pred_y_inter = net_inter(x_inter)
            pred_y_nb = net_nb(x_nb)
            pred_y_mid = net_mid(x_mid)
        
            train_loss_inter =loss(pred_y_inter, y)
            train_loss_nb = loss(pred_y_nb, y)
            train_loss_mid = loss(pred_y_mid, y)
            accumulated_loss += (train_loss_inter.item() + train_loss_nb.item() + train_loss_mid.item()) * y.size(0)
            total_loss = accumulated_loss / len(train_loader.dataset)
            accumulated_loss = 0 
            optimizer.zero_grad()
            train_loss_inter.backward()
            train_loss_nb.backward()
            train_loss_mid.backward()
            optimizer.step()

        total_loss = total_loss/y_train.size(0)
        val_loss, y_pred, val_acc = pred_model(val_loader,net_inter, net_nb, net_mid, loss, device)
        val_loss = val_loss/y_val.size(0)

        fpr_mlp, tpr_mlp, _ = roc_curve(y_val, y_pred)
        val_auc = auc(fpr_mlp, tpr_mlp)

        if best_val<val_auc:
            patient = 0
            best_val = val_auc
            torch.save({
    'epoch': e,
    'model1_state_dict': net_inter.state_dict(),
    'model2_state_dict': net_nb.state_dict(),
    'model3_state_dict': net_mid.state_dict(),
    'loss': total_loss, 
    'optimizer_state_dict': optimizer.state_dict()
}, './best_{0}_models.pt'.format(cell_line))
            
        else:
            patient += 1

        if patient>=8:
            break

    checkpoint = torch.load('./best_{0}_models.pt'.format(cell_line))
    net_inter.load_state_dict(checkpoint['model1_state_dict'])
    net_nb.load_state_dict(checkpoint['model2_state_dict'])
    net_mid.load_state_dict(checkpoint['model3_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    test_loss, y_pred, test_acc = pred_model(test_loader,net_inter, net_nb, net_mid, loss, device)
    test_loss = test_loss/y_test.size(0)

    precision, recall, fmeasure, ACC, AUC, AUPR = eval_results(y_pred, y_test)

    print('*******' * 3)
    print('\tloss: {:.5f}\n\tPrecision: {:.5f}\n\tRecall: {:.5f}\n\tF-measure: {:.5f}\n\tACC: {:.5f}\n\tAUC: {:.5f}\n\tAUPR: {:.5f}'.format(test_loss, precision, recall, fmeasure, ACC, AUC, AUPR))
    print('*******' * 3)
    return y_pred.cpu().detach().numpy(), y_test.cpu().detach().numpy(),AUC,ACC,AUPR,test_loss,precision,recall,fmeasure
    
import sys
import pickle as pkl
if __name__ == '__main__': 
    print("开始运行！")
    size1=[200]
    size2=[250]
    size3=[300]
    cell_lines=["GM12878","IMR90","K562","Hesc"]
    result_table = pd.DataFrame(columns=['Cell Line', 'size','type', 'AUC', 'ACC','AUPR','test_loss','precision','recall','fmeasure'])
    for cell_line in cell_lines:
        for j in range(0,10):
            name=str(size1[0])+"kb"
            names=name
            file_dir="/input/"
            inter_file=file_dir+"{0}/".format(cell_line)+"3d5_"+name+"/"
            name=str(size2[0])+"kb"
            names+=","+name
            mid_file = file_dir+"{0}/".format(cell_line)+"3d5_"+name+"/"
            name=str(size3[0])+"kb"
            names+=","+name
            nb_file = file_dir+"{0}/".format(cell_line)+"3d5_"+name+"/"
            y_pred, y_test,AUC,ACC,AUPR,test_loss,precision,recall,fmeasure = main(cell_line, inter_file,mid_file,nb_file)
            types=str(j+1)
            new_row = pd.DataFrame({                              
                'Cell Line': [cell_line],
                'size': [names],
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