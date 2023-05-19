import numpy as np
import torch as tc
from data_loader import CovidCTDataset
import torchvision.transforms as transforms
from simple_cnn import SimpleCNN

import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import  DataLoader
from sklearn.metrics import roc_auc_score



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def check_cuda_availability():
    if tc.cuda.is_available():
        print("CUDA is available.")
        device = tc.device("cuda")
        print("Using GPU:", tc.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU.")
        device = tc.device("cpu")
    return device

def train(optimizer, epoch):
    
    model.train()
    
    train_loss = 0
    train_correct = 0
    
    for batch_index, batch_samples in enumerate(train_loader):
        
        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
 
        optimizer.zero_grad()
        output = model(data)

        criteria = tc.nn.CrossEntropyLoss()
        loss = criteria(output, target.long())

        train_loss += criteria(output, target.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()
    
        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
    
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))

def val():
    
    model.eval()
    test_loss = 0
    correct = 0
    
    criteria = tc.nn.CrossEntropyLoss()

    with tc.no_grad():
        
        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.long().view_as(pred)).sum().item()
            
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
           
          
    return targetlist, scorelist, predlist

def test():
    
    model.eval()
    test_loss = 0
    correct = 0

    criteria = tc.nn.CrossEntropyLoss()
    # Don't update model
    with tc.no_grad():

        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.long().view_as(pred)).sum().item()

            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
           
    return targetlist, scorelist, predlist

if __name__ == '__main__':
    device = check_cuda_availability()    
    batchsize=10
    trainset = CovidCTDataset(root_dir=r'Data/',
                              txt_COVID=r'Classification_Authors_based_Data_Split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID=r'Classification_Authors_based_Data_Split/NonCOVID/trainCT_NonCOVID.txt',
                              transform= train_transformer)
    valset = CovidCTDataset(root_dir=r'Data/',
                              txt_COVID=r'Classification_Authors_based_Data_Split/COVID/valCT_COVID.txt',
                              txt_NonCOVID=r'Classification_Authors_based_Data_Split/NonCOVID/valCT_NonCOVID.txt',
                              transform= val_transformer)
    testset = CovidCTDataset(root_dir=r'Data/',
                              txt_COVID=r'Classification_Authors_based_Data_Split/COVID/testCT_COVID.txt',
                              txt_NonCOVID=r'Classification_Authors_based_Data_Split/NonCOVID/testCT_NonCOVID.txt',
                              transform= val_transformer)
    print(trainset.__len__())
    print(valset.__len__())
    print(testset.__len__())

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)
    model = SimpleCNN().cuda()
    modelname = 'SimpleCNN'

    # train
    bs = 10
    votenum = 10
    import warnings
    warnings.filterwarnings('ignore')

    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []

    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    total_epoch = 50
    for epoch in range(1, total_epoch+1):
        train(optimizer, epoch)
        
        targetlist, scorelist, predlist = val()
 
        vote_pred = vote_pred + predlist 
        vote_score = vote_score + scorelist 

        if epoch % votenum == 0:
            
            # major vote
            vote_pred[vote_pred <= (votenum/2)] = 0
            vote_pred[vote_pred > (votenum/2)] = 1
            vote_score = vote_score/votenum
            
            print('vote_pred', vote_pred)
            print('targetlist', targetlist)
            TP = ((vote_pred == 1) & (targetlist == 1)).sum()
            TN = ((vote_pred == 0) & (targetlist == 0)).sum()
            FN = ((vote_pred == 0) & (targetlist == 1)).sum()
            FP = ((vote_pred == 1) & (targetlist == 0)).sum()
            
            
            print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
            print('TP+FP',TP+FP)
            p = TP / (TP + FP)
            print('precision',p)
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            print('recall',r)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1',F1)
            print('acc',acc)
            AUC = roc_auc_score(targetlist, vote_score)
            print('AUCp', roc_auc_score(targetlist, vote_pred))
            print('AUC', AUC)
            
            
            tc.save(model.state_dict(), "model_backup/{}.pt".format(modelname))  
            
            vote_pred = np.zeros(valset.__len__())
            vote_score = np.zeros(valset.__len__())
            print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
            epoch, r, p, F1, acc, AUC))

            f = open('model_result/{}.txt'.format(modelname), 'a+')
            f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
            epoch, r, p, F1, acc, AUC))
            f.close()

    # test
    bs = 10
    import warnings
    warnings.filterwarnings('ignore')

    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []

    vote_pred = np.zeros(testset.__len__())
    vote_score = np.zeros(testset.__len__())

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = StepLR(optimizer, step_size=1)

    total_epoch = 10
    for epoch in range(1, total_epoch+1):
        
        targetlist, scorelist, predlist = test()

        vote_pred = vote_pred + predlist 
        vote_score = vote_score + scorelist 
        
        TP = ((predlist == 1) & (targetlist == 1)).sum()
        TN = ((predlist == 0) & (targetlist == 0)).sum()
        FN = ((predlist == 0) & (targetlist == 1)).sum()
        FP = ((predlist == 1) & (targetlist == 0)).sum()

        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1',F1)
        print('acc',acc)
        AUC = roc_auc_score(targetlist, vote_score)
        print('AUC', AUC)

        if epoch % votenum == 0:
            
            # major vote
            vote_pred[vote_pred <= (votenum/2)] = 0
            vote_pred[vote_pred > (votenum/2)] = 1
            
            TP = ((vote_pred == 1) & (targetlist == 1)).sum()
            TN = ((vote_pred == 0) & (targetlist == 0)).sum()
            FN = ((vote_pred == 0) & (targetlist == 1)).sum()
            FP = ((vote_pred == 1) & (targetlist == 0)).sum()
            
            print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
            print('TP+FP',TP+FP)
            p = TP / (TP + FP)
            print('precision',p)
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            print('recall',r)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1',F1)
            print('acc',acc)
            AUC = roc_auc_score(targetlist, vote_score)
            print('AUC', AUC)
            
            
            vote_pred = np.zeros((1,testset.__len__()))
            vote_score = np.zeros(testset.__len__())
            print('vote_pred',vote_pred)
            print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
            epoch, r, p, F1, acc, AUC))

            f = open(f'model_result/test_{modelname}.txt', 'a+')
            f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
            epoch, r, p, F1, acc, AUC))
            f.close()

