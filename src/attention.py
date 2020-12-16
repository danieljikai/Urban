import csv
import data_processing
import seaborn as sns
import matplotlib.pylab as plt

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchsummary import summary
from random import sample


class CreateDataset(Dataset):
    def __init__(self, X_hour, X_day, X_week, X_extra, y, timeslots):
        self.X_main = np.concatenate((X_day, X_hour, X_week), axis = 1)
        self.X_extra = X_extra
        self.y = y
        self.timeslots = timeslots
    
    def __len__(self):
        return self.X_main.shape[0]
    
    def __getitem__(self, index):
        return self.X_main[index], self.X_extra[index], self.y[index], self.timeslots[index]
        

def train_test_split(X_hour, X_day, X_week, X_extra, y, timeslots, ratio = 0.8):
    n = X_hour.shape[0]
    train_index = sorted(sample(range(n),int(ratio*n)))
    test_index = [i for i in range(n) if i not in train_index]
    X_hour_train = X_hour[train_index,:,:,:]
    X_hour_test = X_hour[test_index,:,:,:]
    X_day_train = X_day[train_index,:,:,:]
    X_day_test = X_day[test_index,:,:,:]
    X_week_train = X_week[train_index,:,:,:]
    X_week_test = X_week[test_index,:,:,:]
    X_extra_train = X_extra[train_index,:]
    X_extra_test = X_extra[test_index,:]
    y_train = y[train_index,:,:,:]
    y_test = y[test_index,:,:,:]
    timeslots_train = timeslots[train_index]
    timeslots_test = timeslots[test_index]
    train_dataset = CreateDataset(X_hour_train, X_day_train, X_week_train, X_extra_train, y_train, timeslots_train)
    test_dataset = CreateDataset(X_hour_test, X_day_test, X_week_test, X_extra_test, y_test, timeslots_test)
    return train_dataset, test_dataset
    
            
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(18, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Flatten(),
	        nn.Linear(256*4*4, 64)
        )
        self.encoder = nn.Sequential(
            nn.Linear(21, 5)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64+5, 256*4*4),

        )      
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), #[B, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), #[B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), #[B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 2, 3, 1, 1), #[B, 2, 32, 32]
            nn.Tanh())

    def forward(self, X_main, X_extra):
        X_extra = self.encoder(X_extra).view(X_extra.shape[0],-1)
        X_main = self.conv(X_main)
        X = torch.cat([X_main, X_extra], 1) #[B, 64+5]
        X = self.fc(X).view(X.shape[0], -1, 4, 4) 
        X = self.decoder(X)
        return X

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
                nn.Linear(18*32*32, 18*32*4),
                nn.ReLU(True),
                nn.Linear(18*32*4, 18*16),
                nn.ReLU(True),
                nn.Linear(18*16, 64),
                )
        self.encoder2 = nn.Sequential(
                nn.Linear(21, 5),
                )
        self.decoder = nn.Sequential(
                nn.Linear(64+5, 64*2),
                nn.ReLU(True),
                nn.Linear(2*32*2, 2*32*8),
                nn.ReLU(True),
                nn.Linear(2*32*8, 2*32*32),
                nn.Tanh(),
                )
    def forward(self, X_main, X_extra):
        print(X_main.shape,X_main.view(X_main.shape[0],-1).shape)
        X_main = self.encoder1(X_main.view(X_main.shape[0],-1))
        print(X_main.shape)
        X_extra = self.encoder2(X_extra.view(X_extra.shape[0],-1))
        X = torch.cat([X_main, X_extra], 1)
        out = self.decoder(X).view(X.shape[0],-1,32,32)
        return out

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(18+5, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )  
        self.encoder = nn.Sequential(
            nn.Linear(21, 5)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), #[B, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), #[B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), #[B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 2, 3, 1, 1), #[B, 2, 32, 32]
            nn.Tanh())
        
    def forward(self, X_main, X_extra):
        X_extra = self.encoder(X_extra).view(X_extra.shape[0],-1)
        m,n = X_extra.shape[0], X_extra.shape[1]
        X_extra.unsqueeze_(-1)
        X_extra = X_extra.expand(m,n,32)
        X_extra.unsqueeze_(-1)
        X_extra = X_extra.expand(m,n,32,32)
        
        X = torch.cat([X_main, X_extra], 1)
        out = self.conv(X)
        out = self.decoder(out)
        return out

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(4, 1)
        # self.attn = nn.Sequential(
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True)
        # )
        self.encoder = nn.Sequential(
            nn.Linear(21, 5)
        )
        self.main_encoder = nn.Sequential(
                nn.Linear(18*1024, 18*256),
                nn.ReLU(True),
                nn.Linear(18*256, 18*64),
                nn.ReLU(True),
                nn.Linear(18*64, 18*4),
                nn.ReLU(True), 
        )

        self.attdec = nn.Sequential(
                nn.Linear(77, 2*64),
                nn.ReLU(True),
                nn.Linear(2*64, 2*256),
                nn.ReLU(True),
                nn.Linear(2*256, 2*1024),
                nn.Tanh(),
        )

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, 4, 2, 1), #[B, 128, 8, 8]
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, 64, 4, 2, 1), #[B, 64, 16, 16]
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1), #[B, 32, 32, 32]
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 2, 3, 1, 1), #[B, 2, 32, 32]
        #     nn.Tanh()
        # )
        
    def forward(self, X_main, X_extra):
        # print(X_main.size(), X_extra.size())
        X_main = X_main.view(X_main.shape[0],-1)
        X_extra = X_extra.view(X_extra.shape[0], -1)
        X_main = self.main_encoder(X_main)
        X_extra = self.encoder(X_extra)
        # print(X_main.size(), X_extra.size())
        # input()
        m,n = X_extra.shape[0], X_extra.shape[1]
        # X_extra.unsqueeze_(-1)
        # X_extra = X_extra.expand(m,n,32)
        # X_extra.unsqueeze_(-1)
        # X_extra = X_extra.expand(m,n,32,32)
        X = X_main
        X = X.view(-1, 18, 4)
        X = X.permute(1, 0, 2)

        # print(':::::::', X.size())
        out, out_weight = self.attn(X, X, X)
        # print('out size, out_weight size: ', out.size(), out_weight.size())
        # input()
        out = out.permute(1, 0, 2)#.view(m, -1)

        out = out.reshape(m, -1)
        # print(':::::after permuute', out.size(), X_extra.size())
        out = torch.cat([out, X_extra], 1)

        # print(':::::::', out.size())
        out = self.attdec(out)
        return out.view(-1, 2, 32, 32)
    
def main():
    DIR = "../TaxiBJ"
    X_hour, X_day, X_week, X_extra, y, timeslots = data_processing.main(DIR, True)  
    
    num_epochs = 1
    batch_size = 16
    learning_rate = 1e-3

    model = Attention()
    # summary(model, input_size = [(18,32,32),(1,1,21)])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_dataset, test_dataset = train_test_split(X_hour, X_day, X_week, X_extra, y, timeslots)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for X1, X2, y, timeslot in train_dataloader:    
            X1 = Variable(X1.float())
            X2 = Variable(X2.float())
            y = Variable(y.float())
            output = model(X1, X2)
            # m = output[0][0].detach().numpy()
            # path = '../data/{}_{}.original.png'.format(2016021515, epoch)
            # print('drawing fig')
            # draw_fig(m, path)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch [{}/{}], loss:{}'.format(epoch + 1, num_epochs, loss.data))
    torch.save(model,"model")
    for X1, X2, y, timeslot in test_dataloader:  
        X1 = Variable(X1.float())
        X2 = Variable(X2.float())
        y = Variable(y.float())
        output = model(X1, X2)
        loss = criterion(output, y)
        print(loss.data)
    return model

def draw_fig(m,path):
    ax = sns.heatmap(m, linewidth=0.5, vmin = -1, vmax = 1)
    plt.savefig(path, dpi = 1000)
    plt.clf()
    
if __name__ == "__main__":
    
    model = main()
    

# DIR = "../TaxiBJ"
# X_hour, X_day, X_week, X_extra, y, timeslots = data_processing.main(DIR, True) 
# #model = ConvNN()
# #errors = dict()
# #E,C = 0,0 
# num_epochs = 1
# batch_size = 1
# learning_rate = 1e-3
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(
#        model.parameters(), lr=learning_rate, weight_decay=1e-5)
# dataset = CreateDataset(X_hour, X_day, X_week, X_extra, y, timeslots) 
# dataloader = DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True)
# for epoch in range(num_epochs):
#    for X1, X2, y, timeslot in dataloader:
#        if timeslot[0] != "2016021515":
#            continue
#        X1 = Variable(X1.float())
#        X2 = Variable(X2.float())
#        y = Variable(y.float())
#        output = model(X1, X2)
#        m = output[0][0].detach().numpy()
#        ax = sns.heatmap(m, linewidth=0.5, vmin = -1, vmax = 1)
#        plt.savefig('attention.png', dpi = 1000)
#        plt.clf()
       # loss = criterion(output, y)
       # optimizer.zero_grad()
       # loss.backward()
       # optimizer.step()
       # print('epoch [{}/{}], loss:{}'
       #   .format(epoch + 1, num_epochs, loss.data))
#torch.save(model,"model")
#        errors[timeslot[0]] = float(loss.data)
#        E += float(loss.data)
#        C += 1
#    with open("log.csv","a+") as f:
#        wr = csv.writer(f)
#        wr.writerow([epoch, E, C, E/C])
#    E,C = 0,0
