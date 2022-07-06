import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
torch.set_default_tensor_type(torch.DoubleTensor)

def getedge(x,edge_number):
    df = pd.read_csv('newedge', nrows=edge_number)
    r1 = df.loc[:, 'row'].values
    r2 = df.loc[:, 'column'].values
    return r1, r2


def getdata(file_name,data_type):
    df = pd.read_csv(file_name)
    x=df.loc[df.node==0]
    x = df[data_type].to_numpy()
    return x

def getdata_normalize(file_name,data_type):
    df = pd.read_csv(file_name)
    x = df[data_type[0]].to_numpy()
    x_n = (x - min(x)) / (max(x) - min(x))

    x_2= df[data_type[1]].to_numpy()
    #x_3= df[data_type[2]].to_numpy()
    return x_n,x_2

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def get_train_data(data,history,future):
    data_x = []
    data_y = []
    for i in range(len(data)-history-future):
       data_x.append(data[i:i+history])
       data_y.append(data[i+history:i+history+future])
    data_x=np.array(data_x)
    data_y=np.array(data_y)
    size1 = int(len(data_x) * 0.8)
    size2 = int(len(data_x) * 1)
    train_x = data_x[:size1]
    train_y = data_y[:size1]
    test_x = data_x[size1:size2]
    test_y = data_y[size1:size2]
    val_x = data_x[size2:]
    val_y = data_y[size2:]
    # train_x = np.expand_dims(train_x,axis=-1)
    # train_y = np.expand_dims(train_y,axis=-1)
    # test_x = np.expand_dims(test_x,axis=-1)
    # test_y = np.expand_dims(test_y,axis=-1)
    # val_x = np.expand_dims(val_x,axis=-1)
    # val_y = np.expand_dims(val_y,axis=-1)
    return train_x,train_y,test_x,test_y,val_x,val_y

a1,a2=getdata_normalize('DATA',['node_cpu_usage','node_memory_usage'])
x=np.array(list(zip(a1)))
x=x.reshape(1440,301,1)

history,future=10,3
train_x,train_y,test_x,test_y,val_x,val_y=get_train_data(x,history,future)

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)


batch_size=200

train_dataset=Data.TensorDataset(train_x,train_y)
test_dataset=Data.TensorDataset(test_x,test_y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.rnn=nn.GRU(1,5,batch_first=False)
        self.fc=nn.Linear(5,3)
    def forward(self,x):
        x=x.permute(1,0,2,3)
        x=x.reshape(10,-1,1)
        a,h=self.rnn(x)
        
        y=self.fc(h)
        y = y.reshape(1,-1,301, 3)
        y = y.permute(1, 3, 2, 0)
        return y

mymodel=model().cuda()
num_epochs=20
learning_rate=0.0003
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
output_windows=future
for epochs in range(num_epochs):
  iter1 = 0
  iter2 = 0
  loss_total=0
  RMSET=0
  for i,(batch_x, batch_y) in enumerate (train_loader):
     
     batch_x=batch_x.cuda()
     batch_y=batch_y.cuda()
     outputs = mymodel(batch_x)
     # clear the gradients
     optimizer.zero_grad()
     #loss
     loss = criterion(outputs,batch_y)
     loss_total=loss_total+loss.item()
     #backpropagation
     loss.backward()
     optimizer.step()
     iter1+=1
  loss_avg = loss_total/iter1
  print("epoch:%d,  loss: %1.5f" % (epochs, loss_avg))
  
mae = MeanAbsoluteError().cuda()
mape=MeanAbsolutePercentageError().cuda()
mse=MeanSquaredError().cuda()
net = mymodel.eval().cuda()
real=[]
prediction=[]
history = []
MAE=0
MAPE=0
MSE=0
BAT_=0
for u in range(future):
     for i,(batch_x, batch_y) in enumerate (test_loader):
        batch_x=batch_x.cuda()
        batch_y=batch_y.cuda()
        outputs = net(batch_x)       
        MAE_d=mae(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
        MAPE_d=mape(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
        MSE_d=mse(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
        MAE+=MAE_d
        MAPE+=MAPE_d
        MSE+=MSE_d
        BAT_+=1
     print("TIME:%d ,MAE:%1.5f,  MAPE: %1.5f, MSE: %1.5f" % (30*(u+1),MAE/BAT_, MAPE/BAT_,MSE/BAT_))




                                     