import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from load_data import load_data
import torchvision
batch_size_train = 1


if torch.cuda.is_available():  
    dev = "cuda:1" 
else:  
    dev = "cpu" 
device = torch.device(dev)
print(device)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size =20)
        self.conv2 = nn.Conv1d(10, 10, kernel_size =20)
        self.conv3 = nn.Conv1d(10, 10, kernel_size =20)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(19455, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.m = nn.Softplus()


    def forward(self, x):
        x = self.conv1(x)
        #x = self.m(x)
        #x = F.relu(x)
        x = torch.sigmoid(x)
        
        x = self.conv2(x)
        #x = self.m(x)
        x = torch.sigmoid(x)
        #x = F.relu(x)
        
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc1(x)
        #x = F.relu(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x
    
    
    
class CNN_UCB:
    def __init__(self, dim, lamdba=1, nu=1):
        self.func = CNN().to(device)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).to(device)
        self.nu = nu
        self.right = 0
        self.wrong = 0

    def select(self, context):
        mu = self.func(context.to(device))
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        #print(mu)
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sample_r = fx.item() + sigma.item()
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
        #print(sampled)
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, g_list[arm].norm().item(), ave_sigma, ave_rew
    
    def update(self, context, reward):
        con = torch.unsqueeze(context, 0)
        self.context_list.append(con)
        self.reward.append(reward)
        if reward >0:
            self.right +=1
        else:
            self.wrong +=1

    def train(self):
        optimizer = optim.Adam(self.func.parameters(), lr=0.0005)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        #print(index)
        while True:
            batch_loss = 0
            cnt_w = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                output = self.func(c.to(device))
                loss = (output - r)**2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length

            
            
if __name__== "__main__":
    arg_dataset = 'mnist'
    arg_size = 1
    arg_shuffle = 1
    arg_seed = 0
    arg_nu = 1
    arg_lambda = 0.0001
    arg_hidden = 100
    use_seed = 0


    arg_dataset = 'mnist'
    use_seed = None if arg_seed == 0 else arg_seed
    b = load_data(arg_dataset, is_shuffle=arg_shuffle, seed=use_seed)
    
    l = CNN_UCB(7840, arg_lambda, arg_nu)
    regrets = []
    summ = 0
    for t in range(2000):
        context, rwd =  b.step()
        context = torch.from_numpy(context).float()
        context = torch.unsqueeze(context, 1)
        arm_select, nrm, sig, ave_rwd = l.select(context)
        r = rwd[arm_select]
        reg = 1 - r
        summ+=reg
        l.update(context[arm_select], r)
        if t<1000:
            if t%10 == 0:
                #print(l.right)
                loss = l.train()
        else:
            if t%500 == 0:
                loss = l.train()
        regrets.append(summ)
        if t % 50 == 0:
            print('{}: {}, {:.3f}, {:.3e}, {:.3e}'.format(t, summ, summ/(t+1), loss, nrm))



    
