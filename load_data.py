from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np


class load_cifar10_1d:
    def __init__(self, is_shuffle=True):
        # Fetch data
        batch_size = 1
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
        self.dataiter = iter(trainloader)
        self.n_arm = 3
        self.dim = 9216


        
    def step(self):
        x, y = self.dataiter.next()
        d = x.numpy()[0]
        d = d.reshape(3072)
        target = int(y.item()/4.0)
        X_n = []
        for i in range(3):
            front = np.zeros((3072*i))
            back = np.zeros((3072*(2 - i)))
            new_d = np.concatenate((front,  d, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)    
        rwd = np.zeros(self.n_arm)
        #print(target)
        rwd[target] = 1
        #print(rwd)
        #print(X_n.shape)
        return X_n, rwd
    

    
class load_cifar10_3d:
    def __init__(self, is_shuffle=True):
        # Fetch data
        batch_size = 1
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
        self.dataiter = iter(trainloader)
        self.n_arm = 3
        self.dim = 9216

    def step(self):
        x, y = self.dataiter.next()
        d = x.numpy()[0]
        target = int(y.item()/4.0)
        #print(target)
        #print(data.shape)
        X_n = []
        for i in range(3):
            front = np.zeros((3, 32*i, 32))
            back = np.zeros((3, 32*(2 - i), 32))
            new_d = np.concatenate((front,  d, back), axis=1)
            X_n.append(new_d)
        X_n = np.array(X_n)    
        rwd = np.zeros(self.n_arm)
        #print(target)
        rwd[target] = 1
        #print(rwd)
        #print(X_n.shape)
        return X_n, rwd    
    
    
class load_mnist_1d:
    def __init__(self):
        # Fetch data
        batch_size = 1
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size,
                                      shuffle=True, num_workers=2)
        self.dataiter = iter(train_loader)
        self.n_arm = 10
        self.dim = 7840
 
    def step(self):
        x, y = self.dataiter.next()
        d = x.numpy()[0]
        d = d.reshape(784)
        target = y.item()
        X_n = []
        for i in range(10):
            front = np.zeros((784*i))
            back = np.zeros((784*(9 - i)))
            new_d = np.concatenate((front,  d, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)    
        rwd = np.zeros(self.n_arm)
        #print(target)
        rwd[target] = 1
        return X_n, rwd  
    
    
    

class load_notmnist:
    def __init__(self):
        # Fetch data

        X = np.load('./imagedat.npy', allow_pickle=True)
        y = np.load('./labeldata.npy', allow_pickle=True)
        new_X = []
        for i in X:
            i = i.flatten()
            new_X.append(i)
        X = np.array(new_X)
        print('notmnist', X.shape)
        X[np.isnan(X)] = - 1
        X = normalize(X)
        self.X, self.y =shuffle(X, y)
        # generate one_hot coding:
        self.y_arm = OrdinalEncoder(
            dtype=np.int).fit_transform(self.y.reshape((-1, 1)))
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = np.max(self.y_arm) + 1
        self.dim = self.X.shape[1] * self.n_arm
        self.act_dim = self.X.shape[1]

    def step(self):
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                self.act_dim] = self.X[self.cursor]
        arm = self.y_arm[self.cursor][0]
        rwd = np.zeros((self.n_arm,))
        rwd[arm] = 1
        self.cursor += 1
        return X, rwd
    
    
class load_mnist_2d:
    def __init__(self):
        # Fetch data
        batch_size = 1
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size,
                                      shuffle=True, num_workers=2)
        self.dataiter = iter(train_loader)
        self.n_arm = 10
        self.dim = 7840
 
    def step(self):
        x, y = self.dataiter.next()
        d = x.numpy()[0]
        target = y.item()
        X_n = []
        for i in range(10):
            front = np.zeros((1, 28*i, 28))
            back = np.zeros((1, 28*(9 - i), 28))
            new_d = np.concatenate((front,  d, back), axis=1)
            X_n.append(new_d)
        X_n = np.array(X_n)   
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        return X_n, rwd  
    

class load_notmnist_2d:
    def __init__(self):
        X = np.load('./imagedat.npy', allow_pickle=True)
        y = np.load('./labeldata.npy', allow_pickle=True)
        X[np.isnan(X)] = - 1
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        self.X, self.y =shuffle(X, y)
        self.n_arm = 10
        self.dim = 7840
        self.dataiter_x = iter(self.X)
        self.dataiter_y = iter(self.y)


    def step(self):
        x = [next(self.dataiter_x)]
        y = next(self.dataiter_y)
        X_n = []
        for i in range(10):
            front = np.zeros((1, 28*i, 28))
            back = np.zeros((1, 28*(9 - i), 28))
            new_d = np.concatenate((front,  x, back), axis=1)
            X_n.append(new_d)
        X_n = np.array(X_n)   
        rwd = np.zeros(self.n_arm)
        rwd[y] = 1
        return X_n, rwd            

    
    
class load_yelp:
    def __init__(self):
        # Fetch data
        self.m = np.load("../Yelp/yelp_2000users_10000items.npy")
        self.U = np.load("../Yelp/yelp_2000users_10000items_features.npy")
        self.I = np.load("../Yelp/yelp_10000items_2000users_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in range(len(self.m)):
            for j in range(len(self.m[0])):
                if self.m[i][j] > 0:
                    self.pos_index.append((i,j))
                if self.m[i][j] < -1:
                    self.neg_index.append((i,j))
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        print(self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), 9)]
        neg = self.neg_index[np.random.choice(range(self.n_d))]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]])))
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X), rwd
    