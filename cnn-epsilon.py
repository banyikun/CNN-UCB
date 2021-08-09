from packages import *
from load_data import load_yelp, load_mnist_2d,load_cifar10_3d, load_notmnist_2d
from cnn_class import CNN_2d, CNN_1d

class Neural_epsilon:
    def __init__(self,  hidden = 100, in_channel =1, lamdba=1, nu=0.1, if_2d = 1, kernel_s =2, lr = 0.001, trounds = 5000, p =0.1):
        if if_2d:
            self.func = CNN_2d(hidden, in_channel).to(device)
        else:
            self.func = CNN_1d(hidden, kernel_s).to(device)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).to(device)
        self.nu = nu
        self.p = p  # probability of making exploration
        self.lr = lr
        self.trounds= trounds
        self.loss = torch.nn.MSELoss()
        self.if_2d = if_2d


    def select(self, context):
        tensor = torch.from_numpy(context).float()
        mu = self.func(tensor.to(device))
        res = []
        for fx in mu:
            res.append(fx.item())
        epsilon = np.random.binomial(1, self.p)
        if epsilon:
            #print("random")
            arm = np.random.choice(len(context), 1)
        else:
            #print("greedy")
            arm = np.argmax(res)
        return arm, res
    
    def update(self, context, reward):
        
        if self.if_2d:
            if len(context.shape) < 4:
                new_context =  torch.unsqueeze(torch.from_numpy(context).float(), 0)
            else:
                new_context =  torch.from_numpy(context).float()
        else:
            new_context =  torch.from_numpy(context).float()
            if len(context.shape) != 3:
                new_context = torch.unsqueeze(new_context, 0)

        self.context_list.append(new_context)
        self.reward.append(reward)

    def train(self):
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            cnt_w = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                r = torch.tensor([r]).float().to(device)
                optimizer.zero_grad()
                output = self.func(c.to(device))
                loss = self.loss(output, r)
                #loss = (output - r)**2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= self.trounds:
                    return tot_loss / self.trounds
            if batch_loss / length <= 0.001:
                return batch_loss / length  
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN UCB')
    parser.add_argument('--dataset', default='mnist', type=str, help='mnist, cifar10, notmnist, yelp')
    args = parser.parse_args()
    lr = 0.01
    trounds = 2000
    
    if args.dataset == "mnist":
        b = load_mnist_2d()
        arg_shuffle = 1
        arg_nu = 0.01  
        arg_lambda = 0.00001
        arg_hidden = 31360
        arg_in_channel = 1
        if_2d = 1

    elif args.dataset == "cifar10":
        b = load_cifar10_3d()
        arg_shuffle = 1
        arg_nu = 0.01  
        arg_lambda = 0.00001
        arg_hidden = 12288
        arg_in_channel = 3
        if_2d = 1

    elif args.dataset == "yelp":
        b = load_yelp()
        arg_kernel_size = 2
        arg_hidden = 360
        arg_nu = 1
        arg_lambda = 0.001
        arg_in_channel = 1
        if_2d = 0
        lr = 0.01
        trounds = 1000

    elif args.dataset == "notmnist":
        b = load_notmnist_2d()
        arg_shuffle = 1
        arg_nu = 0.01 
        arg_lambda = 0.00001
        arg_hidden = 31360
        arg_in_channel = 1
        if_2d = 1
        

        
    l = Neural_epsilon(hidden = arg_hidden, in_channel = arg_in_channel, lamdba = arg_lambda, nu = arg_nu, if_2d = if_2d, lr = lr, trounds = trounds)
    regrets = []
    summ = 0
    print("number of rounds: regrets, regrets/rounds, loss")
    for t in range(10000):
        context, rwd =  b.step()
        if not if_2d:
            context = np.expand_dims(context, axis=1)
        arm_select, res = l.select(context)
        r = rwd[arm_select]
        if not np.isscalar(r):
            r= r[0]
        reg = np.max(rwd) - r
        summ+=reg
        arm_star = np.argmax(rwd)
        l.update(context[arm_select], r)

        if t<1000:
            if t%10 == 0:
                #print(l.right)
                loss = l.train()
                #print("values:", f[arm_select], s[arm_select], "right:", f[arm_star], s[arm_star])

        else:
            if t%100 == 0:
                loss = l.train()
                
        regrets.append(summ)
        if t % 50 == 0:
            print('{}: {}, {:.3f}, {:.4f}'.format(t+1, summ, summ/(t+1), loss))
    
    
    
    

            
            
            