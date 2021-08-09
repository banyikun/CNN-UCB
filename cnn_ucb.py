from packages import *
from load_data import load_yelp, load_mnist_2d,load_cifar10_3d, load_notmnist_2d
from cnn_class import CNN_2d, CNN_1d



if torch.cuda.is_available():  
    dev = "cuda:2" 
else:  
    dev = "cpu" 
device = torch.device(dev)


class CNN_UCB:
    def __init__(self,  hidden = 100, in_channel =1, lamdba=1, nu=0.1, if_2d = 1, kernel_s =2, lr = 0.001, trounds = 5000):
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
        self.loss = torch.nn.MSELoss()
        self.lr = lr
        self.trounds= trounds

    def select(self, context):
        mu = self.func(context.to(device))
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        f_l = []
        s_l = []
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
            f_l.append(fx.item())
            s_l.append(sigma.item())
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm,  f_l, s_l
    
    def update(self, context, reward):
        con = torch.unsqueeze(context, 0)
        self.context_list.append(con)
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
    lr = 0.001
    trounds = 5000
    
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
        

        
    l = CNN_UCB(hidden = arg_hidden, in_channel = arg_in_channel, lamdba = arg_lambda, nu = arg_nu, if_2d = if_2d, lr = lr, trounds = trounds)
    regrets = []
    summ = 0
    print("number of rounds: regrets, regrets/rounds, loss")
    for t in range(10000):
        context, rwd =  b.step()
        context = torch.from_numpy(context).float()
        if args.dataset == "yelp":
            context = torch.unsqueeze(context, 1)
        arm_select, f, s = l.select(context)
        r = rwd[arm_select]
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
    np.save("./cnn_%c_3d_10000.npy"%args.dataset, regrets)
    
    
    
    
