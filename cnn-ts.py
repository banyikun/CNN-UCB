from packages import *
from cnn_class import CNN_2d, CNN_1d
from load_data import load_yelp, load_mnist_2d,load_cifar10_3d, load_notmnist_2d


if torch.cuda.is_available():  
    dev = "cuda:3" 
else:  
    dev = "cpu" 
device = torch.device(dev)


## flatten a large tuple containing tensors of different sizes
def flatten(tensor):
    T=torch.tensor([]).to(device)
    for element in tensor:
        T=torch.cat([T,element.to(device).flatten()])
    return T
    
#concatenation of all the parameters of a NN
def get_theta(model):
    return flatten(model.parameters())


class CNN_TS:
    """Neural Thompson Sampling Strategy"""
    def __init__(self,dim, n_arm, m=100, hidden = 100, reg=1,sigma=1,nu=0.001,in_channel =1, if_2d = 1, kernel_s =2,  trounds = 5000):
        self.K = n_arm 
        self.nu=nu 
        self.sigma=sigma 
        self.m = m
        self.d = dim
        if if_2d:
            self.estimator = CNN_2d(hidden, in_channel).to(device)
        else:
            self.estimator = CNN_1d(hidden, kernel_s).to(device)
        self.optimizer = torch.optim.SGD(self.estimator.parameters(), lr = 0.001)
        self.current_loss=0
        self.t=1
        self.total_param = sum(p.numel() for p in self.estimator.parameters() if p.requires_grad)
        self.Design =  torch.ones((self.total_param,)).to(device)
        self.rewards=[]
        self.context_list = []
        self.trounds= trounds        
        
    def select(self, context):
        context = torch.from_numpy(context).float()
        if not if_2d:
            context = torch.unsqueeze(context, 1)
        #self.features = torch.unsqueeze(context, 1).to(device)
        estimated_rewards=[]
        mu = self.estimator(context.to(device))

        for f in mu:
            self.estimator.zero_grad()
            f.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.estimator.parameters()])      
            sigma2 = g * g /self.Design
            sigma = torch.sqrt(torch.sum(sigma2)) * self.nu
            sample = torch.normal(mean=f.item(), std=sigma)
            estimated_rewards.append(sample.item())
        
        arm_to_pull=np.argmax(estimated_rewards)
        
        return arm_to_pull
    
    def update(self, context, reward):
        context = torch.from_numpy(context).float()
        context = torch.unsqueeze(context, 0)
        if not if_2d:
            context = torch.unsqueeze(context, 1)
        self.context_list.append(context)
        self.rewards.append(reward)
        

       
        f_t=self.estimator(context.to(device))
        g=torch.autograd.grad(outputs=f_t,inputs=self.estimator.parameters())
        g=flatten(g)
        g=g/(np.sqrt(self.m))
        self.Design+=torch.matmul(g,g.T).to(device)
        self.t+=1


    def train(self):
        length = len(self.rewards)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx].to(device)
                r = self.rewards[idx]
                delta = self.estimator(c) - r
                self.current_loss = delta * delta
                self.optimizer.zero_grad() 
                    #gradient descent
                if self.t==1:
                    self.current_loss.backward(retain_graph=True)    
                else:
                    self.current_loss.backward()
                self.optimizer.step() 
                batch_loss +=  self.current_loss.item()
                tot_loss +=  self.current_loss.item()
                cnt += 1
                if cnt >= self.trounds:
                    return tot_loss / self.trounds
            if batch_loss / length <= 1e-3:
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
        arg_nu = 0.1
        arg_lambda = 0.001
        arg_in_channel = 1
        lr = 0.01
        trounds = 1000
        if_2d = 0

    elif args.dataset == "notmnist":
        b = load_notmnist_2d()
        arg_shuffle = 1
        arg_nu = 0.01 
        arg_lambda = 0.00001
        arg_hidden = 31360
        arg_in_channel = 1
        if_2d = 1

    l = CNN_TS(b.dim, n_arm = b.n_arm, hidden = arg_hidden, nu = arg_nu, in_channel = arg_in_channel,  if_2d = if_2d, trounds = trounds)
    regrets = []
    summ = 0
    for t in range(10000):
        context, rwd = b.step()
        arm_select = l.select(context)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        summ+=reg
        l.update(context[arm_select], r)
        if t<1000:
            if t%10 == 0:
                loss = l.train()
        else:
            if t%100 == 0:
                loss = l.train()
        regrets.append(summ)
        if t % 50 == 0:
            print('{}: {}, {:.3f}, {:.4f}'.format(t+1, summ, summ/(t+1), loss))



                        
