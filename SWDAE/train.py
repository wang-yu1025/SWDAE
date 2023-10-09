import torch
import numpy as np
from models.dawn import DAWN
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score
import PIL.Image as Image
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import scipy.io as sio
from util.Tens import tensor
import tensorly as tl
tl.set_backend('pytorch')

class Fc2(nn.Module):
    def __init__(self):
        super(Fc2, self).__init__()
 
        self.linear = nn.Linear(4096,2048)
        
    def forward(self, x):
        encoder_f = self.linear(x)
        return encoder_f

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
 
        self.encoder = nn.Sequential(
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Linear(640, 320),
            nn.ReLU(), 
            nn.Linear(320, 128),
            nn.ReLU(), 
            nn.Linear(128, 64),
            nn.ReLU(),)
        
        #self.linear = nn.Linear(64,25)
        
    def forward(self, x):
        encoder_1 = self.encoder(x)
        
        return encoder_1

class Fc(nn.Module):
    def __init__(self):
        super(Fc, self).__init__()
 
        self.linear = nn.Linear(64,25)
        
    def forward(self, x):
        encoder_f = self.linear(x)
        return encoder_f

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
 
        self.decoder = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1280),
            nn.ReLU(),
        )
        
    def forward(self, x):
        decoder_1 = self.decoder(x)

        return decoder_1

class Fc1(nn.Module):
    def __init__(self):
        super(Fc1, self).__init__()
 
        self.linear = nn.Linear(25,1)
        
    def forward(self, x):
        encoder_f = self.linear(x)
        return encoder_f


class TrainerDeepSVDD(object):
    def __init__(self, args, train_loader,test_loader, device, R, nu, writer):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.R1 = {}
        self.count=10
        for i in range(self.count):
            self.R1[i]=torch.tensor(R, device=self.device)
        self.nu = nu
        self.warm_up_n_epochs = 0
        self.writer = writer

    def set_c(self, model,dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""

        model.eval()
        z1={}
        for i in range(self.count):
            z1[i]=[]
        c1={}
        fea_fc={}
        with torch.no_grad():
            # for x, _ in dataloader:
            for x in dataloader:
                x = x.float().to(self.device)
                fea_fc,_,_= model(x)
                # z1.append(z111.detach())
                for i in range(self.count):    # 生成变量    
                    z1[i].append(fea_fc[i].detach())
        for i in range(self.count):
            z1[i] = torch.cat(z1[i])
            c1[i] = torch.mean(z1[i] , dim=0)
        # If c is close to 0, set to +-eps
        # To avoid trivial problem
        
            c1[i][(abs(c1[i] ) < eps) & (c1[i]  < 0)] = -eps
            c1[i][(abs(c1[i] ) < eps) & (c1[i]  > 0)] = eps

        
        return c1

    def MDT(self,feature, time_step):
        feature = feature.T
        fea_dim, sampe_num = feature.shape
        mdt_list = []
        for i in range(sampe_num-time_step+1):
            mdt_list.append(feature[:,i: i+time_step])
        mdt_list = [torch.unsqueeze(i, dim=2) for i in mdt_list]
        #return mdt_list
        return torch.cat(mdt_list, dim=2) 
    

    def MDT_inverst(self,tucker_feature):
        #默认的三维张量视为（batch_first, time_step, fea_dim）
        #张量的三个维度应该与上述每个维度的意义对应
        sampe_num,time_step,fea_dim = tucker_feature.shape  
        mdt_inv = [tucker_feature[0,:,:]]
        mdt_inv = mdt_inv + [tucker_feature[i:i+1,-1,:] for i in range(1,sampe_num)] 
        return torch.cat(mdt_inv, dim=0)
    

    def Tren_loss(self,HI,T):

        len=HI.shape[0]
        loss=torch.sqrt((len*torch.sum(HI**2)-(torch.sum(HI)**2))*(len*torch.sum(T**2)-(torch.sum(T)**2)))/torch.abs(((len*torch.sum(HI*T,dim=0))-((torch.sum(HI))*(torch.sum(T)))))


        return loss

    def _get_fold_tensor(self, tensor, mode, shape):
        if isinstance(tensor,list):
            return [tl.base.fold(ten, mode, shape) for ten in tensor ]
        elif isinstance(tensor, np.ndarray):
            return tl.base.fold(tensor, mode, shape)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")


    def _update_cores(self, n, Us, Xs, es, cores, alpha, beta, lam=1):

        begin_idx = self._p + self._q
        
        T_hat = len(Xs)
        
        unfold_cores = self._get_unfold_tensor(cores, n)
        H = self._get_H(Us, n)
        for t in range(begin_idx, T_hat):
            unfold_Xs = self._get_unfold_tensor(Xs[t], n)
            b = np.sum([ beta[i] * self._get_unfold_tensor(es[t-begin_idx][-(i+1)], n) for i in range(self._q)], axis=0)
            a = np.sum( [ alpha[i] * self._get_unfold_tensor(cores[t-(i+1)], n) for i in range(self._p)], axis=0 )
            unfold_cores[t] = 1/(1+lam) * (lam * np.dot( np.dot(Us[n].T, unfold_Xs), H.T) + a - b)
        return unfold_cores  

    def _get_Xs(self, trans_data):

        T_hat = trans_data.shape[-1]
        Xs = [ trans_data[..., t] for t in range(T_hat)]

        return Xs
    
    def _get_H(self, Us, n):

        Hs = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= n ])
        return Hs
    
    def _compute_convergence(self, new_U, old_U):
        
        new_old = [ n-o for n, o in zip(new_U, old_U)]
        
        a = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_old], axis=0)
        b = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_U], axis=0)
        return a/b  
    
    def get_radius(self,dist: torch.Tensor, nu: float):
    #"""Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    
    def get_data(self, data):

        data1 = data[500:3500]
        data2 = data[3000:5000]
        # data3 = data[4000:6000]
        # data4 = data[6000:8000]
        # data5 = data[8000:12000]
        # data6 = data[12000:16000]
        # data7 = data[16000:]

        rand_index1 = np.random.choice(1000, size=1000)
        rand_index2 = np.random.choice(1000, size=1000)
        # rand_index3 = np.random.choice(2000, size=self.bitch_size)
        # rand_index4 = np.random.choice(2000, size=self.bitch_size)
        # rand_index5 = np.random.choice(4000, size=self.bitch_size)
        # rand_index6 = np.random.choice(4000, size=self.bitch_size)
        # rand_index7 = np.random.choice(2000, size=self.bitch_size)

        x_batch1 = data1[rand_index1]
        x_batch2 = data2[rand_index2]
        # x_batch3 = data3[rand_index3]index 1840 is out of bounds for dimension 0 with size 1096
        # x_batch4 = data4[rand_index4]
        # x_batch5 = data5[rand_index5]
        # x_batch6 = data6[rand_index6]
        # x_batch7 = data7[rand_index7]
        # x_batch1 = torch.tensor(x_batch1)
        # x_batch2 = torch.tensor(x_batch2)
        inputs = torch.cat([x_batch1,x_batch2], 0)
        return inputs#随机选取数据训练
   
    #def train(self):
    def train(self):
        r={}
        rd={}
        ab={}
        for i in range(self.count):
            r[i]=0
            rd[i]=[]
            ab[i]=[]
        s1 = []
        l1 =[]
        time_step=5
        H=[]

        c1={}
        dist1={}
        scores1={}
        loss1={}
        net = DAWN(num_classes=2,
                   big_input=0,
                   first_conv=self.args.first_conv,
                   number_levels=self.args.levels,
                   kernel_size=self.args.kernel_size,
                   no_bootleneck=self.args.no_bootleneck,
                   classifier=self.args.classifier,
                   share_weights=self.args.share_weights,
                   simple_lifting=self.args.simple_lifting,
#                 COLOR=USE_COLOR,
                  regu_details=self.args.regu_details,
                  regu_approx=self.args.regu_approx,
                  haar_wavelet=self.args.haar_wavelet
                  ).to(self.device)
        enc = Encoder().to(self.device)
        
        dec = Decoder().to(self.device)
        fc = Fc().to(self.device)
        fc1=Fc1().to(self.device)
        fc2=Fc2().to(self.device)
    
        print("Start train")                                 
        loss_func = nn.MSELoss()
        
        optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':enc.parameters()},
                                           {'params':dec.parameters()},
                                           {'params':fc.parameters()},{'params':fc1.parameters()}])    
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)                                            
        # c1= self.set_c(net, self.train_loader)
        net.train()#启用 Batch Normalization 和 Dropout
        file = 'data'
        table = 's5'
    
        X1=sio.loadmat('/home/htu/workspace/wyy/SWDAE/data/%s.mat'%(file),)['%s'%(table)]
        # X1=sio.loadmat('/home/htu/workspace/wyy/2/data/S1.mat')['s1']
        
        epoch_loss=0
        X=np.float32(X1)
        a = torch.tensor(X)#.cuda()
        x1 = self.get_data(a)
        xx=[x1[i*50:(i+1)*50,:] for i in range(30)]
        # c1= self.set_c(net, xx)
        for epoch in range(self.args.num_epochs):
            
            total_loss = 0
            tl1= 0

            print('Step :',epoch)             
            tq = tqdm(xx, total=len(xx))
            i = 0
            for x in tq:
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                #f1=fc2(x).to(self.device)
                x1,ff,rs=net(x)
                
                x_enc = enc(x1).to(self.device)
                #print(x_enc.shape)
                fea = fc(x_enc).to(self.device)
                H=fc1(fea).to(self.device)
                #HH=H.detach().numpy()
                ll=len(x1)
                tt=torch.tensor(range(1,ll+1)).reshape(ll,1).to(self.device)
                loss = 0
                x_dec = dec(fea)
                loss1=loss_func(x1, x_dec)
                loss = loss1 + 2*(rs[0]+rs[1]+rs[2])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                errors = {
                    'epoch': epoch,
                    'train loss': loss.item()
                }
                i = i+1
                tq.set_postfix(errors)
            l1.append(loss)
            f = torch.tensor(l1)
            loss_mean = f.detach().numpy()
            sio.savemat('/home/htu/workspace/wyy/2/out/loss_CP.mat',{'loss':loss_mean})
            self.writer.add_scalar("WTAE/Loss", epoch_loss, epoch)
            scheduler.step()
            self.net = net
            self.c1 = c1

        return Encoder,Decoder,Fc,Fc2,self.net,self.c1#,self.train_loader
                                
    def checkpoint_load(self, model_dir,dataloader_test,dataloader_train):

        net = DAWN(num_classes=2,
                   big_input=0,
                   first_conv=self.args.first_conv,
                   number_levels=self.args.levels,
                   kernel_size=self.args.kernel_size,
                   no_bootleneck=self.args.no_bootleneck,
                   classifier=self.args.classifier,
                   share_weights=self.args.share_weights,
                   simple_lifting=self.args.simple_lifting,
#                 COLOR=USE_COLOR,
                  regu_details=self.args.regu_details,
                  regu_approx=self.args.regu_approx,
                  haar_wavelet=self.args.haar_wavelet
                  )
        checkpoint = torch.load(model_dir, map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
        # net = nn.DataParallel(net) 
        net.load_state_dict(checkpoint) # 接着就可以将模型参数load进模型。
        # self.traind(net,dataloader_test,dataloader_train)
        
        net,c1=self.train()
        self.test(net, dataloader_test,dataloader_train,c1)

    def test(self, net, Encoder,Decoder,Fc,Fc2,c1): 
        net.to(self.device)
        net.eval()
        print("Start testing")
        res_path = "/home/htu/workspace/ZEJ/xb_svdd10soft/data/fea"
        bs=16
        labels = []
        scores1 = {}
        scores11 = {}
        label_score = []
        score1={}
        t1 = []
        ff1=[]
        ta1 = {}
        x1={}
        dist1={}
        test_auc={}
        print("test:")
        lt=[]
        ll=[]
        ltt=[]
        tt1=[]
       
        Encoder().eval()
        Decoder().eval()
        Fc().eval()
        # Fc2().eval()
        
        enc = Encoder().to(self.device)
        dec = Decoder().to(self.device)
        fc = Fc().to(self.device)
        fc2=Fc2().to(self.device)

        file = 'data'
        table = 's5'
    

        X1=sio.loadmat('/home/htu/workspace/wyy/SWDAE/data/%s.mat'%(file),)['%s'%(table)]

        X=np.float32(X1)
        a = torch.tensor(X)
        xx=[a[i*50:(i+1)*50,:] for i in range(len(a)//50)]
        with torch.no_grad():
            tq = tqdm(xx, total=len(xx))
            leen=4
            for x in tq:
                print("test")
                x = x.to(self.device)
                # f1=fc2(x).to(self.device)
                x1,ff,rs= net(x)
                # for i in range(300):
                
                t1.append(x1.cpu().detach())
                for ki in range(12):
                    ff[ki] = torch.mean(ff[ki],dim=1)
                ff2 = [ff[i+4].reshape([50,-1])for i in range(8)]
                ff1=torch.cat(ff2[0:8],1)
                # input.unsqueeze_(1)
                for t in ff1:
                    lt.append(t) 
            t1= torch.tensor([item.cpu().detach().numpy() for item in t1]).cuda()
            ltt= torch.tensor([item.cpu().detach().numpy() for item in lt]).cuda()
            # ff1= torch.tensor([i.cpu().detach().numpy() for i in ff1]).cuda()
            tensor_04 = t1.reshape([t1.size()[-1], -1]).T
            xball = tensor_04.cpu().detach().numpy()
            internal = ltt.cpu().detach().numpy()
            x_enc = enc(tensor_04)
            fea = fc(x_enc)
            fea1  = fea.cpu().detach().numpy()
            
            sio.savemat('/home/htu/workspace/wyy/2/out/{}_{}_wtae.mat'.format(file,table), {'tae':fea1})
            sio.savemat('/home/htu/workspace/wyy/2/out/{}_{}_xb.mat'.format(file,table), {'xball':xball})
            sio.savemat('/home/htu/workspace/wyy/2/out/{}_{}_{}_xbfour.mat'.format(file,table,beta), {'fourball':internal})
            print("over")
        return 1



