import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, BatchNorm2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss)
from model.synthesizer.transformer import ImageTransformer,DataTransformer
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
from dython.nominal import compute_associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore")


def cos_sim(real,fake):
    real_sampled  = real
    fake_sampled = fake

    scalerR = StandardScaler()
    scalerR.fit(real_sampled)
    scalerF = StandardScaler()
    scalerF.fit(fake_sampled)
    df_real_scaled = scalerR.transform(real_sampled)
    df_fake_scaled = scalerF.transform(fake_sampled)
   
    
    return (dot(real,fake.T)/(norm(real)*norm(fake))).mean()


def random_choice_prob_index_sampling(probs,col_idx): 
    option_list = []
    for i in col_idx:
        pp = probs[i] + 1e-6 
        pp = pp / sum(pp)      
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)

class Condvec(object):
    def __init__(self, data, output_info):
              
        self.model = []
        self.interval = []
        self.n_col = 0  
        self.n_opt = 0 
        self.p_log_sampling = []  
        self.p_sampling = [] 
         
        st = 0
        for item in output_info:          
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':       
                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                self.interval.append((self.n_opt, item[0]))
                self.n_col += 1
                self.n_opt += item[0]
                freq = np.sum(data[:, st:ed], axis=0)  
                log_freq = np.log(freq + 1)  
                log_pmf = log_freq / np.sum(log_freq)
                self.p_log_sampling.append(log_pmf)
                pmf = freq / np.sum(freq)
                self.p_sampling.append(pmf)
                st = ed        
        self.interval = np.asarray(self.interval)
        
    def sample_train(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
	vec = np.zeros((batch, self.n_opt), dtype='float32')        
        idx = np.random.choice(np.arange(self.n_col), batch)        
        mask = np.zeros((batch, self.n_col), dtype='float32')
        mask[np.arange(batch), idx] = 1  
        opt1prime = random_choice_prob_index_sampling(self.p_log_sampling,idx) 
        
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
       
        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None
   
        batch = batch 
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,idx)
 
        for i in np.arange(batch):   
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
        return vec

def cond_loss(data, output_info, c, m):
    tmp_loss = [] 
    st = 0
    st_c = 0

    for item in output_info:  
        if item[1] == 'tanh':
            st += item[0]
            continue
        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
            data[:, st:ed],
            torch.argmax(c[:, st_c:ed_c], dim=1),
            reduction='none')
            tmp_loss.append(tmp)
            st = ed
            st_c = ed_c

    tmp_loss = torch.stack(tmp_loss, dim=1)
    loss = (tmp_loss * m).sum() / data.size()[0]

    return loss

class Sampler(object):
    def __init__(self, data, output_info): 
        super(Sampler, self).__init__() 
        self.data = data
        self.model = []
        self.n = len(data)
        st = 0

        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
 
                for j in range(item[0]): 
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed  
    def sample(self, n, col, opt): 
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []

        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]

def get_st_ed(target_col_index,output_info):
    st = 0
    c= 0
    tc= 0
   
    for item in output_info:
        if c==target_col_index:
            break
        if item[1]=='tanh':
            st += item[0]
        elif item[1] == 'softmax':
            st += item[0]
            c+=1 
        tc+=1    

    ed= st+output_info[tc][0] 
    
    return (st,ed)

class Classifier(Module):
    def __init__(self,input_dim, class_dims,st_ed):
        super(Classifier,self).__init__()
        self.dim = input_dim-(st_ed[1]-st_ed[0])
        self.str_end = st_ed

        seq = []
        tmp_dim = self.dim
        for item in list(class_dims):
            seq += [
                Linear(tmp_dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            tmp_dim = item
  
        if (st_ed[1]-st_ed[0])==2:
            seq += [Linear(tmp_dim, 1),Sigmoid()]
        
        else: seq += [Linear(tmp_dim,(st_ed[1]-st_ed[0]))]   
        self.seq = Sequential(*seq)

    def forward(self, input):
        label = torch.argmax(input[:, self.str_end[0]:self.str_end[1]], axis=-1)
        new_imp = torch.cat((input[:,:self.str_end[0]],input[:,self.str_end[1]:]),1)

        if ((self.str_end[1]-self.str_end[0])==2):
            return self.seq(new_imp).view(-1), label
        else: return self.seq(new_imp), label

class Discriminator(Module):
    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:len(layers)-2])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)

class Generator(Module):   
    def __init__(self, layers):
        super(Generator, self).__init__()
        self.seq = Sequential(*layers)

    def forward(self, input):
        return self.seq(input)

def determine_layers_disc(side, num_channels):  
    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:

        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))
    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), 
        Sigmoid() 
    ]
    
    return layers_D

def determine_layers_gen(side, random_dim, num_channels):
    layer_dims = [(1, side), (num_channels, side // 2)]
    
    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))
    
  
    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]
    
   
    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]

    return layers_G

def apply_activate(data, output_info):
    data_t = []
    st = 0

    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        
        elif item[1] == 'softmax':
            ed = st + item[0]
   
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    
    act_data = torch.cat(data_t, dim=1) 

    return act_data

def weights_init(model): 
    classname = model.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0)

class BAMTGANSynthesizer:  
    def __init__(self,
                 class_dim=(512, 512, 512, 512),
                 random_dim=50,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=200,
                 epochs=1):
                 
        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = None

    def fit(self, train_data=pd.DataFrame, categorical=[], mixed={}, type={}):
        problem_type = None
        target_index = None
        
        if type:
            problem_type = list(type.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(type[problem_type])

        self.transformer = DataTransformer(train_data=train_data, categorical_list=categorical, mixed_dict=mixed)
        self.transformer.fit() 
        train_data = self.transformer.transform(train_data.values)
        
        data_dim = self.transformer.output_dim
        
        data_sampler = Sampler(train_data, self.transformer.output_info)

        self.cond_generator = Condvec(train_data, self.transformer.output_info)

		
        sides = [4, 8, 16, 24, 32]
       
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break
        
       	
        sides = [4, 8, 16, 24, 32]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break

        layers_G = determine_layers_gen(self.gside, self.random_dim+self.cond_generator.n_opt, self.num_channels)
        layers_D = determine_layers_disc(self.dside, self.num_channels)
        self.generator = Generator(layers_G).to(self.device)
        discriminator = Discriminator(layers_D).to(self.device)

        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        st_ed = None
        classifier=None
        optimizerC= None
        if target_index != None:
            st_ed= get_st_ed(target_index,self.transformer.output_info)
            classifier = Classifier(data_dim,self.class_dim,st_ed).to(self.device)
            optimizerC = optim.Adam(classifier.parameters(),**optimizer_params)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)       
        self.Dtransformer = ImageTransformer(self.dside)

        fake_cnt=1
        steps_per_epoch = max(1, len(train_data) // self.batch_size)
        for i in tqdm(range(self.epochs)):
            for _ in range(steps_per_epoch):
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample_train(self.batch_size)
                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)

                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
 
                perm = np.arange(self.batch_size)
                np.random.shuffle(perm)
                real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                real = torch.from_numpy(real.astype('float32')).to(self.device)

                real_dataframe = pd.DataFrame(real.numpy())
                
                c_perm = c[perm]
                
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fake_dataframe = pd.DataFrame(faket.detach().numpy())
              
                fakeact = apply_activate(faket, self.transformer.output_info)
                fakeact_dataframe = pd.DataFrame(fakeact.detach().numpy())

                cosine_similarity = cos_sim(real.numpy(),fakeact.detach().numpy())

                fake_cat = torch.cat([fakeact, c], dim=1)
               
                real_cat = torch.cat([real, c_perm], dim=1)

                real_cat_d = self.Dtransformer.transform(real_cat)
                fake_cat_d = self.Dtransformer.transform(fake_cat)

                optimizerD.zero_grad()
      
                y_real,_ = discriminator(real_cat_d)
               
                y_fake,_ = discriminator(fake_cat_d)
               
                loss_d = (-(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
           
                loss_d.backward()
           
                optimizerD.step()

                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample_train(self.batch_size)
                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

                optimizerG.zero_grad()

                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.output_info)

                fake_cat = torch.cat([fakeact, c], dim=1) 
                
                fake_cat = self.Dtransformer.transform(fake_cat)
                
                fake_cnt+=1

                y_fake,info_fake = discriminator(fake_cat)
          
                _,info_real = discriminator(real_cat_d)
                
                cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)

                g =  -0.5*(torch.log(y_fake + 1e-4).mean())+(cosine_similarity*1000)
    
                g.backward(retain_graph=True)
                
                loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1), dim=0) - torch.mean(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1), dim=0) - torch.std(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_info = loss_mean + loss_std 
              
                loss_info.backward()
        
                optimizerG.step()

                if problem_type:
                    c_loss = None
                    
                    if (st_ed[1] - st_ed[0])==2:
                        c_loss = BCELoss()
                 
                    else: c_loss = CrossEntropyLoss() 

                    optimizerC.zero_grad()
                    
                    real_pre, real_label = classifier(real)

                    if (st_ed[1] - st_ed[0])==2:
                        real_label = real_label.type_as(real_pre)

                    realact_list = []
                    for i in range(len(real_pre)):
                        if i<=torch.mean(real_pre):
                            realact_list.append(i)        
                    real_mean = torch.mean(real_pre.double()) - torch.mean(real_label.double())
                    loss_cc = c_loss(real_pre, real_label) +real_mean
                  
                    loss_cc.backward()
                    optimizerC.step()
                    optimizerG.zero_grad()
                   
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, self.transformer.output_info)
                    fake_pre, fake_label = classifier(fakeact)

                    fakeact_list = []
                    for i in range(len(fake_pre)):
                        if abs(fake_pre[i] - fake_label[i]) > torch.mean(fake_pre):
                            fakeact_list.append(i)
                  
                    fake_mean = torch.mean(fake_pre.double()) - torch.mean(fake_label.double()) 
          
                    if (st_ed[1] - st_ed[0])==2:
                        fake_label = fake_label.type_as(fake_pre)
                   
                    loss_cg = c_loss(fake_pre, fake_label) + fake_mean
                  
                    loss_cg.backward()
                    optimizerG.step()
                    
                            
    def sample(self, n):
        self.generator.eval()
        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for _ in range(steps):
            
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(self.batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket,output_info)
            data.append(fakeact.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)  
        result = self.transformer.inverse_transform(data)

        return result[0:n]
