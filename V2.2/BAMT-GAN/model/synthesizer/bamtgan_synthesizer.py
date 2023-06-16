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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
warnings.filterwarnings("ignore")


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
        # for improved stability
        pp = probs[i] + 1e-6 
        pp = pp / sum(pp)
        # sampled based on given probability mass distribution of categories within the given one-hot-encoding 
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
        
        # iterating through the transformed input data columns 
        st = 0
        for item in output_info:
            # ignoring columns that do not represent one-hot-encodings
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                # using starting (st) and ending (ed) position of any given one-hot-encoded representation to obtain relevant information
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
        
        # each conditional vector in vec is a one-hot vector used to highlight a specific category across all possible one-hot-encoded representations 
        # (i.e., including modes of continuous and mixed columns)
        vec = np.zeros((batch, self.n_opt), dtype='float32')

        # choosing one specific one-hot-encoding from all possible one-hot-encoded representations 
        idx = np.random.choice(np.arange(self.n_col), batch)

        # matrix of shape (batch x total no. of one-hot-encoded representations) with 1 in indexes of chosen representations and 0 elsewhere
        mask = np.zeros((batch, self.n_col), dtype='float32')
        mask[np.arange(batch), idx] = 1  
        
        # producing a list of selected categories within each of selected one-hot-encoding representation
        opt1prime = random_choice_prob_index_sampling(self.p_log_sampling,idx) 
        
        # assigning the appropriately chosen category for each corresponding conditional vector
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
       
        return vec, mask, idx, opt1prime

    def sample(self, batch):
        
        """
        Used to create the conditional vectors for feeding it to the generator after training is finished

        Inputs:
        1) batch -> no. of data records to be generated in a batch

        Outputs:
        1) vec -> an array containing a conditional vector for each data point to be generated 
        """

        if self.n_col == 0:
            return None
        
        batch = batch

        # each conditional vector in vec is a one-hot vector used to highlight a specific category across all possible one-hot-encoded representations 
        # (i.e., including modes of continuous and mixed columns)
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        
        # choosing one specific one-hot-encoding from all possible one-hot-encoded representations 
        idx = np.random.choice(np.arange(self.n_col), batch)

        # producing a list of selected categories within each of selected one-hot-encoding representation
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,idx)
        
        # assigning the appropriately chosen category for each corresponding conditional vector
        for i in np.arange(batch):   
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

   
        return vec

def cond_loss(data, output_info, c, m):
    
    # used to store cross entropy loss between conditional vector and all generated one-hot-encodings
    tmp_loss = []
    # counter to iterate generated data columns
    st = 0
    # counter to iterate conditional vector
    st_c = 0
    # iterating through column information
    for item in output_info:
        # ignoring numeric columns
        if item[1] == 'tanh':
            st += item[0]
            continue
        # computing cross entropy loss between generated one-hot-encoding and corresponding encoding of conditional vector
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

    # computing the loss across the batch only and only for the relevant one-hot-encodings by applying the mask 
    tmp_loss = torch.stack(tmp_loss, dim=1)
    loss = (tmp_loss * m).sum() / data.size()[0]

    return loss

class Sampler(object):
    
    """
    This class is used to sample the transformed real data according to the conditional vector 

    Variables:
    1) data -> real transformed input data
    2) model -> stores the index values of data records corresponding to any given selected categories for all columns
    3) n -> size of the input data

    Methods:
    1) __init__() -> initiates the sampler object and stores class variables 
    2) sample() -> takes as input the number of rows to be sampled (n), chosen column (col)
                   and category within the column (opt) to sample real records accordingly
    """

    def __init__(self, data, output_info):
        
        super(Sampler, self).__init__()
        
        self.data = data
        self.model = []
        self.n = len(data)
        
        # counter to iterate through columns
        st = 0
        # iterating through column information
        for item in output_info:
            # ignoring numeric columns
            if item[1] == 'tanh':
                st += item[0]
                continue
            # storing indices of data records for all categories within one-hot-encoded representations
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                # iterating through each category within a one-hot-encoding
                for j in range(item[0]):
                    # storing the relevant indices of data records for the given categories
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed
                
    def sample(self, n, col, opt):
        
        # if there are no one-hot-encoded representations, we may ignore sampling using a conditional vector
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        
        # used to store relevant indices of data records based on selected category within a chosen one-hot-encoding
        idx = []
        
        # sampling a data record index randomly from all possible indices that meet the given criteria of the chosen category and one-hot-encoding
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        
        return self.data[idx]

def get_st_ed(target_col_index,output_info):
    
    """
    Used to obtain the start and ending positions of the target column as per the transformed data to be used by the classifier 

    Inputs:
    1) target_col_index -> column index of the target column used for machine learning tasks (binary/multi-classification) in the raw data 
    2) output_info -> column information corresponding to the data after applying the data transformer

    Outputs:
    1) starting (st) and ending (ed) positions of the target column as per the transformed data
    
    """
    # counter to iterate through columns
    st = 0
    # counter to check if the target column index has been reached
    c= 0
    # counter to iterate through column information
    tc= 0
    # iterating until target index has reached to obtain starting position of the one-hot-encoding used to represent target column in transformed data
    for item in output_info:
        # exiting loop if target index has reached
        if c==target_col_index:
            break
        if item[1]=='tanh':
            st += item[0]
        elif item[1] == 'softmax':
            st += item[0]
            c+=1 
        tc+=1    
    
    # obtaining the ending position by using the dimension size of the one-hot-encoding used to represent the target column
    ed= st+output_info[tc][0] 
    
    return (st,ed)
'''
class VotingClassifier(Module):
    """
    This class represents the ensemble of classifiers used for voting the final output of the generator

    Variables:
    1) classifiers -> list of classifier objects which will be used for voting
    2) voting_type -> 'hard' or 'soft' voting based on whether the ensemble uses class predictions or class probabilities

    Methods:
    1) __init__() -> initializes and builds the layers of the classifier module 
    2) forward() -> executes the forward pass of the classifier module on the corresponding input data and
                    outputs the predictions for the generator output using voting\
    
    """
    
    def __init__(self,input_dim, class_dims,st_ed):
        super(VotingClassifier, self).__init__()
       
        
       # subtracting the target column size from the input dimensionality 
        self.dim = input_dim-(st_ed[1]-st_ed[0])
        # storing the starting and ending positons of the target column in the input data
        self.str_end = st_ed
        
        # building the layers of the network with same hidden layers as discriminator
        seq = []
        tmp_dim = self.dim
        for item in list(class_dims):
            seq += [
                Linear(tmp_dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            tmp_dim = item
        
        # in case of binary classification the last layer outputs a single numeric value which is squashed to a probability with sigmoid
        if (st_ed[1]-st_ed[0])==2:
            seq += [Linear(tmp_dim, 1),Sigmoid()]
        # in case of multi-classs classification, the last layer outputs an array of numeric values associated to each class
        else: seq += [Linear(tmp_dim,(st_ed[1]-st_ed[0]))] 
            
        self.seq = Sequential(*seq)

    def forward(self, input):
        
        # true labels obtained from the input data
        label = torch.argmax(input[:, self.str_end[0]:self.str_end[1]], axis=-1)
        
        # input to be fed to the classifier module
        new_imp = torch.cat((input[:,:self.str_end[0]],input[:,self.str_end[1]:]),1)
        
        # returning predictions and true labels for binary/multi-class classification 
        if ((self.str_end[1]-self.str_end[0])==2):
            return self.seq(new_imp).view(-1), label
        else: return self.seq(new_imp), label
'''    
class Classifier(Module):

    """
    This class represents the classifier module used along side the discriminator to train the generator network

    Variables:
    1) dim -> column dimensionality of the transformed input data after removing target column
    2) class_dims -> list of dimensions used for the hidden layers of the classifier network
    3) str_end -> tuple containing the starting and ending positions of the target column in the transformed input data

    Methods:
    1) __init__() -> initializes and builds the layers of the classifier module 
    2) forward() -> executes the forward pass of the classifier module on the corresponding input data and
                    outputs the predictions and corresponding true labels for the target column 
    
    """
    
    def __init__(self,input_dim, class_dims,st_ed):
        super(Classifier,self).__init__()
        # subtracting the target column size from the input dimensionality 
        self.dim = input_dim-(st_ed[1]-st_ed[0])
        # storing the starting and ending positons of the target column in the input data
        self.str_end = st_ed
        
        # building the layers of the network with same hidden layers as discriminator
        seq = []
        tmp_dim = self.dim
        for item in list(class_dims):
            seq += [
                Linear(tmp_dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            tmp_dim = item
        
        # in case of binary classification the last layer outputs a single numeric value which is squashed to a probability with sigmoid
        if (st_ed[1]-st_ed[0])==2:
            seq += [Linear(tmp_dim, 1),Sigmoid()]
        # in case of multi-classs classification, the last layer outputs an array of numeric values associated to each class
        else: seq += [Linear(tmp_dim,(st_ed[1]-st_ed[0]))] 
            
        self.seq = Sequential(*seq)

    def forward(self, input):
        
        # true labels obtained from the input data
        label = torch.argmax(input[:, self.str_end[0]:self.str_end[1]], axis=-1)
        
        # input to be fed to the classifier module
        new_imp = torch.cat((input[:,:self.str_end[0]],input[:,self.str_end[1]:]),1)
        
        # returning predictions and true labels for binary/multi-class classification 
        if ((self.str_end[1]-self.str_end[0])==2):
            return self.seq(new_imp).view(-1), label
        else: return self.seq(new_imp), label

class Discriminator(Module):

    """
    This class represents the discriminator network of the model

    Variables:
    1) seq -> layers of the network used for making the final prediction of the discriminator model
    2) seq_info -> layers of the discriminator network used for computing the information loss

    Methods:
    1) __init__() -> initializes and builds the layers of the discriminator model
    2) forward() -> executes a forward pass on the input data to output the final predictions and corresponding 
                    feature information associated with the penultimate layer used to compute the information loss 
    
    """
    
    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:len(layers)-2])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)

class Generator(Module):
    
    """
    This class represents the discriminator network of the model
    
    Variables:
    1) seq -> layers of the network used by the generator

    Methods:
    1) __init__() -> initializes and builds the layers of the generator model
    2) forward() -> executes a forward pass using noise as input to generate data 

    """
    
    def __init__(self, layers):
        super(Generator, self).__init__()
        self.seq = Sequential(*layers)

    def forward(self, input):
        return self.seq(input)

def determine_layers_disc(side, num_channels):
    
    """
    This function describes the layers of the discriminator network as per DCGAN (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

    Inputs:
    1) side -> height/width of the input fed to the discriminator
    2) num_channels -> no. of channels used to decide the size of respective hidden layers 

    Outputs:
    1) layers_D -> layers of the discriminator network
    
    """
    if side is None:
        # side가 None인 경우를 대비한 예외 처리
        side = 1  # 이 값은 상황에 따라 적절히 조정해야 합니다.
    # computing the dimensionality of hidden layers 
    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        # the number of channels increases by a factor of 2 whereas the height/width decreases by the same factor with each layer
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    # constructing the layers of the discriminator network based on the recommendations mentioned in https://arxiv.org/abs/1511.06434 
    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    # last layer reduces the output to a single numeric value which is squashed to a probabability using sigmoid function
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), 
        Sigmoid() 
    ]
    
    return layers_D

def determine_layers_gen(side, random_dim, num_channels):
    
    """
    This function describes the layers of the generator network
    
    Inputs:
    1) random_dim -> height/width of the noise matrix to be fed for generation 
    2) num_channels -> no. of channels used to decide the size of respective hidden layers

    Outputs:
    1) layers_G -> layers of the generator network

    """
    
    # computing the dimensionality of hidden layers
    layer_dims = [(1, side), (num_channels, side // 2)]
    
    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))
    
    # similarly constructing the layers of the generator network based on the recommendations mentioned in https://arxiv.org/abs/1511.06434 
    # first layer of the generator takes the channel dimension of the noise matrix to the desired maximum channel size of the generator's layers 
    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]
    
    # the following layers are then reversed with respect to the discriminator 
    # such as the no. of channels reduce by a factor of 2 and height/width of generated image increases by the same factor with each layer 
    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]

    return layers_G

def apply_activate(data, output_info):
    
    """
    This function applies the final activation corresponding to the column information associated with transformer

    Inputs:
    1) data -> input data generated by the model in the same format as the transformed input data
    2) output_info -> column information associated with the transformed input data

    Outputs:
    1) act_data -> resulting data after applying the respective activations 

    """
    
    data_t = []
    # used to iterate through columns
    st = 0
    # used to iterate through column information
    for item in output_info:
        # for numeric columns a final tanh activation is applied
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        # for one-hot-encoded columns, a final gumbel softmax (https://arxiv.org/pdf/1611.01144.pdf) is used 
        # to sample discrete categories while still allowing for back propagation 
        elif item[1] == 'softmax':
            ed = st + item[0]
            # note that as tau approaches 0, a completely discrete one-hot-vector is obtained
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    
    act_data = torch.cat(data_t, dim=1) 

    return act_data

def weights_init(model):
    
    """
    This function initializes the learnable parameters of the convolutional and batch norm layers

    Inputs:
    1) model->  network for which the parameters need to be initialized
    
    Outputs:
    1) network with corresponding weights initialized using the normal distribution
    
    """
    
    classname = model.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0)

class BAMTGANSynthesizer:

    """
    This class represents the main model used for training the model and generating synthetic data
    """
    
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

        
        print("------하이퍼파라미터------")
        print("random_dim:",random_dim)
        print("class_dim:",class_dim)
        print("num_channels:",num_channels)
        print("bach_size:",batch_size)
        print("epochs:",epochs)
        
    def fit(self, train_data=pd.DataFrame, categorical=[], mixed={}, type={}):
        
        # obtaining the column index of the target column used for ML tasks
        problem_type = None
        target_index = None
        
        if type:
            problem_type = list(type.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(type[problem_type])

        # transforming pre-processed training data according to different data types 
        # i.e., mode specific normalisation for numeric and mixed columns and one-hot-encoding for categorical columns
        self.transformer = DataTransformer(train_data=train_data, categorical_list=categorical, mixed_dict=mixed)
        self.transformer.fit() 
        train_data = self.transformer.transform(train_data.values)
        # storing column size of the transformed training data
        data_dim = self.transformer.output_dim
        
        # initializing the sampler object to execute training-by-sampling 
        data_sampler = Sampler(train_data, self.transformer.output_info)
        # initializing the condvec object to sample conditional vectors during training
        self.cond_generator = Condvec(train_data, self.transformer.output_info)

        # obtaining the desired height/width for converting tabular data records to square images for feeding it to discriminator network 		
        sides = [4, 8, 16, 24, 32]
        # the discriminator takes the transformed training data concatenated by the corresponding conditional vectors as input
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break
        
        # obtaining the desired height/width for generating square images from the generator network that can be converted back to tabular domain 		
        sides = [4, 8, 16, 24, 32]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break
		
        # constructing the generator and discriminator networks
        layers_G = determine_layers_gen(self.gside, self.random_dim+self.cond_generator.n_opt, self.num_channels)
        layers_D = determine_layers_disc(self.dside, self.num_channels)
        self.generator = Generator(layers_G).to(self.device)
        discriminator = Discriminator(layers_D).to(self.device)
        
        # assigning the respective optimizers for the generator and discriminator networks
        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

       
        st_ed = None
        classifier=None
        optimizerC= None
        
        
        if target_index != None:
            # obtaining the one-hot-encoding starting and ending positions of the target column in the transformed data
            st_ed= get_st_ed(target_index,self.transformer.output_info)
            # configuring the classifier network and it's optimizer accordingly 
         
            
            classifier = Classifier(data_dim,self.class_dim,st_ed).to(self.device)
            #classifier = Classifier(data_dim,self.class_dim,st_ed).to(self.device)
            optimizerC = optim.Adam(classifier.parameters(),**optimizer_params)
        
        # initializing learnable parameters of the discrimnator and generator networks  
        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        # initializing the image transformer objects for the generator and discriminator networks for transitioning between image and tabular domain 
        self.Gtransformer = ImageTransformer(self.gside)       
        self.Dtransformer = ImageTransformer(self.dside)
        
        # initiating the training by computing the number of iterations per epoch
        fake_cnt=1
        steps_per_epoch = max(1, len(train_data) // self.batch_size)
        for i in tqdm(range(self.epochs)):
            for _ in range(steps_per_epoch):
                
                # sampling noise vectors using a standard normal distribution 
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                # sampling conditional vectors 
                condvec = self.cond_generator.sample_train(self.batch_size)
                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)
                # concatenating conditional vectors and converting resulting noise vectors into the image domain to be fed to the generator as input
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

                # sampling real data according to the conditional vectors and shuffling it before feeding to discriminator to isolate conditional loss on generator    
                perm = np.arange(self.batch_size)
                np.random.shuffle(perm)
                real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                real = torch.from_numpy(real.astype('float32')).to(self.device)
                
                #print("real:",real)
                #print("real shape",real.shape)
                print("-------real data vector-----")
                real_dataframe = pd.DataFrame(real.numpy())
                #print(real_dataframe)
                
                # storing shuffled ordering of the conditional vectors
                c_perm = c[perm]
                # generating synthetic data as an image
                fake = self.generator(noisez)
          
                # converting it into the tabular domain as per format of the trasformed training data
                faket = self.Gtransformer.inverse_transform(fake)
                fake_dataframe = pd.DataFrame(faket.detach().numpy())
                #print("-------fake data vector-----")
                #print(fake_dataframe)
                # applying final activation on the generated data (i.e., tanh for numeric and gumbel-softmax for categorical)
                fakeact = apply_activate(faket, self.transformer.output_info)
                
                
                fakeact_dataframe = pd.DataFrame(fakeact.detach().numpy())
                #print("-------fakeact vector-----")
                #print(fakeact_dataframe)
                
                
                #유사도 계산
                print("cosine similarity",cos_sim(real.numpy(),fakeact.detach().numpy()))
                cosine_similarity = cos_sim(real.numpy(),fakeact.detach().numpy())
                                            
                #print("fakeect",fakeact)
                #print("fakeect",fakeact.shape)
                # the generated data is then concatenated with the corresponding condition vectors 
                fake_cat = torch.cat([fakeact, c], dim=1)
                # the real data is also similarly concatenated with corresponding conditional vectors    
                real_cat = torch.cat([real, c_perm], dim=1)
                
                # transforming the real and synthetic data into the image domain for feeding it to the discriminator
                #-------------------테이블 데이터 24X24 로 변환
                real_cat_d = self.Dtransformer.transform(real_cat)
                fake_cat_d = self.Dtransformer.transform(fake_cat)
                #print("real_cat_d",real_cat_d)
                #print("real_cat_d shape",real_cat_d.shape)
                
                #print("fake_cat_d",fake_cat_d)
                #print("fake_cat_d shape",fake_cat_d.shape)
                
                # executing the gradient update step for the discriminator    
                optimizerD.zero_grad()
                # computing the probability of the discriminator to correctly classify real samples hence y_real should ideally be close to 1
                y_real,_ = discriminator(real_cat_d)
                # computing the probability of the discriminator to correctly classify fake samples hence y_fake should ideally be close to 0
                y_fake,_ = discriminator(fake_cat_d)
                # computing the loss to essentially maximize the log likelihood of correctly classifiying real and fake samples as log(D(x))+log(1−D(G(z)))
                # or equivalently minimizing the negative of log(D(x))+log(1−D(G(z))) as done below
                #loss_d = (-(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
                loss_d = (-(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
                # accumulating gradients based on the loss
                loss_d.backward()
                # computing the backward step to update weights of the discriminator
                optimizerD.step()

                # similarly sample noise vectors and conditional vectors
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample_train(self.batch_size)
                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

                # executing the gradient update step for the generator    
                optimizerG.zero_grad()

                # similarly generating synthetic data and applying final activation
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.output_info)
                # concatenating conditional vectors and converting it to the image domain to be fed to the discriminator
                fake_cat = torch.cat([fakeact, c], dim=1) 
                
                fake_cat = self.Dtransformer.transform(fake_cat)
                
                fake_cnt+=1

               

                # computing the probability of the discriminator classifiying fake samples as real 
                # along with feature representaions of fake data resulting from the penultimate layer 
                y_fake,info_fake = discriminator(fake_cat)
                # extracting feature representation of real data from the penultimate layer of the discriminator 
                _,info_real = discriminator(real_cat_d)
                # computing the conditional loss to ensure the generator generates data records with the chosen category as per the conditional vector
                cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)
                print("cross_entropy:",cross_entropy)
                print("cosine_similarity:",cosine_similarity)
                # computing the loss to train the generator where we want y_fake to be close to 1 to fool the discriminator 
                # and cross_entropy to be close to 0 to ensure generator's output matches the conditional vector  
                 
                #g = -(torch.log(y_fake + 1e-4).mean()) + cross_entropy
                g =  -0.5*(torch.log(y_fake + 1e-4).mean())+(cosine_similarity*1000)
                #g = -0.5*((torch.log(y_fake + 1e-4).mean())+cross_entropy)+(cosine_similarity*10000)**2
                # in order to backprop the gradient of separate losses w.r.t to the learnable weight of the network independently
                # we may use retain_graph=True in backward() method in the first back-propagated loss 
                # to maintain the computation graph to execute the second backward pass efficiently
                g.backward(retain_graph=True)
                # computing the information loss by comparing means and stds of real/fake feature representations extracted from discriminator's penultimate layer
                loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1), dim=0) - torch.mean(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1), dim=0) - torch.std(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_info = loss_mean + loss_std 
                # computing the finally accumulated gradients
                loss_info.backward()
                # executing the backward step to update the weights
                optimizerG.step()

                # the classifier module is used in case there is a target column associated with ML tasks 
                if problem_type:
                    
                    c_loss = None
                    # in case of binary classification, the binary cross entropy loss is used 
                    if (st_ed[1] - st_ed[0])==2:
                        c_loss = BCELoss()
                    # in case of multi-class classification, the standard cross entropy loss is used
                    else: c_loss = CrossEntropyLoss() 
                    
                    # updating the weights of the classifier
                    optimizerC.zero_grad()
                    # computing classifier's target column predictions on the real data along with returning corresponding true labels
                    real_pre, real_label = classifier(real)
                    
                    
                    
                    if (st_ed[1] - st_ed[0])==2:
                        real_label = real_label.type_as(real_pre)
                    # computing the loss to train the classifier so that it can perform well on the real data
                    
               
                    realact_list = []
                    for i in range(len(real_pre)):
                        if i<=torch.mean(real_pre):
                            realact_list.append(i)
                           
                    real_mean = torch.mean(real_pre.double()) - torch.mean(real_label.double())
                    loss_cc = c_loss(real_pre, real_label) +real_mean
                    
                    loss_cc.backward()
                    optimizerC.step()
                    
                    # updating the weights of the generator
                    optimizerG.zero_grad()
                    # generate synthetic data and apply the final activation
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, self.transformer.output_info)
                    # computing classifier's target column predictions on the fake data along with returning corresponding true labels
                    fake_pre, fake_label = classifier(fakeact)
                    
                  
                    
                 
                    fakeact_list = []
                    for i in range(len(fake_pre)):
                        if abs(fake_pre[i] - fake_label[i]) > torch.mean(fake_pre):
                            fakeact_list.append(i)
             
                    fake_mean = torch.mean(fake_pre.double()) - torch.mean(fake_label.double()) 
                    #fake_pre,fake_label = fake_pre[fakeact_list],fake_label[fakeact_list]
                    
                    if (st_ed[1] - st_ed[0])==2:
                        fake_label = fake_label.type_as(fake_pre)
                    # computing the loss to train the generator to improve semantic integrity between target column and rest of the data
                    loss_cg = c_loss(fake_pre, fake_label) + fake_mean
                  
                    loss_cg.backward()
                    optimizerG.step()
                    
                            
    def sample(self, n):
        
        # turning the generator into inference mode to effectively use running statistics in batch norm layers
        self.generator.eval()
        # column information associated with the transformer fit to the pre-processed training data
        output_info = self.transformer.output_info
        print("output_info:",output_info)
        # generating synthetic data in batches accordingly to the total no. required
        steps = n // self.batch_size + 1
        print("steps:",steps)
        
        data = []
        batch_data = []
        logistic_regression = LogisticRegression()
        knn = KNeighborsClassifier(n_neighbors=8)
        dt = DecisionTreeClassifier()
        svm = SVC(C=0.1,gamma='auto', probability=True)
        
        classifiers = [logistic_regression, knn,dt,svm]

        for _ in range(steps):
            # generating synthetic data using sampled noise and conditional vectors
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
            
            batch_data= np.concatenate(data, axis=0)
            batch_result = self.transformer.inverse_transform(batch_data)
            voting_model = VotingClassifier(estimators=[('LogisticRegression', logistic_regression),
                                            ('KNN', knn), ('DT', dt),
                                            ('SVM', svm)], voting='soft')

            result_df = pd.DataFrame(batch_result)    
            X = result_df.iloc[:,:-1]
            y = result_df.iloc[:,-1]

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)


            for classifier in classifiers:
                classifier.fit(X_train, y_train)
                pred = classifier.predict(X_test)
                class_name = classifier.__class__.__name__
                print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_test, pred)))

            voting_model.fit(X_train, y_train)
            pred = voting_model.predict(X_test)
            print('보팅 분류기의 정확도: {0: .4f}'.format(accuracy_score(y_test, pred)))
            
        data = np.concatenate(data, axis=0)
        
       
        
        
        result = self.transformer.inverse_transform(data)
        
      

   
        return result[0:n]
