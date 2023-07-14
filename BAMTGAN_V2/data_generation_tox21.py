import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep 
from model.synthesizer.bamtgan_synthesizer_tox21 import BAMTGANSynthesizer 

import warnings

warnings.filterwarnings("ignore")

class BAMTGAN():
    def __init__(self,
                 raw_csv_path = "./data/adult1000.csv",
                 test_ratio = 0.20,
                 categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": 'income'},
                 epochs = 1,num_records=20):

        self.__name__ = 'BAMTGAN'
        self.synthesizer = BAMTGANSynthesizer(epochs = epochs)      
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.num_records = num_records
        
    def fit(self):
        
        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], 
        mixed = self.data_prep.column_types["mixed"],type=self.problem_type)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")

    def generate_samples(self):
        aug_record_num = int(len(self.raw_df) *(1+(0.01*self.num_records)))
        print("증강 레코드 개수 :",aug_record_num)
        sample = self.synthesizer.sample(aug_record_num) 
        sample_df = self.data_prep.inverse_prep(sample)
       
        return sample_df
        
def data_generation(dataset,categorical_columns,mixed_columns,integer_columns,label):
 
    real_path = './record_clustering/'+str(dataset)+'_record_clustering.csv'
    num_records_percent_start = 100
    num_records_percent_end = 120
    next_percent = 20
    mixed_columns = {key: [0.0] for key in mixed_columns}

    synthesizer =  BAMTGAN(raw_csv_path = real_path,
                     test_ratio = 0.20,  
                    categorical_columns = categorical_columns, 
                     log_columns = [],
                     mixed_columns= mixed_columns, 
                     integer_columns = integer_columns,
                     problem_type= {"Classification": label},
                     epochs = 150,num_records=20) 
    

    for i in range(num_records_percent_start, num_records_percent_end, next_percent):
        synthesizer.fit()
        syn = synthesizer.generate_samples()
        syn.to_csv("./GAN/"+dataset+"_"+"GAN.csv".format(exp=i), index= False)