"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from CTAB_GAN_Plus.model.pipeline.data_preparation import DataPrep
from CTAB_GAN_Plus.model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(self,
                 df,
                 metadata=None,
                 meta=None,
                 test_ratio = 0.20,
                 categorical_columns = [],
                 log_columns = [],
                 mixed_columns= {},
                 general_columns = [],
                 non_categorical_columns = [],
                 integer_columns = [],
                 problem_type= {},
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size_ctab=500,
                 epochs=150,
                 lr=2e-4,
                 device="cpu",
                 private=False,
                 sigma=None):

        if private:
            self.__name__ = f'DPCTABGAN{sigma}'
        else:
            self.__name__ = 'CTABGAN'
        self.multiprocess = False #True
        self.datatype = pd.DataFrame
        self.metadata = metadata
        
        if self.metadata != None:
            self.num_cols = []
            self.cat_cols = []
            for col in self.metadata['columns'][:-1]:
                if col['type'] == 'Integer' or col['type'] == 'Float':
                    self.num_cols.append(col['name'])
                elif col['type'] == 'Categorical' or col['type'] == 'Ordinal':
                    self.cat_cols.append(col['name'])
              
        self.synthesizer = CTABGANSynthesizer(
                class_dim=class_dim,
                random_dim=random_dim,
                num_channels=num_channels,
                l2scale=l2scale,
                lr=lr,
                batch_size=batch_size_ctab,
                epochs=epochs,
                device=device,
                private=private,
                sigma=sigma,
                meta=meta
        )
        self.raw_df = df
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        if 'Unsupervised' in problem_type.keys():
            self.problem_type = None
        else:
            self.problem_type = problem_type
                
    def fit(self, df=None):
        
        if type(df) == pd.DataFrame:
            self.raw_df = df #for stadler
        
        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
        general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"], type=self.problem_type)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self, num_samples, seed=0):
        
        sample = self.synthesizer.sample(num_samples, seed)
        if len(sample) == 0:
            return sample
        else:
            sample_df = self.data_prep.inverse_prep(sample)
            return sample_df