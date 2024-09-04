import tempfile
import json
import torch
import lib

import pandas as pd
import numpy as np

from CTGAN.CTGAN.ctgan.synthesizers.tvae import TVAESynthesizer
from CTGAN.train_sample_tvae import train_tvae, sample_rec_tvae
from synthetic_data_release.generative_models.data_synthesiser import (IndependentHistogram,
                                                BayesianNet,
                                                PrivBayes)
from CTAB_GAN_Plus.train_sample_ctabganp import train_ctabgan, sample_ctabgan
from tab_ddpm.train import train_ddpm
from tab_ddpm.sample import sample_ddpm
from pathlib import Path

def run_AEframework(dir_, real_data_path, tvae_params, GenModel, params, n_samples, n_samplesets, device):
    
    #output dir_ with 1 synthetic dataset <- tune_models
    #output 1 synthetic dataset (as dataframe) <- mia_attack (make another script)
        
    dir_ = Path(dir_)
        
    with tempfile.TemporaryDirectory() as latent_dir_:
        
        if 'tvae' in GenModel:
            
            print('tvae')
            
            tvae = train_tvae(
            parent_dir=dir_,
            real_data_path=real_data_path,
            latent_dir=latent_dir_,
            train_params=tvae_params,
            change_val=False,
            device=device,
            ae=False
            ) #save noise
            
            latent = pd.DataFrame(np.load(latent_dir_ + '/X_num_train.npy'))
            latent.columns = [str(x) for x in range(len(latent.columns))]
            
            print('output of tvae')
            print(latent.shape)
            
        elif 'ae' in GenModel:
            
            print('ae')
            
            tvae = train_tvae(
            parent_dir=dir_,
            real_data_path=real_data_path,
            latent_dir=latent_dir_,
            train_params=tvae_params,
            change_val=False,
            device=device,
            ae=True
            ) #save features
        
            latent = pd.DataFrame(np.load(latent_dir_ + '/X_num_train.npy'))
            latent.columns = [str(x) for x in range(len(latent.columns))]
            
            print('output of ae')
            print(latent)
            print(latent.shape)
            print(latent.columns)
        
        if GenModel in ['baynet', 'privbay', 'aebaynet', 'aeprivbay', 'tvaebaynet', 'tvaeprivbay', 'privaebaynet', 'privtvaebaynet', 'privaeprivbay']:
            
            if 'ae' not in GenModel:
                
                print('baynet/privbay')
                
                with open(real_data_path + '/meta.json', "r") as json_file:
                    meta = json.load(json_file)
            
                if 'diabetes' in real_data_path or 'mimic_large' in real_data_path:
                    num, _, y, _, _, _ = lib.read_changed_val(real_data_path)
                    data = np.concatenate([num, y.reshape(-1, 1)], axis=1, dtype=object)
                else:
                    num, cat, y, _, _, _ = lib.read_changed_val(real_data_path)
                    data = np.concatenate([num, cat, y.reshape(-1, 1)], axis=1, dtype=object)
                
                data = pd.DataFrame(data)
                data.columns = [column["name"] for column in meta["columns"]]
                
                cat_cols = [column["name"] for column in meta["columns"] if column["type"] in ["Ordinal", "Categorical"]]
                num_cols = [column["name"] for column in meta["columns"] if column["type"] in ["Integer", "Float"]]
                data[cat_cols] = data[cat_cols].astype(str)
                
                print('MEEEEEETA')
                print(meta)
            
                if 'baynet' in GenModel:
                    baynet = BayesianNet(meta, histogram_bins=params['histogram_bins'], degree=params['degree'])
                elif 'privbay' in GenModel:
                    baynet = PrivBayes(meta, epsilon=params['epsilon'], histogram_bins=params['histogram_bins'], degree=params['degree'])
                
                baynet.fit(data)
                
                samples = []
                for i in range(n_samplesets):
                    gen_data = baynet.generate_samples(n_samples)
                    samples.append(gen_data)
                    
                #####
                
                '''sample = gen_data
                
                y = sample[sample.columns[-1]].values
                if len(np.unique(y)) == 1:
                    y[0] = 0
                    y[1] = 1
            
                if 'diabetes' in real_data_path or 'mimic_large' in real_data_path:
                    X_cat = None
                    X_num = sample[num_cols].values
                else:
                    X_cat = sample[cat_cols].drop(sample.columns[-1], axis=1, errors="ignore").values
                    X_num = sample[num_cols].values
            
                if X_num is not None:
                    np.save(dir_ / 'X_num_train', X_num.astype(float))
                if X_cat is not None:
                    np.save(dir_ / 'X_cat_train', X_cat.astype(str))
                y = y.astype(float)
                y = y.astype(int)
                np.save(dir_ / 'y_train', y)'''
                
                #####
                    
                return samples
            
            else:
            
                print('aebaynet/aeprivbay/tvaebaynet/tvaeprivbay')
                
                info = []
                for col in latent.columns:
                    curr = {'name': str(col), "type": "Float", "min": float(latent[col].min()), "max": float(latent[col].max())}
                    info.append(curr)
                meta = {
                    "columns": info
                }
                
                if 'baynet' in GenModel:
                    baynet = BayesianNet(meta, histogram_bins=params['histogram_bins'], degree=params['degree'])
                elif 'privbay' in GenModel:
                    baynet = PrivBayes(meta, epsilon=params['epsilon'], histogram_bins=params['histogram_bins'], degree=params['degree'])
                
                baynet.fit(latent)
                
                latent_list = []
                for i in range(n_samplesets):
                    latent = baynet.generate_samples(n_samples)
                    latent_list.append(latent)
            
        elif GenModel in ['ctabgan', 'dpctabgan', 'aectabgan', 'aedpctabgan', 'tvaectabgan', 'tvaedpctabgan', 'privaectabgan', 'privtvaectabgan']:
            
            if 'ae' in GenModel:
                
                print('aectabgan/aedpctabgan/tvaectabgan/tvaedpctabgan')
                
                with open('CTAB_GAN_Plus/columns.json', 'r') as json_file:
                    column_info = json.load(json_file)
                
                info = {'categorical_columns': [],
                        'mixed_columns': {},
                        'general_columns': [],
                        'integer_columns': [],
                        'problem_type': {'Unsupervised': None}
                }
                for col in latent.columns:
                    info['integer_columns'].append(str(col))
                    
                column_info['latent'] = info
                    
                with open('CTAB_GAN_Plus/columns.json', 'w') as json_file:
                    json.dump(column_info, json_file, indent=4)
                    
                meta = []
                for col in latent.columns:
                    curr = {"name": col, "type": "continuous", "min": float(latent[col].min()),"max": float(latent[col].max())}
                    meta.append(curr)
                    
                ctabgan = train_ctabgan(
                        parent_dir=dir_,
                        real_data_path=latent_dir_,
                        train_params=params,
                        change_val=False,
                        device=device,
                        meta_data=meta
                    )
                    
                latent_list = []
                
                for i in range(n_samplesets):
                    latent = sample_ctabgan(
                        ctabgan,
                        parent_dir=dir_,
                        real_data_path=latent_dir_,
                        num_samples=n_samples,
                        train_params=params,
                        change_val=False,
                        seed=i,
                        device=device)
                        
                    latent = pd.DataFrame(np.load(str(dir_) + '/X_num_train.npy'))
                    latent.columns = [str(x) for x in range(len(latent.columns))]
                    
                    latent_list.append(latent)
                    
            else:
                
                print('ctabgan/dpctabgan')
                
                with open(real_data_path + '/meta_ctab.json', "r") as json_file:
                    meta = json.load(json_file)
                
                ctabgan = train_ctabgan(
                        parent_dir=dir_,
                        real_data_path=real_data_path,
                        train_params=params,
                        change_val=False,
                        device=device,
                        meta_data=meta
                    )
                    
                samples = []
                for i in range(n_samplesets):
                    gen_data = sample_ctabgan(
                        ctabgan,
                        parent_dir=dir_,
                        real_data_path=real_data_path,
                        num_samples=n_samples,
                        train_params=params,
                        change_val=False,
                        seed=i,
                        device=device)
                    samples.append(gen_data)
                    
                return samples
                
        elif GenModel in ['ddpm', 'aeddpm', 'tvaeddpm', 'privaeddpm', 'privtvaeddpm']:
            
            T_dict = {"seed": 0, "normalization": "quantile", "num_nan_policy": None, "cat_nan_policy": None, "cat_min_frequency": None, "cat_encoding": None, "y_policy": "default"}
            
            if 'ae' in GenModel:
                
                print('aeddpm/tvaeddpm')
            
                ddpm = train_ddpm(
                        parent_dir=dir_,
                        real_data_path=latent_dir_,
                        steps=params['steps_syn'],
                        lr=params['lr'],
                        batch_size=params['batch_size'],
                        model_params=params['model_params'],
                        num_timesteps=params['num_timesteps'],
                        gaussian_loss_type=params['gaussian_loss_type'],
                        scheduler=params['scheduler'],
                        T_dict=T_dict,
                        device=device,
                        change_val=False
                    )
                    
                latent_list = []
                for i in range(n_samplesets):
                    sample_ddpm(
                        parent_dir=dir_,
                        real_data_path=latent_dir_,
                        batch_size=params['batch_size'],
                        num_samples=n_samples,
                        model_type='mlp',
                        model_params=params['model_params'],
                        model_path=str(dir_)+'/model.pt',
                        num_timesteps=params['num_timesteps'],
                        gaussian_loss_type=params['gaussian_loss_type'],
                        scheduler=params['scheduler'],
                        num_numerical_features=len(latent.columns),
                        T_dict=T_dict,
                        device=device,
                        seed=i,
                        change_val=False)
                    latent = np.load(str(dir_)+'/X_num_train.npy')
                    y = np.load(str(dir_) + '/y_train.npy')
                    latent = pd.DataFrame(latent)
                    latent['y'] = y
                    latent.columns = [str(j) for j in range(len(latent.columns))]
                    latent_list.append(pd.DataFrame(latent))
                
            else:
                
                print('ddpm')
                
                with open(real_data_path + '/meta.json', "r") as json_file:
                    meta = json.load(json_file)
                
                num_cols = [column["name"] for column in meta["columns"] if column["type"] in ["Integer", "Float"]]
                
                ddpm = train_ddpm(
                    parent_dir=dir_,
                    real_data_path=real_data_path,
                    steps=params['steps_syn'],
                    lr=params['lr'],
                    batch_size=params['batch_size'],
                    model_params=params['model_params'],
                    num_timesteps=params['num_timesteps'],
                    gaussian_loss_type=params['gaussian_loss_type'],
                    scheduler=params['scheduler'],
                    T_dict=T_dict,
                    device=device,
                    change_val=False
                )
                
                samples = []
                for i in range(n_samplesets):
                    sample_ddpm(
                        parent_dir=dir_,
                        real_data_path=real_data_path,
                        batch_size=params['batch_size'],
                        num_samples=n_samples,
                        model_type='mlp',
                        model_params=params['model_params'],
                        model_path=str(dir_)+'/model.pt',
                        num_timesteps=params['num_timesteps'],
                        gaussian_loss_type=params['gaussian_loss_type'],
                        scheduler=params['scheduler'],
                        num_numerical_features=len(num_cols),
                        T_dict=T_dict,
                        device=device,
                        seed=i,
                        change_val=False)
                    gen_data = np.load(str(dir_)+'/X_num_train.npy', allow_pickle=True)
                    if 'cardio' in real_data_path or 'mimic_small' in real_data_path:
                        cat = np.load(str(dir_)+'/X_cat_train.npy', allow_pickle=True)
                        gen_data = np.concatenate((gen_data, cat), axis=1)
                    y = np.load(str(dir_) + '/y_train.npy')
                    gen_data = pd.DataFrame(gen_data)
                    gen_data['y'] = y
                    gen_data.columns = [str(j) for j in range(len(gen_data.columns))]
                    samples.append(pd.DataFrame(gen_data))
                    
                return samples
            
        if GenModel == 'ae' or GenModel == 'dpae':
            
            sample = sample_rec_tvae(
                tvae,
                parent_dir=dir_,
                real_data_path=real_data_path,
                num_samples=n_samples,
                train_params=tvae_params,
                change_val=False,
                seed=0,
                device=device,
                latent=latent)
                
            return [sample]
            
        elif GenModel == 'tvae' or GenModel == 'dptvae':
            
            samples = []
            
            for sample_seed in range(n_samplesets):
                    
                if 'tvae' in GenModel:
                    
                    sample = sample_rec_tvae(
                        tvae,
                        parent_dir=dir_,
                        real_data_path=real_data_path,
                        num_samples=n_samples,
                        train_params=tvae_params,
                        change_val=False,
                        seed=sample_seed,
                        device=device,
                        latent=None)
                        
                    samples.append(sample)
                    
            return samples
            
        else:
            
            samples = []
            
            for sample_seed, latent in enumerate(latent_list):
                    
                sample = sample_rec_tvae(
                    tvae,
                    parent_dir=dir_,
                    real_data_path=real_data_path,
                    num_samples=n_samples,
                    train_params=tvae_params,
                    change_val=False,
                    seed=sample_seed,
                    device=device,
                    latent=latent)
                        
                samples.append(sample)
            
            return samples