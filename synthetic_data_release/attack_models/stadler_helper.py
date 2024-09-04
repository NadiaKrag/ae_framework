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

def run_AEframework(data, GenModel, tvae_params, params, n_samples, n_samplesets, device, dname):
    
    print('data shape', data.shape)
    
    num_cols = GenModel.num_cols
    cat_cols = GenModel.cat_cols
    
    print(num_cols)
    print(cat_cols)
    
    with tempfile.TemporaryDirectory() as data_dir_:
        
        info = {
                "name": "Data",
                "id": "data--default",
                "task_type": "binclass",
                "n_num_features": len(num_cols),
                "n_cat_features": len(cat_cols),
                "test_size": len(data),
                "train_size": len(data),
                "val_size": len(data)
            }
    
        with open(data_dir_ + '/info.json', 'w') as json_file:
            json.dump(info, json_file, indent=4)
         
        if 'BayesianNet' in GenModel.__name__ or 'PrivBayes' in GenModel.__name__:    
            data[data.columns[-1]] = data[data.columns[-1]].astype(str)
        else:
            data[data.columns[-1]] = data[data.columns[-1]].astype(float)
        
        np.save(data_dir_ + '/X_num_train.npy', data[num_cols].values)
        np.save(data_dir_ + '/X_num_val.npy', data[num_cols].values)
        np.save(data_dir_ + '/X_num_test.npy', data[num_cols].values)
        if len(cat_cols) != 0:
            np.save(data_dir_ + '/X_cat_train.npy', data[cat_cols].values)
            np.save(data_dir_ + '/X_cat_val.npy', data[cat_cols].values)
            np.save(data_dir_ + '/X_cat_test.npy', data[cat_cols].values)
        np.save(data_dir_ + '/y_train.npy', data[data.columns[-1]].values)
        np.save(data_dir_ + '/y_val.npy', data[data.columns[-1]].values)
        np.save(data_dir_ + '/y_test.npy', data[data.columns[-1]].values)
        
        with tempfile.TemporaryDirectory() as dir_:
        
            with tempfile.TemporaryDirectory() as latent_dir_:
                
                if 'TVAE' in GenModel.__name__:
                    
                    print('TVAE')
                    
                    tvae = train_tvae(
                    parent_dir=dir_,
                    real_data_path=data_dir_,
                    latent_dir=latent_dir_,
                    train_params=tvae_params,
                    change_val=False,
                    device=device,
                    ae=False
                    )
                    
                    latent = pd.DataFrame(np.load(latent_dir_ + '/X_num_train.npy'))
                    latent.columns = [str(x) for x in range(len(latent.columns))]
                    
                elif 'AE' in GenModel.__name__:
                    
                    print('AE')
                    
                    tvae = train_tvae(
                    parent_dir=dir_,
                    real_data_path=data_dir_,
                    latent_dir=latent_dir_,
                    train_params=tvae_params,
                    change_val=False,
                    device=device,
                    ae=True
                    )
                
                    latent = pd.DataFrame(np.load(latent_dir_ + '/X_num_train.npy'))
                    latent.columns = [str(x) for x in range(len(latent.columns))]
                
                if 'PrivBayes' in GenModel.__name__ or 'BayesianNet' in GenModel.__name__:
                    
                    if 'AE' not in GenModel.__name__:
                        
                        print('BayesianNet/PrivBayes')
                        
                        print(data)
                        print(GenModel.metadata)
                        print(data.columns)
                        
                        GenModel.fit(data)
                        
                        samples = []
                        for i in range(n_samplesets):
                            gen_data = GenModel.generate_samples(n_samples)
                            samples.append(gen_data)
                            
                        return samples
                    
                    else:
                    
                        print('TVAE-BayesianNet/TVAE-PrivBayes')
                        
                        info = []
                        for col in latent.columns:
                            curr = {'name': str(col), "type": "Float", "min": float(latent[col].min()), "max": float(latent[col].max())}
                            info.append(curr)
                        meta = {
                            "columns": info
                        }
                        
                        if 'BayesianNet' in GenModel.__name__:
                            baynet = BayesianNet(meta, histogram_bins=params['histogram_bins'], degree=params['degree'])
                        elif 'PrivBayes' in GenModel.__name__:
                            baynet = PrivBayes(meta, epsilon=params['epsilon'], histogram_bins=params['histogram_bins'], degree=params['degree'])
                        
                        baynet.fit(latent)
                        
                        latent_list = []
                        for i in range(n_samplesets):
                            latent = baynet.generate_samples(n_samples)
                            latent_list.append(latent)
                    
                elif 'CTABGAN' in GenModel.__name__:
                    
                    if 'AE' in GenModel.__name__:
                        
                        print('TVAE-CTABGAN/TVAE-DPCTABGAN')
                        
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
                        
                        print('CTABGAN/DPCTABGAN')
                        
                        #with open(real_data_path + '/meta_ctab.json', "r") as json_file:
                            #meta = json.load(json_file)
                        
                        ctabgan = train_ctabgan(
                                parent_dir=dir_,
                                real_data_path=data_dir_,
                                train_params=params,
                                change_val=False,
                                device=device,
                                ds_name=dname,
                                meta_data=[]
                            )
                            
                        samples = []
                        for i in range(n_samplesets):
                            gen_data = sample_ctabgan(
                                ctabgan,
                                parent_dir=dir_,
                                real_data_path=data_dir_,
                                num_samples=n_samples,
                                train_params=params,
                                change_val=False,
                                seed=i,
                                device=device,
                                ds_name=dname
                            )
                            samples.append(gen_data)
                            
                        return samples
                        
                elif 'DDPM' in GenModel.__name__: #in ['DDPM', 'AE-DDPM', 'TVAE-DDPM']:
                    
                    T_dict = {"seed": 0, "normalization": "quantile", "num_nan_policy": None, "cat_nan_policy": None, "cat_min_frequency": None, "cat_encoding": None, "y_policy": "default"}
                    
                    if 'AE' in GenModel.__name__:
                        
                        print('TVAE-DDPM')
                    
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
                            print('num_numerical_features', len(latent.columns))
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
                        
                        print('DDPM')
                        
                        ddpm = train_ddpm(
                            parent_dir=dir_,
                            real_data_path=data_dir_,
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
                                real_data_path=data_dir_,
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
                            try:
                                cat = np.load(str(dir_)+'/X_cat_train.npy', allow_pickle=True)
                                gen_data = np.concatenate((gen_data, cat), axis=1)
                            except:
                                print('NO CATEGORICAL DATA')
                            y = np.load(str(dir_) + '/y_train.npy')
                            gen_data = pd.DataFrame(gen_data)
                            gen_data['y'] = y
                            gen_data.columns = [str(j) for j in range(len(gen_data.columns))]
                            samples.append(pd.DataFrame(gen_data))
                            
                        return samples
                    
                if GenModel.__name__ == 'AE' or GenModel.__name__ == 'PrivAE':
                    
                    sample = sample_rec_tvae(
                        tvae,
                        parent_dir=dir_,
                        real_data_path=data_dir_,
                        num_samples=n_samples,
                        train_params=tvae_params,
                        change_val=False,
                        seed=0,
                        device=device,
                        latent=latent)
                        
                    return [sample]
                    
                elif GenModel.__name__ == 'TVAE' or GenModel.__name__ == 'PrivTVAE':
                    
                    samples = []
                    
                    for sample_seed in range(n_samplesets):
                            
                        sample = sample_rec_tvae(
                            tvae,
                            parent_dir=dir_,
                            real_data_path=data_dir_,
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
                            real_data_path=data_dir_,
                            num_samples=n_samples,
                            train_params=tvae_params,
                            change_val=False,
                            seed=sample_seed,
                            device=device,
                            latent=latent)
                                
                        samples.append(sample)
                    
                    return samples