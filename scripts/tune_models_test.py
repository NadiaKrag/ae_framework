import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of the script (your project directory) to the Python path
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from multiprocessing.sharedctypes import RawValue
import tempfile
import subprocess
import lib
import os
import optuna
import argparse
import time
import json
import sys
import signal
import numpy as np
from pathlib import Path
from CTGAN.train_sample_tvae import train_tvae, sample_rec_tvae
from scripts.eval_catboost import train_catboost
from scripts.eval_simple import train_simple
from scripts.AEframework_helper import run_AEframework

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('model_type', type=str)
parser.add_argument('eval_type', type=str)
parser.add_argument('dim_size', type=int)
parser.add_argument('device', type=str)
parser.add_argument('n_trials', type=int)
parser.add_argument('trial_start', type=int)
parser.add_argument('dp_type', type=str)

args = parser.parse_args()
real_data_path = args.data_path
eval_type = args.eval_type
train_size = args.train_size
model_type = args.model_type
dim_size = args.dim_size
device = args.device
n_trials = args.n_trials
trial_start = args.trial_start
dp_type = args.dp_type
print('DP TYPE', dp_type)
assert eval_type in ('merged', 'synthetic')

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution time exceeded.")

def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def objective(trial):
    
    print('Trial number is:', trial.number)
    
    start_time = time.time()
    
    ##########
    
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)

    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    
    # construct model
    min_n_layers, max_n_layers, d_min, d_max = 1, 3, 6, 9
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    ####

    steps = trial.suggest_categorical('steps', [5000, int(20000/2), int(30000/2)]) #500, 2000, 3000, 5000
    batch_size = trial.suggest_categorical('batch_size', [256, 4096])

    #num_samples = train_size #int(train_size * (2 ** trial.suggest_int('frac_samples', -2, 3)))
    if dim_size == 0:
        if 'baynet' in model_type or 'privbay' in model_type:
            embedding_dim = 2 ** trial.suggest_int('embedding_dim', 3, 4)
        elif 'ctabgan' in model_type:
            embedding_dim = 2 ** trial.suggest_int('embedding_dim', 3, 4)
        else:
            embedding_dim = 2 ** trial.suggest_int('embedding_dim', 3, 7) #3, 7 dim_size
    else:
        embedding_dim = 2 ** trial.suggest_int('embedding_dim', dim_size, dim_size)
        
    loss_factor = trial.suggest_loguniform('loss_factor', 0.001, 10)

    if "privae" in model_type or "privtvae" in model_type:
        private = True
        epsilon = 0.1
    else:
        private = False
        epsilon = None

    train_params = {
        "lr": 7.847786840606716e-05,
        "epochs": 15000,
        "embedding_dim": 8,
        "batch_size": 4096,
        "loss_factor": 6.661199682776916,
        "compress_dims": [64, 8],
        "decompress_dims": [64, 8],
        "private": private,
        "epsilon": epsilon,
        "dp_type": dp_type
    }

    if 'ae' in model_type: #OR REC WON'T WORK
        train_params['compress_dims'][-1] = train_params['embedding_dim']
        train_params['decompress_dims'][-1] = train_params['embedding_dim']

    trial.set_user_attr("train_params", train_params)
    #trial.set_user_attr("num_samples", num_samples)
    
    params = {}
    
    if 'baynet' in model_type:
        
        histogram_bins = trial.suggest_int('histogram_bins', 5, 30)
        degree = trial.suggest_int('degree', 1, 1)
    
        params = {
            'histogram_bins': 12,
            'degree': degree
        }
        
    elif 'privbay' in model_type:
        
        histogram_bins = trial.suggest_int('histogram_bins', 5, 30)
        degree = trial.suggest_int('degree', 1, 1)
    
        params = {
            'histogram_bins': histogram_bins,
            'degree': degree,
            'epsilon': 0.1
        }
        
    elif 'ctabgan' in model_type:
        
        lr_ctab = trial.suggest_loguniform('lr', 0.00001, 0.003)
        steps_ctab = trial.suggest_categorical('steps_ctab', [1000, int(5000/2), int(10000/2)]) #100, 500, 1000
        batch_size_ctab = 2 ** trial.suggest_int('batch_size_ctab', 9, 11)
        random_dim = 2 ** trial.suggest_int('random_dim', 4, 7)
        num_channels = 2 ** trial.suggest_int('num_channels', 4, 6)
        
        if 'dpctabgan' in model_type:
            private = True
            sigma = 3.0
            steps_ctab = 10
        else:
            private = False
            sigma = None
        
        params = {
            "lr": 7.847786840606716e-05,
            "epochs": 2500,
            "class_dim": [64, 8],
            "batch_size_ctab": 1024,
            "random_dim": 32,
            "num_channels": 16,
            "private": private,
            "sigma": sigma,
        }

        
    elif 'ddpm' in model_type:
        
        d_layers = _suggest_mlp_layers(trial)
        weight_decay = 0.0  
        batch_size = trial.suggest_categorical('batch_size', [256, 4096])
        steps_ddpm = trial.suggest_categorical('steps_ddpm', [int(3000/2), int(5000/2), int(10000/2)]) #300, 500, 1000, 3000
        gaussian_loss_type = 'mse'
        num_timesteps = trial.suggest_categorical('num_timesteps', [100, int(1000/2)])
        scheduler = "cosine"
        
        if 'ae' in model_type:
            num_classes = 0
            is_y_cond = 1
        else:
            num_classes = 2
            is_y_cond = 1
        
        params = {
            'steps_syn': 2500,
            'lr': 0.00012259668395421265,
            'batch_size': 256,
            'model_params': {
                'num_classes': 0,
                'is_y_cond': 1,
                'rtdl_params': {
                    'd_layers': [64, 256, 256, 256],
                    'dropout': 0.0
                }
            },
            'num_timesteps': 500,
            'gaussian_loss_type': 'mse',
            'scheduler': 'cosine'
        }

    
    trial.set_user_attr("params", params)
    
    ##########
    
    if trial.number < trial_start:
        raise optuna.TrialPruned()
    
    #try:
        
    results_catboost = []
    results_simple = []
    
    results_catboost_real = []
    results_simple_real = []
        
    score = 0.0
    
    if model_type == 'ae':
        n_samplesets = 1
    else:
        n_samplesets = 3

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1200)

    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
            
        #output: tvae, latent_space (real or synthetic)
        #samples = run_AEframework(dir_, real_data_path, train_params, model_type, params=params, n_samples=train_size, n_samplesets=n_samplesets, device=device)
        
        try:
            #result = func()  # Execute the function
            samples = run_AEframework(dir_, real_data_path, train_params, model_type, params=params, n_samples=train_size, n_samplesets=n_samplesets, device=device)
        except TimeoutError:
            print("Function execution timed out.")
            # Handle timeout as needed
            raise optuna.TrialPruned()
        else:
            signal.alarm(0)  # Canel the alarm if function completes before timeout
            #return result
        
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        
        with open(real_data_path + '/meta.json', "r") as json_file:
            meta = json.load(json_file)
            
        cat_cols = [column["name"] for column in meta["columns"] if column["type"] in ["Ordinal", "Categorical"]]
        num_cols = [column["name"] for column in meta["columns"] if column["type"] in ["Integer", "Float"]]
        
        for sample in samples:
        
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
            np.save(dir_ / 'y_train', y)
            
            print('SAMPLE MEAN', np.mean(X_num))
    
            score_eval = 0.0
    
            for eval_seed in range(3):
    
                T_dict = {
                    "seed": 0,
                    "normalization": None,
                    "num_nan_policy": None,
                    "cat_nan_policy": None,
                    "cat_min_frequency": None,
                    "cat_encoding": None,
                    "y_policy": "default"
                }
                metrics = train_catboost(
                    parent_dir=dir_,
                    real_data_path=real_data_path, 
                    eval_type=eval_type,
                    T_dict=T_dict,
                    change_val=False,
                    seed = eval_seed
                )
    
                score_eval += metrics.get_val_score()
                results_catboost.append(metrics.get_res())
                
                for model_name in ["tree", "rf", "lr", "mlp"]:
                    
                    metrics = train_simple(
                        parent_dir = dir_,
                        real_data_path=real_data_path,
                        eval_type=eval_type,
                        T_dict=T_dict,
                        change_val=False,
                        seed=eval_seed,
                        model_name=model_name
                    )
                    
                    results_simple.append(metrics.get_res())
                
            score += score_eval / 3
            
        '''if trial.number == 0:
            
            for eval_seed in range(3):
            
                metrics = train_catboost(
                        parent_dir=dir_,
                        real_data_path=real_data_path, 
                        eval_type='real',
                        T_dict=T_dict,
                        change_val=False,
                        seed = eval_seed
                    )
                    
                results_catboost_real.append(metrics.get_res())
                    
                for model_name in ["tree", "rf", "lr", "mlp"]:
                        
                    metrics = train_simple(
                        parent_dir = dir_,
                        real_data_path=real_data_path,
                        eval_type='real',
                        T_dict=T_dict,
                        change_val=False,
                        seed=eval_seed,
                        model_name=model_name
                    )
                    
                    results_simple_real.append(metrics.get_res())'''
        
    #save results
    
    stats_catboost = lib.get_results(results_catboost)
    stats_simple = lib.get_results(results_simple)
    if trial.number == 0:
        stats_catboost_real = lib.get_results(results_catboost_real)
        stats_simple_real = lib.get_results(results_simple_real)
    
    #end_time = time.time()
    #elapsed_time_seconds = end_time - start_time
    stats_catboost['time'] = elapsed_time_seconds/60
    
    os.makedirs(f"exp/{Path(real_data_path).name}/{model_type}_{dim_size}/", exist_ok=True)
    result_path = os.path.join(f"exp/{Path(real_data_path).name}/{model_type}_{dim_size}", "tune_results.json")
    if os.path.exists(result_path):
        with open(result_path, 'r') as json_file:
            result_list = json.load(json_file)
        result_list.extend([stats_catboost, stats_simple, train_params, params])
        with open(result_path, 'w') as json_file:
            json.dump(result_list, json_file)
    else:
        if trial.number == 0:
            result_list = [stats_catboost_real, stats_simple_real, stats_catboost, stats_simple, train_params, params]
            with open(result_path, 'w') as json_file:
                json.dump(result_list, json_file)
        else:
            result_list = [stats_catboost, stats_simple, train_params, params]
            with open(result_path, 'w') as json_file:
                json.dump(result_list, json_file)
                
    print(train_params)
    print(params)
            
    return score / n_samplesets
        
    '''except Exception as e:
        
        print("An exception occurred:", e)
        print("Trial number is:", trial.number)

        try:
            print('SAVING BEST SYNTHETIC DATASET')
            os.makedirs(f"exp/{Path(real_data_path).name}/{model_type}_{dim_size}/synthetic_{trial.number}", exist_ok=True)
            run_AEframework(f"exp/{Path(real_data_path).name}/{model_type}_{dim_size}/synthetic_{trial.number}/", real_data_path, study.best_trial.user_attrs["train_params"], model_type, study.best_trial.user_attrs["params"], train_size, n_samplesets=1, device=device)
            return 0.0
        except:
            return 0.0
            
        return 0.0'''

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

os.makedirs(f"exp/{Path(real_data_path).name}/{model_type}_{dim_size}/synthetic", exist_ok=True)

run_AEframework(f"exp/{Path(real_data_path).name}/{model_type}_{dim_size}/synthetic/", real_data_path, study.best_trial.user_attrs["train_params"], model_type, study.best_trial.user_attrs["params"], train_size, n_samplesets=1, device=device)