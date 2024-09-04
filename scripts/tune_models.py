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
from pathlib import Path
from CTGAN.train_sample_tvae import train_tvae, sample_tvae
from scripts.eval_catboost import train_catboost
from scripts.eval_simple import train_simple

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('dim_size', type=int)
parser.add_argument('device', type=str)

args = parser.parse_args()
real_data_path = args.data_path
eval_type = args.eval_type
train_size = args.train_size
dim_size = args.dim_size
device = args.device
assert eval_type in ('merged', 'synthetic')

def objective(trial):
    
    print('Trial number:', trial.number)
    
    start_time = time.time()
        
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)

    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    
    # construct model
    min_n_layers, max_n_layers, d_min, d_max = 1, 3, 3, 7 #3, 7
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

    steps = trial.suggest_categorical('steps', [5000, int(20000/2), int(30000/2)])
    # steps = trial.suggest_categorical('steps', [1000])
    batch_size = trial.suggest_categorical('batch_size', [256, 4096])

    num_samples = train_size #int(train_size * (2 ** trial.suggest_int('frac_samples', -2, 3)))
    embedding_dim = 2 ** trial.suggest_int('embedding_dim', dim_size, dim_size) #3, 7
    loss_factor = trial.suggest_loguniform('loss_factor', 0.001, 10)


    train_params = {
        "lr": lr,
        "epochs": steps,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "loss_factor": loss_factor,
        "compress_dims": d_layers,
        "decompress_dims": d_layers
    }

    trial.set_user_attr("train_params", train_params)
    trial.set_user_attr("num_samples", num_samples)
    
    if trial.number < 0:
        raise optuna.TrialPruned()
    
    try:
        
        results_catboost = []
        results_simple = []
        
        results_catboost_real = []
        results_simple_real = []
        
        stats_folds = {}
        
        score_fold = 0.0
        
        for fold in range(5):
            
            score = 0.0
        
            with tempfile.TemporaryDirectory() as dir_:
                dir_ = Path(dir_)
                ctabgan = train_tvae(
                    parent_dir=dir_,
                    real_data_path=Path(real_data_path) / f"kfolds/{fold}",
                    train_params=train_params,
                    change_val=False,
                    device=device
                )
        
                for sample_seed in range(3):
                    sample_tvae(
                        ctabgan,
                        parent_dir=dir_,
                        real_data_path=Path(real_data_path) / f"kfolds/{fold}",
                        num_samples=num_samples,
                        train_params=train_params,
                        change_val=False,
                        seed=sample_seed,
                        device=device
                    )
        
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
                            real_data_path=real_data_path + f"/kfolds/{fold}", 
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
                                real_data_path=real_data_path + f"/kfolds/{fold}",
                                eval_type=eval_type,
                                T_dict=T_dict,
                                change_val=False,
                                seed=eval_seed,
                                model_name=model_name
                            )
                            
                            results_simple.append(metrics.get_res())
                        
                    score += score_eval / 3
                    
                if trial.number == 0:
                    
                    for eval_seed in range(3):
                    
                        metrics = train_catboost(
                                parent_dir=dir_,
                                real_data_path=real_data_path + f"/kfolds/{fold}", 
                                eval_type='real',
                                T_dict=T_dict,
                                change_val=False,
                                seed = 0
                            )
                            
                        results_catboost_real.append(metrics.get_res())
                            
                        for model_name in ["tree", "rf", "lr", "mlp"]:
                                
                            metrics = train_simple(
                                parent_dir = dir_,
                                real_data_path=real_data_path + f"/kfolds/{fold}",
                                eval_type='real',
                                T_dict=T_dict,
                                change_val=False,
                                seed=0,
                                model_name=model_name
                            )
                            
                            results_simple_real.append(metrics.get_res())
                    
            score_fold += score / 3
            
            stats_folds[f"{fold}_catboost"] = lib.get_results(results_catboost[fold*9:(fold*9)+9])
            stats_folds[f"{fold}_simple"] = lib.get_results(results_simple[fold*9*4:(fold*9*4)+9*4])
            if trial.number == 0:
                stats_folds[f"{fold}_catboost_real"] = lib.get_results(results_catboost_real[fold*3:(fold*3)+3])
                stats_folds[f"{fold}_simple_real"] = lib.get_results(results_simple_real[fold*3*4:(fold*3*4)+3*4])
            
        #save results
        
        stats_catboost = lib.get_results(results_catboost)
        stats_simple = lib.get_results(results_simple)
        if trial.number == 0:
            stats_catboost_real = lib.get_results(results_catboost_real)
            stats_simple_real = lib.get_results(results_simple_real)
        
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        stats_catboost['time'] = elapsed_time_seconds/60
        
        os.makedirs(f"exp/{Path(real_data_path).name}/tvae_{dim_size}/", exist_ok=True)
        result_path = f"exp/{Path(real_data_path).name}/tvae_{dim_size}/tune_results.json"
        if trial.number == 0:
            result_list = [stats_catboost_real, stats_simple_real, stats_catboost, stats_simple, stats_folds, train_params]
            with open(result_path, 'w') as json_file:
                json.dump(result_list, json_file)
        else:
            with open(result_path, 'r') as json_file:
                result_list = json.load(json_file)
            result_list.extend([stats_catboost, stats_simple, stats_folds, train_params])
            with open(result_path, 'w') as json_file:
                json.dump(result_list, json_file)
                
        return score_fold / 5
        
    except Exception as e:
        
        print("An exception occurred:", e)
        print("Trial number is:", trial.number)

        if trial.number == 0:
            sys.exit(1)
        else:
            if len(study.trials) > 0:
                os.makedirs(f"exp/{Path(real_data_path).name}/tvae_{dim_size}/", exist_ok=True)
                config = {
                    "parent_dir": f"exp/{Path(real_data_path).name}/tvae_{dim_size}/",
                    "real_data_path": real_data_path,
                    "seed": 0,
                    "device": args.device,
                    "train_params": study.best_trial.user_attrs["train_params"],
                    "sample": {"seed": 0, "num_samples": study.best_trial.user_attrs["num_samples"]},
                    "eval": {
                        "type": {"eval_model": "catboost", "eval_type": eval_type},
                        "T": {
                            "seed": 0,
                            "normalization": None,
                            "num_nan_policy": None,
                            "cat_nan_policy": None,
                            "cat_min_frequency": None,
                            "cat_encoding": None,
                            "y_policy": "default"
                        },
                    }
                }
                
                os.makedirs(f"exp/{Path(real_data_path).name}/tvae_{dim_size}/synthetic_{trial.number}", exist_ok=True)
                
                ctabgan = train_tvae(
                    parent_dir=f"exp/{Path(real_data_path).name}/tvae_{dim_size}/synthetic_{trial.number}/",
                    real_data_path=real_data_path,
                    train_params=study.best_trial.user_attrs["train_params"],
                    change_val=False,
                    device=device
                )
                
                sample_tvae(
                        ctabgan,
                        parent_dir= f"exp/{Path(real_data_path).name}/tvae_{dim_size}/synthetic_{trial.number}/",
                        real_data_path=real_data_path,
                        num_samples=(train_size+int(train_size*0.2)),
                        train_params=study.best_trial.user_attrs["train_params"],
                        change_val=False,
                        device=device
                    )
                
                lib.dump_config(config, config["parent_dir"]+"config.toml")
                
                return 0.0
            
            else:
                return 0.0

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=10, show_progress_bar=True)

os.makedirs(f"exp/{Path(real_data_path).name}/tvae_{dim_size}/", exist_ok=True)
config = {
    "parent_dir": f"exp/{Path(real_data_path).name}/tvae_{dim_size}/",
    "real_data_path": real_data_path,
    "seed": 0,
    "device": args.device,
    "train_params": study.best_trial.user_attrs["train_params"],
    "sample": {"seed": 0, "num_samples": study.best_trial.user_attrs["num_samples"]},
    "eval": {
        "type": {"eval_model": "catboost", "eval_type": eval_type},
        "T": {
            "seed": 0,
            "normalization": None,
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default"
        },
    }
}

os.makedirs(f"exp/{Path(real_data_path).name}/tvae_{dim_size}/synthetic", exist_ok=True)

ctabgan = train_tvae(
    parent_dir=f"exp/{Path(real_data_path).name}/tvae_{dim_size}/synthetic/",
    real_data_path=real_data_path,
    train_params=study.best_trial.user_attrs["train_params"],
    change_val=False,
    device=device
)

sample_tvae(
        ctabgan,
        parent_dir= f"exp/{Path(real_data_path).name}/tvae_{dim_size}/synthetic/",
        real_data_path=real_data_path,
        num_samples=(train_size+int(train_size*0.2)),
        train_params=study.best_trial.user_attrs["train_params"],
        change_val=False,
        device=device
    )

lib.dump_config(config, config["parent_dir"]+"config.toml")

#subprocess.run(['python3.9', "scripts/eval_seeds.py", '--config', f'{config["parent_dir"]+"config.toml"}',
                #'10', "tvae", eval_type, "catboost", "5"], check=True)