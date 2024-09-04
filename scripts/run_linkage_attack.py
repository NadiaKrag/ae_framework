import subprocess
import json

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('model', type=str)
parser.add_argument('tid', type=str)
parser.add_argument('device', type=str)
parser.add_argument('dp_type', type=str)

args = parser.parse_args()
ds_name = args.ds_name
model = args.model
tid = args.tid
dp_type = args.dp_type
device = args.device

tvae_params = {}

if ds_name == 'cardio':
    if model == 'BayesianNet':
        print('BayesianNet')
        params = {'histogram_bins': 21, 'degree': 1}
    elif model == 'PrivBayes':
        print('PrivBayes')
        params = {'histogram_bins': 21, 'degree': 1, 'epsilon': 0.1}
    elif model == 'CTABGAN':
        print('CTABGAN')
        params = {'lr': 0.0019117884903855674, 'epochs': 1000, 'class_dim': [256, 128], 'batch_size_ctab': 512, 'random_dim': 128, 'num_channels': 16, 'private': False, 'sigma': None}
    elif model == 'DPCTABGAN':
        print('DPCTABGAN')
        params = {}
    elif model == 'DDPM':
        print('DDPM')
        params = {'steps_syn': 5000, 'lr': 0.00022433973285237988, 'batch_size': 256, 'model_params': {'num_classes': 2, 'is_y_cond': 1, 'rtdl_params': {'d_layers': [512, 512, 512, 512, 512, 64], 'dropout': 0.0, 'd_in': 128, 'd_out': 19}, 'd_in': 19}, 'num_timesteps': 100, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'}
    elif 'AE' in model:
        print('AE')
        tvae_params = {}
        params = {}
        if 'Priv' in model:
            private = True
            epsilon = 0.1
        else:
            private = False
            epsilon = None
        if '-' in model:
            if model.split('-')[1] == 'BayesianNet':
                print('AE-BayesianNet')
                if 'TVAE' in model:
                    print('HEYOU')
                    #tvae_params = {'lr': 0.0009467725267737506, 'epochs': 5000, 'embedding_dim': 16, 'batch_size': 4096, 'loss_factor': 8.311201677559318, 'compress_dims': [128, 512, 512, 16], 'decompress_dims': [128, 512, 512, 16], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    #params = {'histogram_bins': 13, 'degree': 1}
                    #tvae_params = {'lr': 0.0010852752924672504, 'epochs': 10000, 'embedding_dim': 16, 'batch_size': 4096, 'loss_factor': 2.978558018856716, 'compress_dims': [128, 128, 128, 16], 'decompress_dims': [128, 128, 128, 16], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    #params = {'histogram_bins': 11, 'degree': 1}
                    #tvae_params = {'lr': 0.0009467725267737506, 'epochs': 5000, 'embedding_dim': 16, 'batch_size': 4096, 'loss_factor': 8.311201677559318, 'compress_dims': [128, 512, 512, 16], 'decompress_dims': [128, 512, 512, 16], 'private': False, 'epsilon': None}
                    #params = {'histogram_bins': 13, 'degree': 1}
                    tvae_params = {'lr': 0.0019625245852390133, 'epochs': 15000, 'embedding_dim': 8, 'batch_size': 256, 'loss_factor': 1.3246974647468452, 'compress_dims': [64, 8], 'decompress_dims': [64, 8], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'histogram_bins': 8, 'degree': 1}
                else:
                    tvae_params = {'lr': 0.002476375879755182, 'epochs': 15000, 'embedding_dim': 16, 'batch_size': 256, 'loss_factor': 0.5693097755771417, 'compress_dims': [128, 256, 256, 16], 'decompress_dims': [128, 256, 256, 16], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'histogram_bins': 11, 'degree': 1}
            elif model.split('-')[1] == 'PrivBayes':
                print('AE-PrivBayes')
                tvae_params = {'lr': 0.002476375879755182, 'epochs': 15000, 'embedding_dim': 16, 'batch_size': 256, 'loss_factor': 0.5693097755771417, 'compress_dims': [128, 256, 256, 16], 'decompress_dims': [128, 256, 256, 16], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                params = {'histogram_bins': 11, 'degree': 1, 'epsilon': 0.1}
            elif model.split('-')[1] == 'CTABGAN':
                print('AE-CTABGAN')
                if 'TVAE' in model:
                    tvae_params = {'lr': 0.0003550981089558474, 'epochs': 5000, 'embedding_dim': 16, 'batch_size': 4096, 'loss_factor': 2.6659523689948066, 'compress_dims': [512, 16], 'decompress_dims': [512, 16], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'lr': 0.0003550981089558474, 'epochs': 1000, 'class_dim': [512, 16], 'batch_size_ctab': 2048, 'random_dim': 16, 'num_channels': 32, 'private': False, 'sigma': None}
                else:
                    tvae_params = {'lr': 0.0016903806867294324, 'epochs': 5000, 'embedding_dim': 16, 'batch_size': 4096, 'loss_factor': 0.4645279386915541, 'compress_dims': [256, 16], 'decompress_dims': [256, 16], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'lr': 0.0016903806867294324, 'epochs': 1000, 'class_dim': [256, 16], 'batch_size_ctab': 512, 'random_dim': 16, 'num_channels': 32, 'private': False, 'sigma': None}
            elif model.split('-')[1] == 'DPCTABGAN':
                print('AE-CTABGAN')
                if 'TVAE' in model:
                    model_params = {}
                    params = {}
            elif model.split('-')[1] == 'DDPM':
                print('AE-DDPM')
                if 'TVAE' in model:
                    tvae_params = {'lr': 0.0002844170439065166, 'epochs': 5000, 'embedding_dim': 128, 'batch_size': 256, 'loss_factor': 2.831604372728374, 'compress_dims': [256, 128], 'decompress_dims': [256, 128], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'steps_syn': 1500, 'lr': 0.0002844170439065166, 'batch_size': 256, 'model_params': {'num_classes': 0, 'is_y_cond': 1, 'rtdl_params': {'d_layers': [256, 512], 'dropout': 0.0, 'd_in': 128, 'd_out': 128}, 'd_in': 128}, 'num_timesteps': 500, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'}

                else:
                    #tvae_params = {'lr': 0.001800501157703768, 'epochs': 5000, 'embedding_dim': 128, 'batch_size': 4096, 'loss_factor': 1.0083759203824427, 'compress_dims': [64, 128, 128, 128, 128, 128], 'decompress_dims': [64, 128, 128, 128, 128, 128], 'private': False, 'epsilon': None}
                    #params = {'steps_syn': 1500, 'lr': 0.001800501157703768, 'batch_size': 4096, 'model_params': {'num_classes': 0, 'is_y_cond': 1, 'rtdl_params': {'d_layers': [64, 128, 128, 128, 128, 128], 'dropout': 0.0, 'd_in': 128, 'd_out': 128}, 'd_in': 128}, 'num_timesteps': 100, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'}
                    tvae_params = {'lr': 0.002141060640841936, 'epochs': 15000, 'embedding_dim': 16, 'batch_size': 256, 'loss_factor': 0.6296876056157568, 'compress_dims': [256, 512, 512, 16], 'decompress_dims': [256, 512, 512, 16], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'steps_syn': 5000, 'lr': 0.002141060640841936, 'batch_size': 256, 'model_params': {'num_classes': 0, 'is_y_cond': 1, 'rtdl_params': {'d_layers': [256, 512, 512, 512], 'dropout': 0.0, 'd_in': 128, 'd_out': 16}, 'd_in': 16}, 'num_timesteps': 500, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'}


elif ds_name == 'mimic_large':
    if model == 'BayesianNet':
        print('BayesianNet')
        params = {}
    elif model == 'PrivBayes':
        print('PrivBayes')
        params = {}
    elif model == 'CTABGAN':
        print('CTABGAN')
        params = {}
    elif model == 'DPCTABGAN':
        print('DPCTABGAN')
        params = {}
    elif model == 'DDPM':
        print('DDPM')
        params = {'steps_syn': 5000, 'lr': 1.0880558989703993e-05, 'batch_size': 256, 'model_params': {'num_classes': 2, 'is_y_cond': 1, 'rtdl_params': {'d_layers': [256, 512], 'dropout': 0.0, 'd_in': 128, 'd_out': 100}, 'd_in': 100}, 'num_timesteps': 500, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'}
    elif 'AE' in model:
        print('AE')
        tvae_params = {}
        params = {}
        if 'Priv' in model:
            private = True
            epsilon = 0.1
        else:
            private = False
            epsilon = None
        if '-' in model:
            if model.split('-')[1] == 'BayesianNet':
                print('AE-BayesianNet')
                tvae_params = {}
                params = {}
            elif model.split('-')[1] == 'PrivBayes':
                print('AE-PrivBayes')
                tvae_params = {}
                params = {}
            elif model.split('-')[1] == 'CTABGAN':
                print('AE-CTABGAN')
                tvae_params = {}
                params = {}
            elif model.split('-')[1] == 'DPCTABGAN':
                print('AE-CTABGAN')
                tvae_params = {}
                params = {}
            elif model.split('-')[1] == 'DDPM':
                print('AE-DDPM')
                if 'TVAE' in model:
                    tvae_params = {}
                    params = {}
                else:
                    tvae_params = {'lr': 0.00047756638508070437, 'epochs': 15000, 'embedding_dim': 128, 'batch_size': 256, 'loss_factor': 0.06144711744841965, 'compress_dims': [256, 128], 'decompress_dims': [256, 128], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'steps_syn': 1500, 'lr': 0.00047756638508070437, 'batch_size': 256, 'model_params': {'num_classes': 0, 'is_y_cond': 1, 'rtdl_params': {'d_layers': [256, 512], 'dropout': 0.0, 'd_in': 128, 'd_out': 128}, 'd_in': 128}, 'num_timesteps': 100, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'}

elif ds_name == 'mimic_small':
    if model == 'BayesianNet':
        print('BayesianNet')
        params = {'histogram_bins': 15, 'degree': 1}
    elif model == 'PrivBayes':
        print('PrivBayes')
        params = {'histogram_bins': 15, 'degree': 1, 'epsilon': 0.1}
    elif model == 'CTABGAN':
        print('CTABGAN')
        params = {'lr': 3.2602984141273626e-05, 'epochs': 1000, 'class_dim': [128, 64, 64, 64, 64, 128], 'batch_size_ctab': 512, 'random_dim': 128, 'num_channels': 64, 'private': False, 'sigma': None}
    elif model == 'DPCTABGAN':
        print('DPCTABGAN')
        params = {}
    elif model == 'DDPM':
        print('DDPM')
        params = {'steps_syn': 5000, 'lr': 0.0011903951277436099, 'batch_size': 4096, 'model_params': {'num_classes': 2, 'is_y_cond': 1, 'rtdl_params': {'d_layers': [512, 128], 'dropout': 0.0, 'd_in': 128, 'd_out': 71}, 'd_in': 71}, 'num_timesteps': 500, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'}
    elif 'AE' in model:
        print('AE')
        tvae_params = {}
        params = {}
        if 'Priv' in model:
            private = True
            epsilon = 0.1
        else:
            private = False
            epsilon = None
        if '-' in model:
            if model.split('-')[1] == 'BayesianNet':
                print('AE-BayesianNet')
                if 'TVAE' in model:
                    tvae_params = {'lr': 6.17685272185741e-05, 'epochs': 10000, 'embedding_dim': 8, 'batch_size': 4096, 'loss_factor': 9.434868832844716, 'compress_dims': [128, 512, 512, 8], 'decompress_dims': [128, 512, 512, 8], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'histogram_bins': 6, 'degree': 1}
                else:
                    print('HEO2')
                    #tvae_params = {'lr': 0.0025083463846739835, 'epochs': 15000, 'embedding_dim': 8, 'batch_size': 4096, 'loss_factor': 8.439808575526627, 'compress_dims': [128, 8], 'decompress_dims': [128, 8], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    #params = {'histogram_bins': 12, 'degree': 1}
                    tvae_params = {'lr': 0.002686304157938728, 'epochs': 15000, 'embedding_dim': 8, 'batch_size': 4096, 'loss_factor': 7.7126532854259935, 'compress_dims': [64, 512, 512, 8], 'decompress_dims': [64, 512, 512, 8], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'histogram_bins': 13, 'degree': 1}
                    #tvae_params = {'lr': 0.00017794588399446697, 'epochs': 15000, 'embedding_dim': 16, 'batch_size': 4096, 'loss_factor': 8.69121900570005, 'compress_dims': [128, 64, 64, 16], 'decompress_dims': [128, 64, 64, 16], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    #params = {'histogram_bins': 16, 'degree': 1}
            elif model.split('-')[1] == 'PrivBayes':
                print('AE-PrivBayes')
                tvae_params = {'lr': 0.002686304157938728, 'epochs': 15000, 'embedding_dim': 8, 'batch_size': 4096, 'loss_factor': 7.7126532854259935, 'compress_dims': [64, 512, 512, 8], 'decompress_dims': [64, 512, 512, 8], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                params = {'histogram_bins': 13, 'degree': 1, 'epsilon': 0.1}
            elif model.split('-')[1] == 'CTABGAN':
                print('AE-CTABGAN')
                if 'TVAE' in model:
                    tvae_params = {'lr': 2.985964086885175e-05, 'epochs': 15000, 'embedding_dim': 16, 'batch_size': 256, 'loss_factor': 1.7601629345058718, 'compress_dims': [64, 512, 512, 512, 512, 16], 'decompress_dims': [64, 512, 512, 512, 512, 16], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'lr': 2.985964086885175e-05, 'epochs': 1000, 'class_dim': [64, 512, 512, 512, 512, 16], 'batch_size_ctab': 512, 'random_dim': 128, 'num_channels': 16, 'private': False, 'sigma': None}
                else:
                    print('HAI')
                    #tvae_params = {'lr': 7.847786840606716e-05, 'epochs': 15000, 'embedding_dim': 8, 'batch_size': 4096, 'loss_factor': 6.661199682776916, 'compress_dims': [64, 8], 'decompress_dims': [64, 8], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    #params = {'lr': 7.847786840606716e-05, 'epochs': 2500, 'class_dim': [64, 8], 'batch_size_ctab': 1024, 'random_dim': 32, 'num_channels': 16, 'private': False, 'sigma': None}
                    #tvae_params = {'lr': 0.000120140731128545, 'epochs': 15000, 'embedding_dim': 8, 'batch_size': 4096, 'loss_factor': 0.2898933527712397, 'compress_dims': [512, 256, 256, 256, 256, 8], 'decompress_dims': [512, 256, 256, 256, 256, 8], 'private': False, 'epsilon': None}
                    #params = {'lr': 0.000120140731128545, 'epochs': 2500, 'class_dim': [512, 256, 256, 256, 256, 8], 'batch_size_ctab': 1024, 'random_dim': 32, 'num_channels': 16, 'private': False, 'sigma': None}
                    tvae_params = {'lr': 0.0002288113668475521, 'epochs': 15000, 'embedding_dim': 16, 'batch_size': 256, 'loss_factor': 0.13049073550362397, 'compress_dims': [256, 256, 256, 256, 256, 16], 'decompress_dims': [256, 256, 256, 256, 256, 16], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'lr': 0.0002288113668475521, 'epochs': 2500, 'class_dim': [256, 256, 256, 256, 256, 16], 'batch_size_ctab': 512, 'random_dim': 16, 'num_channels': 64, 'private': False, 'sigma': None}
            elif model.split('-')[1] == 'DPCTABGAN':
                print('AE-CTABGAN')
                tvae_params = {}
                params = {}
            elif model.split('-')[1] == 'DDPM':
                print('AE-DDPM')
                if 'TVAE' in model:
                    #tvae_params = {'lr': 7.076361265115194e-05, 'epochs': 15000, 'embedding_dim': 8, 'batch_size': 4096, 'loss_factor': 0.9234766895471748, 'compress_dims': [256, 128, 128, 8], 'decompress_dims': [256, 128, 128, 8], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    #params = {'steps_syn': 2500, 'lr': 7.076361265115194e-05, 'batch_size': 4096, 'model_params': {'num_classes': 0, 'is_y_cond': 1, 'rtdl_params': {'d_layers': [256, 128, 128, 512], 'dropout': 0.0, 'd_in': 128, 'd_out': 8}, 'd_in': 8}, 'num_timesteps': 100, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'}
                    tvae_params = {'lr': 0.00023985665329096526, 'epochs': 10000, 'embedding_dim': 128, 'batch_size': 256, 'loss_factor': 2.170878013147847, 'compress_dims': [256, 128], 'decompress_dims': [256, 128], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'steps_syn': 1500, 'lr': 0.00023985665329096526, 'batch_size': 256, 'model_params': {'num_classes': 0, 'is_y_cond': 1, 'rtdl_params': {'d_layers': [256, 512], 'dropout': 0.0, 'd_in': 128, 'd_out': 128}, 'd_in': 128}, 'num_timesteps': 500, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'}
                else:
                    tvae_params = {'lr': 0.00012259668395421265, 'epochs': 15000, 'embedding_dim': 128, 'batch_size': 256, 'loss_factor': 0.03220480855516045, 'compress_dims': [64, 256, 256, 128], 'decompress_dims': [64, 256, 256, 128], 'private': private, 'epsilon': epsilon, 'dp_type': dp_type}
                    params = {'steps_syn': 2500, 'lr': 0.00012259668395421265, 'batch_size': 256, 'model_params': {'num_classes': 0, 'is_y_cond': 1, 'rtdl_params': {'d_layers': [64, 256, 256, 256], 'dropout': 0.0, 'd_in': 128, 'd_out': 128}, 'd_in': 128}, 'num_timesteps': 500, 'gaussian_loss_type': 'mse', 'scheduler': 'cosine'}

runconfig = {
  "nIter": 15,
  "sizeRawA": 10000,
  "nSynA": 10,
  "nShadows": 10,
  "sizeRawT": 1000,
  "sizeSynT": 1000,
  "nSynT": 5,
  "nTargets": 0,
  "Targets": [tid],
  "generativeModels": {
    model: [[tvae_params, params]]
  },
  "device": device
}

'''
runconfig = {
  "nIter": 15,
  "sizeRawA": 10000,
  "nSynA": 10,
  "nShadows": 10,
  "sizeRawT": 1000,
  "sizeSynT": 1000,
  "nSynT": 5,
  "nTargets": 0,
  "Targets": [tid],
  "generativeModels": {
    model: [params]
  }
}
'''

with open(f'/content/gdrive/MyDrive/thesis_02/tab-ddpm/exp/{ds_name}/stadler/runconfig_{model}_{tid}.json', 'w') as json_file:
    json.dump(runconfig, json_file, indent=4)

subprocess.run(['python', '/content/gdrive/MyDrive/thesis_02/tab-ddpm/synthetic_data_release/linkage_cli.py', '-D', f'/content/gdrive/MyDrive/thesis_02/tab-ddpm/data/{ds_name}/{ds_name}', '-RC', f'/content/gdrive/MyDrive/thesis_02/tab-ddpm/exp/{ds_name}/stadler/runconfig_{model}_{tid}.json', '-O', f'/content/gdrive/MyDrive/thesis_02/tab-ddpm/exp/{ds_name}/stadler/'], check=True)
