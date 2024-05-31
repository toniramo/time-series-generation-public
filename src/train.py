import itertools
import os
from os import path as pt

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd


from hyperparameters import SIGCWGAN_CONFIGS
from lib import ALGOS
from lib.algos.base import BaseConfig
from lib.data import download_man_ahl_dataset, download_mit_ecg_dataset
from lib.data import get_data
from lib.plot import savefig, create_summary, create_subplots, plot_all
from lib.utils import pickle_it, get_range
from evaluate import evaluate_generator


def get_algo_config(dataset, data_params):
    """ Get the algorithms parameters. """
    key = dataset
    if dataset == 'VAR':
        key += str(data_params['dim'])
    elif dataset == 'STOCKS':
        key += '_' + '_'.join(data_params['assets'])
    return SIGCWGAN_CONFIGS[key]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_algo(algo_id, base_config, dataset, data_params, x_real):
    if algo_id == 'SigCWGAN':
        algo_config = get_algo_config(dataset, data_params)
        algo = ALGOS[algo_id](x_real=x_real, config=algo_config, base_config=base_config)
    else:
        algo = ALGOS[algo_id](x_real=x_real, base_config=base_config)
    return algo
    

def run(algo_id, base_config, base_dir, dataset, spec, data_params={}, args=None):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s, %s' % (algo_id, dataset, spec))
    experiment_directory = pt.join(base_dir, dataset, spec, 'seed={}'.format(base_config.seed), algo_id)
    if not pt.exists(experiment_directory):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_directory)
    # Set seed for exact reproducibility of the experiments
    with open(os.path.join(experiment_directory,'conf.txt'), 'w') as f:
        f.write(f'{base_config = }\n{args = }')
    set_seed(base_config.seed)
    # initialise dataset and algo
    pipeline, x_real_raw, x_real, labels = get_data(dataset, base_config.p, base_config.q, **data_params)
    x_real = x_real.to(base_config.device)
    ind_train = int(x_real.shape[0] * 0.8)
    x_real_train, x_real_test = x_real[:ind_train], x_real[ind_train:] #train_test_split(x_real, train_size = 0.8)

    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_train)
    # Train the algorithm
    algo.fit()
    if args.skip_plots:
        return experiment_directory
    # create summary
    #create_summary(dataset, base_config.device, algo.G, base_config.p, base_config.q, x_real_test)
    #savefig('summary.png', experiment_directory)
    
    x_real_full = pipeline.transform(x_real_raw)
    count = len(x_real_full[0])
    x_fake_full = create_summary(dataset, base_config.device, algo.G, base_config.p, count, x_real[0], one=True)

    x_all = np.concatenate((x_fake_full.cpu().numpy()[0],
                            x_real_full.cpu().numpy()[0]),
                           axis=0)

    ymins, ymaxs = get_range(x_all)
    
    #ymins_fake, ymaxs_fake = get_range(x_fake_full)
    ymins_real, ymaxs_real = get_range(x_real_full)
    print(ymins, ymaxs)

    value_range = (ymaxs.max()-ymins.min()) / (ymaxs_real.max()-ymins_real.min())
                
    plot_all(x_fake_full, x_real_full, [50,200,2000,count], labels, experiment_directory)
    
    # Pickle generator weights, real path and hyperparameters.
    pickle_it(x_real, pt.join(pt.dirname(experiment_directory), 'x_real.torch'))
    pickle_it(x_real_test, pt.join(pt.dirname(experiment_directory), 'x_real_test.torch'))
    pickle_it(x_real_train, pt.join(pt.dirname(experiment_directory), 'x_real_train.torch'))
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cpu').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    
    # Log some results at the end of training
    algo.plot_losses()
    savefig('losses.png', experiment_directory)
    return experiment_directory, value_range


def get_dataset_configuration(dataset, normalizer, trial_id="1"):
    if dataset == 'ECG':
        generator = [('id=100', dict(filenames=['100']))]
    elif dataset == 'STOCKS':
        generator = (('_'.join(asset), dict(assets=asset)) for asset in [('SPX',), ('SPX', 'DJI')])
    elif dataset == 'VAR':
        par1 = itertools.product([1], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8)])
        par2 = itertools.product([2], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8), (0.8, 0.2), (0.8, 0.5)])
        par3 = itertools.product([3], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8), (0.8, 0.2), (0.8, 0.5)])
        combinations = itertools.chain(par1, par2, par3)
        generator = (
            ('dim={}_phi={}_sigma={}'.format(dim, phi, sigma), dict(dim=dim, phi=phi, sigma=sigma))
            for dim, (phi, sigma) in combinations
        )
    elif dataset == 'ARCH':
        generator = (('lag={}'.format(lag), dict(lag=lag)) for lag in [3])
    elif dataset == 'SINE':
        generator = [('a', dict())]
    elif dataset == 'CUSTOM':
        types = ['city_bikes','omxhpi', 'weather_kumpula', 'covid19', 'fingrid']
        filenames = ['hsl_2021_05_5min.csv',
                     'omxhpi_20010306-20240515_daily.csv', 
                     'fmi_20230515_20240514_1h.csv', 
                     'thl_covid19_weekly.csv',
                     'fingrid_2023_3min.csv']
        generator = [(t, dict(filename=f)) for t, f in zip(types, filenames)]
    elif dataset in ['HSL', 'OMX', 'FMI', 'THL', 'ELEC']:
        custom_datasets = ['HSL', 'OMX', 'FMI', 'THL', 'ELEC']
        filenames = ['hsl_2021_05_5min.csv',
                     'omxhpi_20010306-20240515_daily.csv', 
                     'fmi_20230515_20240514_1h.csv', 
                     'thl_covid19_weekly.csv',
                     'fingrid_20231202-31_3min.csv']
        d = dict(zip(custom_datasets, filenames))
        generator = [(str(trial_id), dict(filename=d[dataset], normalizer=normalizer))] 
    else:
        raise Exception('%s not a valid data type.' % dataset)
    return generator

def get_device(args):
    return 'cuda:{}'.format(args.device) if args.use_cuda and torch.cuda.is_available() else 'cpu'

def main(args):
    if not pt.exists('./data'):
        os.mkdir('./data')
    #if not pt.exists('./data/oxfordmanrealizedvolatilityindices.csv'):
    #    print('Downloading Oxford MAN AHL realised library...')
    #    download_man_ahl_dataset()
    #if not pt.exists('./data/mitdb'):
    #    print('Downloading MIT-ECG database...')
    #    download_mit_ecg_dataset()

    print('Start of training. CUDA: %s' % args.use_cuda)
    for dataset in args.datasets:
        for algo_id in args.algos:
            for seed in range(args.initial_seed, args.initial_seed + args.num_seeds):
                base_config = BaseConfig(
                    device = get_device(args),
                    seed=seed,
                    batch_size=args.batch_size,
                    hidden_dims=tuple(args.hidden_dims),
                    p=args.p,
                    q=args.q,
                    total_steps=args.total_steps,
                    mc_samples=1000,
                    D_per_G_steps = args.D_per_G_steps,
                    lr_D = args.lr_D,
                    lr_G = args.lr_G                
                    )
                set_seed(seed)
                generator = get_dataset_configuration(dataset, args.normalizer, args.trial_id)
                for spec, data_params in generator:
                    if "batch_size" in data_params:
                        base_config.batch_size = data_params['batch_size']
                    experiment_dir, value_range = run(
                        algo_id=algo_id,
                        base_config=base_config,
                        data_params=data_params,
                        dataset=dataset,
                        base_dir=args.base_dir,
                        spec=spec,
                        args=args
                    )
                    if args.optim:
                        metrics: pd.DataFrame = evaluate_generator(algo_id, 
                                                                   seed, 
                                                                   experiment_dir, 
                                                                   dataset, 
                                                                   False,
                                                                   True,
                                                                   base_config,)
                        metrics["value_range"] = value_range
                        return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Meta parameters
    parser.add_argument('-base_dir', default='./numerical_results', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-device', default=1, type=int)
    parser.add_argument('-num_seeds', default=1, type=int)
    parser.add_argument('-initial_seed', default=0, type=int)
    #parser.add_argument('-datasets', default=['ARCH', 'STOCKS', 'ECG', 'VAR', ], nargs="+")
    parser.add_argument('-datasets', default=['HSL', 'FMI', 'OMX', 'ELEC', 'THL' ], nargs="+")
    #parser.add_argument('-algos', default=['SigCWGAN', 'GMMN', 'RCGAN', 'TimeGAN', 'RCWGAN', 'CWGAN',], nargs="+")
    parser.add_argument('-algos', default=['TimeGAN', 'RCGAN', 'RCWGAN', ], nargs="+")


    # Algo hyperparameters
    parser.add_argument('-batch_size', default=200, type=int)
    parser.add_argument('-p', default=3, type=int)
    parser.add_argument('-q', default=3, type=int)
    parser.add_argument('-hidden_dims', default=3 * (50,), type=int, nargs="+")
    parser.add_argument('-total_steps', default=1000, type=int)
    parser.add_argument('-D_per_G_steps', default=2, type=int)
    parser.add_argument('-lr_D', default=2e-4, type=float)
    parser.add_argument('-lr_G', default=1e-4, type=float)
    parser.add_argument('-lr_GMMN', default=1e-4, type=float)    
    parser.add_argument('-normalizer', default='minmax', type=str)
    parser.add_argument('-optim', action='store_true')
    parser.add_argument('-trial_id', default=1, type=int)
    parser.add_argument('-skip_plots', action='store_true')


    args = parser.parse_args()
    main(args)
