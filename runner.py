import pandas as pd
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from experiment_domainadapt_meanteacher_hila import experiment

dataset_options = ['svhn_mnist', 'mnist_svhn', 'svhn_mnist_rgb', 'mnist_svhn_rgb', 'cifar_stl',
                   'stl_cifar', 'mnist_usps', 'usps_mnist', 'syndigits_svhn', 'synsigns_gtsrb']


def runner(exp, confidence_thresh, rampup, teacher_alpha, unsup_weight, cls_balance, learning_rate,
               arch='', loss='var', double_softmax=False, fix_ema=False,
               cls_bal_scale=False, cls_bal_scale_range=0.0, cls_balance_loss='bce',
               combine_batches=False, standardise_samples=False,
               src_affine_std=0.0, src_xlat_range=0.0, src_hflip=False,
               src_intens_flip=False, src_intens_scale_range='', src_intens_offset_range='', src_gaussian_noise_std=0.1,
               tgt_affine_std=0.0, tgt_xlat_range=0.0, tgt_hflip=False,
               tgt_intens_flip='', tgt_intens_scale_range='', tgt_intens_offset_range='', tgt_gaussian_noise_std=0.1,
               num_epochs=100, batch_size=64, epoch_size='small', seed=0,
               log_file='', model_file='', device='cpu'):
    settings = locals().copy()

    import os
    import sys
    import pickle
    import cmdline_helpers


    # src_intens_scale_range_lower, src_intens_scale_range_upper, src_intens_offset_range_lower, src_intens_offset_range_upper = \
    #     cmdline_helpers.intens_aug_options(src_intens_scale_range, src_intens_offset_range)
    # tgt_intens_scale_range_lower, tgt_intens_scale_range_upper, tgt_intens_offset_range_lower, tgt_intens_offset_range_upper = \
    #     cmdline_helpers.intens_aug_options(tgt_intens_scale_range, tgt_intens_offset_range)
    src_intens_scale_range_lower, src_intens_scale_range_upper, src_intens_offset_range_lower, src_intens_offset_range_upper = \
        None, None, None, None
    tgt_intens_scale_range_lower, tgt_intens_scale_range_upper, tgt_intens_offset_range_lower, tgt_intens_offset_range_upper = \
        None, None, None, None

    import time
    import math
    import numpy as np
    from batchup import data_source, work_pool
    import data_loaders
    import standardisation
    import network_architectures
    import augmentation
    import torch, torch.cuda
    from torch import nn
    from torch.nn import functional as F
    import optim_weight_ema


    n_chn = 0


    if exp == 'svhn_mnist':
        d_source = data_loaders.load_svhn(zero_centre=False, greyscale=True)
        d_target = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True, val=False)
    elif exp == 'mnist_svhn':
        d_source = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True)
        d_target = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True, val=False)

        # d_target = data_loaders.load_svhn(zero_centre=False, greyscale=True, val=False)
    elif exp == 'svhn_mnist_rgb':
        d_source = data_loaders.load_svhn(zero_centre=False, greyscale=False)
        d_target = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True, val=False, rgb=True)
    elif exp == 'mnist_svhn_rgb':
        d_source = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True, rgb=True)
        d_target = data_loaders.load_svhn(zero_centre=False, greyscale=False, val=False)
    elif exp == 'cifar_stl':
        d_source = data_loaders.load_cifar10(range_01=False)
        d_target = data_loaders.load_stl(zero_centre=False, val=False)
    elif exp == 'stl_cifar':
        d_source = data_loaders.load_stl(zero_centre=False)
        d_target = data_loaders.load_cifar10(range_01=False, val=False)
    elif exp == 'mnist_usps':
        d_source = data_loaders.load_mnist(zero_centre=False)
        d_target = data_loaders.load_usps(zero_centre=False, scale28=True, val=False)
    elif exp == 'usps_mnist':
        d_source = data_loaders.load_usps(zero_centre=False, scale28=True)
        d_target = data_loaders.load_mnist(zero_centre=False, val=False)
    elif exp == 'syndigits_svhn':
        d_source = data_loaders.load_syn_digits(zero_centre=False)
        d_target = data_loaders.load_svhn(zero_centre=False, val=False)
    elif exp == 'synsigns_gtsrb':
        d_source = data_loaders.load_syn_signs(zero_centre=False)
        d_target = data_loaders.load_gtsrb(zero_centre=False, val=False)
    else:
        print('Unknown experiment type \'{}\''.format(exp))
        return

    source_x = np.array(list(d_source.train_X[:]) + list(d_source.test_X[:]))
    target_x = np.array(list(d_target.train_X[:]) + list(d_target.test_X[:]))
    source_y = np.array(list(d_source.train_y[:]) + list(d_source.test_y[:]))
    target_y = np.array(list(d_target.train_y[:]) + list(d_target.test_y[:]))

    print('Loaded data')
    return source_x, source_y, target_x, target_y


if __name__ == '__main__':
    source_x, source_y, target_x, target_y = runner(exp='mnist_svhn', confidence_thresh=0.96837722, rampup=0.0, teacher_alpha=0.99, unsup_weight=3.0, cls_balance=0.005, learning_rate=0.001)
    experiment(source_x, source_y, target_x, target_y, exp='mnist_svhn')

    for idx in range(10):
        num_iter = 50
        init_points = 5
        params = {
            'confidence_thresh':0.96837722, 'rampup':0.0, 'teacher_alpha':0.99, 'unsup_weight':3.0, 'cls_balance':0.005, 'learning_rate':0.001
        }

        domain_adapt_BO = BayesianOptimization(experiment, {'confidence_thresh':(0.5,1.0),
                                                  'rampup':(0.0, 0.5),
                                                  'teacher_alpha':(0.5,0.99),
                                                  'unsup_weight':(1.0,3.0),
                                                  'cls_balance':(0.005,0.01),
                                                  'learning_rate':(0.0001, 0.01)
                                                    })

        domain_adapt_BO.maximize(init_points=init_points, n_iter=num_iter)
