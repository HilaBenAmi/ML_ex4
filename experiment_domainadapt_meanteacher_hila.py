"""
Incorporates mean teacher, from:

Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results
Antti Tarvainen, Harri Valpola
https://arxiv.org/abs/1703.01780

"""
from bayes_opt import BayesianOptimization
import numpy as np
import data_loaders
from sklearn.model_selection import StratifiedKFold

import time
import math
import numpy as np
from batchup import data_source, work_pool
import network_architectures
import augmentation
import torch, torch.cuda
from torch import nn
from torch.nn import functional as F
import optim_weight_ema

import os
import pickle
import cmdline_helpers

INNER_K_FOLD = 2 # 3
OUTER_K_FOLD = 2 # 10
num_epochs = 1 # 100
bo_num_iter = 2  # 50
init_points = 5

exp = 'mnist_svhn'
log_file = f'results_exp_meanteacher_hila/log_{exp}_example_run.txt'
model_file = ''

seed = 0
device = 'cpu'
epoch_size = 'target'
batch_size = 60


torch_device = torch.device(device)
pool = work_pool.WorkerThreadPool(2)

# Setup output
def log(text):
    print(text)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(text + '\n')
            f.flush()
            f.close()


def runner(exp):
    settings = locals().copy()

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
    n_classes = d_source.n_classes

    print('Loaded data')
    return source_x, source_y, target_x, target_y, n_classes


def build_and_train_model(source_train_x_inner, source_train_y_inner, target_train_x_inner,
                source_validation_x, source_validation_y, target_validation_x, target_validation_y,
               confidence_thresh, rampup, teacher_alpha, unsup_weight, cls_balance, learning_rate,
               arch='', loss='var', double_softmax=False, fix_ema=False,
               cls_bal_scale=False, cls_bal_scale_range=0.0, cls_balance_loss='bce',
               combine_batches=False, standardise_samples=False,
               src_affine_std=0.0, src_xlat_range=0.0, src_hflip=False,
               src_intens_flip=False, src_gaussian_noise_std=0.1,
               tgt_affine_std=0.0, tgt_xlat_range=0.0, tgt_hflip=False,
               tgt_intens_flip='', tgt_gaussian_noise_std=0.1):

    net_class, expected_shape = network_architectures.get_net_and_shape_for_architecture(arch)

    settings = locals().copy()
    rampup = round(rampup)

    use_rampup = rampup > 0.0

    src_intens_scale_range_lower, src_intens_scale_range_upper, src_intens_offset_range_lower, src_intens_offset_range_upper = \
        None, None, None, None
    tgt_intens_scale_range_lower, tgt_intens_scale_range_upper, tgt_intens_offset_range_lower, tgt_intens_offset_range_upper = \
        None, None, None, None

    if expected_shape != source_train_x_inner.shape[1:]:
        print('Architecture {} not compatible with experiment {}; it needs samples of shape {}, '
              'data has samples of shape {}'.format(arch, exp, expected_shape, source_train_x_inner.shape[1:]))
        return

    student_net = net_class(n_classes).to(torch_device)
    teacher_net = net_class(n_classes).to(torch_device)
    student_params = list(student_net.parameters())
    teacher_params = list(teacher_net.parameters())
    for param in teacher_params:
        param.requires_grad = False

    student_optimizer = torch.optim.Adam(student_params, lr=learning_rate)
    if fix_ema:
        teacher_optimizer = optim_weight_ema.EMAWeightOptimizer(teacher_net, student_net, alpha=teacher_alpha)
    else:
        teacher_optimizer = optim_weight_ema.OldWeightEMA(teacher_net, student_net, alpha=teacher_alpha)
    classification_criterion = nn.CrossEntropyLoss()

    print('Built network')

    src_aug = augmentation.ImageAugmentation(
        src_hflip, src_xlat_range, src_affine_std,
        intens_flip=src_intens_flip,
        intens_scale_range_lower=src_intens_scale_range_lower, intens_scale_range_upper=src_intens_scale_range_upper,
        intens_offset_range_lower=src_intens_offset_range_lower,
        intens_offset_range_upper=src_intens_offset_range_upper,
        gaussian_noise_std=src_gaussian_noise_std
    )
    tgt_aug = augmentation.ImageAugmentation(
        tgt_hflip, tgt_xlat_range, tgt_affine_std,
        intens_flip=tgt_intens_flip,
        intens_scale_range_lower=tgt_intens_scale_range_lower, intens_scale_range_upper=tgt_intens_scale_range_upper,
        intens_offset_range_lower=tgt_intens_offset_range_lower,
        intens_offset_range_upper=tgt_intens_offset_range_upper,
        gaussian_noise_std=tgt_gaussian_noise_std
    )

    if combine_batches:
        def augment(X_sup, y_src, X_tgt):
            X_src_stu, X_src_tea = src_aug.augment_pair(X_sup)
            X_tgt_stu, X_tgt_tea = tgt_aug.augment_pair(X_tgt)
            return X_src_stu, X_src_tea, y_src, X_tgt_stu, X_tgt_tea
    else:
        def augment(X_src, y_src, X_tgt):
            X_src = src_aug.augment(X_src)
            X_tgt_stu, X_tgt_tea = tgt_aug.augment_pair(X_tgt)
            return X_src, y_src, X_tgt_stu, X_tgt_tea

    rampup_weight_in_list = [0]

    cls_bal_fn = network_architectures.get_cls_bal_function(cls_balance_loss)

    def compute_aug_loss(stu_out, tea_out):
        # Augmentation loss
        if use_rampup:
            unsup_mask = None
            conf_mask_count = None
            unsup_mask_count = None
        else:
            conf_tea = torch.max(tea_out, 1)[0]
            unsup_mask = conf_mask = (conf_tea > confidence_thresh).float()
            unsup_mask_count = conf_mask_count = conf_mask.sum()

        if loss == 'bce':
            aug_loss = network_architectures.robust_binary_crossentropy(stu_out, tea_out)
        else:
            d_aug_loss = stu_out - tea_out
            aug_loss = d_aug_loss * d_aug_loss

        # Class balance scaling
        if cls_bal_scale:
            if use_rampup:
                n_samples = float(aug_loss.shape[0])
            else:
                n_samples = unsup_mask.sum()
            avg_pred = n_samples / float(n_classes)
            bal_scale = avg_pred / torch.clamp(tea_out.sum(dim=0), min=1.0)
            if cls_bal_scale_range != 0.0:
                bal_scale = torch.clamp(bal_scale, min=1.0 / cls_bal_scale_range, max=cls_bal_scale_range)
            bal_scale = bal_scale.detach()
            aug_loss = aug_loss * bal_scale[None, :]

        aug_loss = aug_loss.mean(dim=1)

        if use_rampup:
            unsup_loss = aug_loss.mean() * rampup_weight_in_list[0]
        else:
            unsup_loss = (aug_loss * unsup_mask).mean()

        # Class balance loss
        if cls_balance > 0.0:
            # Compute per-sample average predicated probability
            # Average over samples to get average class prediction
            avg_cls_prob = stu_out.mean(dim=0)
            # Compute loss
            equalise_cls_loss = cls_bal_fn(avg_cls_prob, float(1.0 / n_classes))

            equalise_cls_loss = equalise_cls_loss.mean() * n_classes

            if use_rampup:
                equalise_cls_loss = equalise_cls_loss * rampup_weight_in_list[0]
            else:
                if rampup == 0:
                    equalise_cls_loss = equalise_cls_loss * unsup_mask.mean(dim=0)

            unsup_loss += equalise_cls_loss * cls_balance

        return unsup_loss, conf_mask_count, unsup_mask_count

    if combine_batches:
        def f_train(X_src0, X_src1, y_src, X_tgt0, X_tgt1):
            X_src0 = torch.tensor(X_src0, dtype=torch.float, device=torch_device)
            X_src1 = torch.tensor(X_src1, dtype=torch.float, device=torch_device)
            y_src = torch.tensor(y_src, dtype=torch.long, device=torch_device)
            X_tgt0 = torch.tensor(X_tgt0, dtype=torch.float, device=torch_device)
            X_tgt1 = torch.tensor(X_tgt1, dtype=torch.float, device=torch_device)

            n_samples = X_src0.size()[0]
            n_total = n_samples + X_tgt0.size()[0]

            student_optimizer.zero_grad()
            student_net.train()
            teacher_net.train()

            # Concatenate source and target mini-batches
            X0 = torch.cat([X_src0, X_tgt0], 0)
            X1 = torch.cat([X_src1, X_tgt1], 0)

            student_logits_out = student_net(X0)
            student_prob_out = F.softmax(student_logits_out, dim=1)

            src_logits_out = student_logits_out[:n_samples]
            src_prob_out = student_prob_out[:n_samples]

            teacher_logits_out = teacher_net(X1)
            teacher_prob_out = F.softmax(teacher_logits_out, dim=1)

            # Supervised classification loss
            if double_softmax:
                clf_loss = classification_criterion(src_prob_out, y_src)
            else:
                clf_loss = classification_criterion(src_logits_out, y_src)

            unsup_loss, conf_mask_count, unsup_mask_count = compute_aug_loss(student_prob_out, teacher_prob_out)

            loss_expr = clf_loss + unsup_loss * unsup_weight

            loss_expr.backward()
            student_optimizer.step()
            teacher_optimizer.step()

            outputs = [float(clf_loss) * n_samples, float(unsup_loss) * n_total]
            if not use_rampup:
                mask_count = float(conf_mask_count) * 0.5
                unsup_count = float(unsup_mask_count) * 0.5

                outputs.append(mask_count)
                outputs.append(unsup_count)
            return tuple(outputs)
    else:
        def f_train(X_src, y_src, X_tgt0, X_tgt1):
            X_src = torch.tensor(X_src, dtype=torch.float, device=torch_device)
            y_src = torch.tensor(y_src, dtype=torch.long, device=torch_device)
            X_tgt0 = torch.tensor(X_tgt0, dtype=torch.float, device=torch_device)
            X_tgt1 = torch.tensor(X_tgt1, dtype=torch.float, device=torch_device)

            student_optimizer.zero_grad()
            student_net.train()
            teacher_net.train()

            src_logits_out = student_net(X_src)
            student_tgt_logits_out = student_net(X_tgt0)
            student_tgt_prob_out = F.softmax(student_tgt_logits_out, dim=1)
            teacher_tgt_logits_out = teacher_net(X_tgt1)
            teacher_tgt_prob_out = F.softmax(teacher_tgt_logits_out, dim=1)

            # Supervised classification loss
            if double_softmax:
                clf_loss = classification_criterion(F.softmax(src_logits_out, dim=1), y_src)
            else:
                clf_loss = classification_criterion(src_logits_out, y_src)

            unsup_loss, conf_mask_count, unsup_mask_count = compute_aug_loss(student_tgt_prob_out, teacher_tgt_prob_out)

            loss_expr = clf_loss + unsup_loss * unsup_weight

            loss_expr.backward()
            student_optimizer.step()
            teacher_optimizer.step()

            n_samples = X_src.size()[0]

            outputs = [float(clf_loss) * n_samples, float(unsup_loss) * n_samples]
            if not use_rampup:
                mask_count = float(conf_mask_count)
                unsup_count = float(unsup_mask_count)

                outputs.append(mask_count)
                outputs.append(unsup_count)
            return tuple(outputs)

    print('Compiled training function')

    def f_pred_src(X_sup):
        X_var = torch.tensor(X_sup, dtype=torch.float, device=torch_device)
        student_net.eval()
        teacher_net.eval()
        return (F.softmax(student_net(X_var), dim=1).detach().cpu().numpy(),
                F.softmax(teacher_net(X_var), dim=1).detach().cpu().numpy())

    def f_pred_tgt(X_sup):
        X_var = torch.tensor(X_sup, dtype=torch.float, device=torch_device)
        student_net.eval()
        teacher_net.eval()
        return (F.softmax(student_net(X_var), dim=1).detach().cpu().numpy(),
                F.softmax(teacher_net(X_var), dim=1).detach().cpu().numpy())

    def f_eval_src(X_sup, y_sup):
        y_pred_prob_stu, y_pred_prob_tea = f_pred_src(X_sup)
        y_pred_stu = np.argmax(y_pred_prob_stu, axis=1)
        y_pred_tea = np.argmax(y_pred_prob_tea, axis=1)
        return (float((y_pred_stu != y_sup).sum()), float((y_pred_tea != y_sup).sum()))

    def f_eval_tgt(X_sup, y_sup):
        y_pred_prob_stu, y_pred_prob_tea = f_pred_tgt(X_sup)
        y_pred_stu = np.argmax(y_pred_prob_stu, axis=1)
        y_pred_tea = np.argmax(y_pred_prob_tea, axis=1)
        return (float((y_pred_stu != y_sup).sum()), float((y_pred_tea != y_sup).sum()))

    print('Compiled evaluation function')

    cmdline_helpers.ensure_containing_dir_exists(log_file)

    # Report setttings
    log('Settings: {}'.format(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))])))

    # Report dataset size
    log('Dataset:')
    log('SOURCE Train: X.shape={}, y.shape={}'.format(source_train_x_inner.shape, source_train_y_inner.shape))
    log('SOURCE Test: X.shape={}, y.shape={}'.format(source_validation_x.shape, source_validation_y.shape))
    log('TARGET Train: X.shape={}'.format(target_train_x_inner.shape))
    log('TARGET Test: X.shape={}, y.shape={}'.format(target_validation_x.shape, target_validation_y.shape))

    print('Training...')
    sup_ds = data_source.ArrayDataSource([source_train_x_inner, source_train_y_inner], repeats=-1)
    tgt_train_ds = data_source.ArrayDataSource([target_train_x_inner], repeats=-1)
    train_ds = data_source.CompositeDataSource([sup_ds, tgt_train_ds]).map(augment)
    train_ds = pool.parallel_data_source(train_ds)
    if epoch_size == 'large':
        n_samples = max(source_train_x_inner.shape[0], target_train_x_inner.shape[0])
    elif epoch_size == 'small':
        n_samples = min(source_train_x_inner.shape[0], target_train_x_inner.shape[0])
    elif epoch_size == 'target':
        n_samples = target_train_x_inner.shape[0]
    n_train_batches = n_samples // batch_size

    source_test_ds = data_source.ArrayDataSource([source_validation_x, source_validation_y])
    target_test_ds = data_source.ArrayDataSource([target_validation_x, target_validation_y])

    if seed != 0:
        shuffle_rng = np.random.RandomState(seed)
    else:
        shuffle_rng = np.random

    train_batch_iter = train_ds.batch_iterator(batch_size=batch_size, shuffle=shuffle_rng)

    best_teacher_model_state = {k: v.cpu().numpy() for k, v in teacher_net.state_dict().items()}

    best_conf_mask_rate = 0.0
    best_src_test_err = 1.0
    for epoch in range(num_epochs):
        t1 = time.time()

        if use_rampup:
            if epoch < rampup:
                p = max(0.0, float(epoch)) / float(rampup)
                p = 1.0 - p
                rampup_value = math.exp(-p * p * 5.0)
            else:
                rampup_value = 1.0

            rampup_weight_in_list[0] = rampup_value

        train_res = data_source.batch_map_mean(f_train, train_batch_iter, n_batches=n_train_batches)

        train_clf_loss = train_res[0]
        if combine_batches:
            unsup_loss_string = 'unsup (both) loss={:.6f}'.format(train_res[1])
        else:
            unsup_loss_string = 'unsup (tgt) loss={:.6f}'.format(train_res[1])

        src_test_err_stu, src_test_err_tea = source_test_ds.batch_map_mean(f_eval_src, batch_size=batch_size * 2)
        tgt_test_err_stu, tgt_test_err_tea = target_test_ds.batch_map_mean(f_eval_tgt, batch_size=batch_size * 2)

        if use_rampup:
            unsup_loss_string = '{}, rampup={:.3%}'.format(unsup_loss_string, rampup_value)
            if src_test_err_stu < best_src_test_err:
                best_src_test_err = src_test_err_stu
                best_teacher_model_state = {k: v.cpu().numpy() for k, v in teacher_net.state_dict().items()}
                improve = '*** '
                best_target_tea_err = tgt_test_err_tea
            else:
                improve = ''
        else:
            conf_mask_rate = train_res[-2]
            unsup_mask_rate = train_res[-1]
            if conf_mask_rate > best_conf_mask_rate:
                best_conf_mask_rate = conf_mask_rate
                improve = '*** '
                best_teacher_model_state = {k: v.cpu().numpy() for k, v in teacher_net.state_dict().items()}
                best_target_tea_err = tgt_test_err_tea
            else:
                improve = ''
            unsup_loss_string = '{}, conf mask={:.3%}, unsup mask={:.3%}'.format(
                unsup_loss_string, conf_mask_rate, unsup_mask_rate)

        t2 = time.time()

        log('{}Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}, {}; '
            'SRC TEST ERR={:.3%}, TGT TEST student err={:.3%}, TGT TEST teacher err={:.3%}'.format(
            improve, epoch, t2 - t1, train_clf_loss, unsup_loss_string, src_test_err_stu, tgt_test_err_stu,
            tgt_test_err_tea))

    return best_target_tea_err, best_teacher_model_state


def get_arch(exp, arch):
    if arch == '':
        if exp in {'mnist_usps', 'usps_mnist'}:
            arch = 'mnist-bn-32-64-256'
        if exp in {'svhn_mnist', 'mnist_svhn'}:
            arch = 'grey-32-64-128-gp'
        if exp in {'cifar_stl', 'stl_cifar', 'syndigits_svhn', 'svhn_mnist_rgb', 'mnist_svhn_rgb'}:
            arch = 'rgb-128-256-down-gp'
        if exp in {'synsigns_gtsrb'}:
            arch = 'rgb40-96-192-384-gp'
    return arch


def evaluate_exp(confidence_thresh, rampup, teacher_alpha, unsup_weight, cls_balance, learning_rate,
               arch='', loss='var', double_softmax=False, fix_ema=False,
               cls_bal_scale=False, cls_bal_scale_range=0.0, cls_balance_loss='bce',
               combine_batches=False, standardise_samples=False,
               src_affine_std=0.0, src_xlat_range=0.0, src_hflip=False,
               src_intens_flip=False, src_gaussian_noise_std=0.1,
               tgt_affine_std=0.0, tgt_xlat_range=0.0, tgt_hflip=False,
               tgt_intens_flip='', tgt_gaussian_noise_std=0.1):

    arch = get_arch(exp, arch)

    cv_source = StratifiedKFold(n_splits=INNER_K_FOLD, shuffle=True)
    source_train_validation_list = []
    for train_idx, validation_idx in cv_source.split(source_x, source_y):
        source_dict = {}
        train_data, validation_data = source_x[train_idx], source_x[validation_idx]
        train_target, validation_target = source_y[train_idx], source_y[validation_idx]
        source_dict['source_train_x'] = train_data
        source_dict['source_train_y'] = train_target
        source_dict['source_validation_x'] = validation_data
        source_dict['source_validation_y'] = validation_target
        source_train_validation_list.append(source_dict)

    cv_target = StratifiedKFold(n_splits=INNER_K_FOLD, shuffle=True)
    target_train_validation_list = []
    for train_idx, validation_idx in cv_target.split(target_x, target_y):
        target_dict = {}
        train_data, validation_data = target_x[train_idx], target_x[validation_idx]
        validation_target = target_y[validation_idx]
        target_dict['target_train_x'] = train_data
        target_dict['target_validation_x'] = validation_data
        target_dict['target_validation_y'] = validation_target
        target_train_validation_list.append(target_dict)

    target_test_err_list = []
    for cv_idx in range(INNER_K_FOLD):
        source_train_x_inner = source_train_validation_list[cv_idx]['source_train_x']
        source_train_y_inner = source_train_validation_list[cv_idx]['source_train_y']
        source_validation_x = source_train_validation_list[cv_idx]['source_validation_x']
        source_validation_y = source_train_validation_list[cv_idx]['source_validation_y']
        target_train_x_inner = target_train_validation_list[cv_idx]['target_train_x']
        target_validation_x = target_train_validation_list[cv_idx]['target_validation_x']
        target_validation_y = target_train_validation_list[cv_idx]['target_validation_y']

        best_target_tea_err, best_teacher_model_state = build_and_train_model(
            source_train_x_inner, source_train_y_inner, target_train_x_inner,
            source_validation_x, source_validation_y, target_validation_x, target_validation_y,
            confidence_thresh, rampup, teacher_alpha, unsup_weight, cls_balance, learning_rate,
            arch)
        target_test_err_list.append(best_target_tea_err)

        # Save network
        if model_file != '':
            cmdline_helpers.ensure_containing_dir_exists(model_file)
            with open(model_file, 'wb') as f:
                pickle.dump(best_teacher_model_state, f)

    return -np.mean(target_test_err_list)


def rebuild_and_test_model(params, source_train_x, source_train_y, target_train_x, source_test_x,
                           source_test_y, target_test_x, target_test_y, arch=''):
    arch = get_arch(exp, arch)
    best_target_tea_err, best_teacher_model_state = build_and_train_model(
        source_train_x, source_train_y, target_train_x, source_test_x, source_test_y, target_test_x, target_test_y,
        arch=arch, **params)
    return best_target_tea_err



if __name__ == '__main__':
    # confidence_thresh = 0.96837722, rampup = 0.0, teacher_alpha = 0.99, unsup_weight = 3.0, cls_balance = 0.005, learning_rate = 0.001
    global source_x, source_y, target_x, target_y, n_classes

    source_x, source_y, target_x, target_y, n_classes = runner(exp=exp)

    if log_file == '':
        log_file = 'output_aug_log_{}.txt'.format(exp)
    elif log_file == 'none':
        log_file = None

    if log_file is not None:
        if os.path.exists(log_file):
            msg = 'Output log file {} already exists'.format(log_file)
            print(msg)
            raise Exception(msg)

    cv_source = StratifiedKFold(n_splits=OUTER_K_FOLD, shuffle=True)
    source_train_test_list = []
    for train_idx, test_idx in cv_source.split(source_x, source_y):
        source_dict = {}
        train_data, test_data = source_x[train_idx], source_x[test_idx]
        train_target, test_target = source_y[train_idx], source_y[test_idx]
        source_dict['source_train_x'] = train_data
        source_dict['source_train_y'] = train_target
        source_dict['source_test_x'] = test_data
        source_dict['source_test_y'] = test_target
        source_train_test_list.append(source_dict)

    cv_target = StratifiedKFold(n_splits=OUTER_K_FOLD, shuffle=True)
    target_train_test_list = []
    for train_idx, test_idx in cv_target.split(target_x, target_y):
        target_dict = {}
        train_data, test_data = target_x[train_idx], target_x[test_idx]
        train_target, test_target = target_y[train_idx], target_y[test_idx]
        target_dict['target_train_x'] = train_data
        target_dict['target_train_y'] = train_target
        target_dict['target_test_x'] = test_data
        target_dict['target_test_y'] = test_target
        target_train_test_list.append(target_dict)

    target_test_err_list = []
    for cv_idx in range(OUTER_K_FOLD):
        source_train_x = source_train_test_list[cv_idx]['source_train_x']
        source_train_y = source_train_test_list[cv_idx]['source_train_y']
        source_test_x = source_train_test_list[cv_idx]['source_test_x']
        source_test_y = source_train_test_list[cv_idx]['source_test_y']
        target_train_x = target_train_test_list[cv_idx]['target_train_x']
        target_train_y = target_train_test_list[cv_idx]['target_train_y']
        target_test_x = target_train_test_list[cv_idx]['target_test_x']
        target_test_y = target_train_test_list[cv_idx]['target_test_y']

        domain_adapt_BO = BayesianOptimization(evaluate_exp, {'confidence_thresh': (0.5, 1.0),
                                                            'rampup': (0.0, 100.0),
                                                            'teacher_alpha': (0.5, 0.99),
                                                            'unsup_weight': (1.0, 3.0),
                                                            'cls_balance': (0.005, 0.01),
                                                            'learning_rate': (0.0001, 0.01)
                                                            })
        domain_adapt_BO.maximize(init_points=init_points, n_iter=bo_num_iter)
        params_domain_adapt = domain_adapt_BO.max['params']
        params_domain_adapt['rampup'] = round(params_domain_adapt['rampup'])

        log(f'CV {cv_idx} - Opt hyper params: {params_domain_adapt}')

        # todo- more metrics on the test datasets
        best_target_tea_err = rebuild_and_test_model(params_domain_adapt, source_train_x, source_train_y, target_train_x, source_test_x,
                           source_test_y, target_test_x, target_test_y)
        log(f'TEST target res on teacher: {best_target_tea_err}')
