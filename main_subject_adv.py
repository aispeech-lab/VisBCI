# -*- coding: utf-8 -*-
# Created on 2020/10
# Author: NZY & XJM

import pickle
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import random
import time
from logger import Logger
import os
import shutil
import math
from utils import Adjust_lr, Early_stop
from data_processing import get_data_from_files
from models_adv import ActionPredBasedEEG
from sklearn.model_selection import train_test_split
import config_adv as config
from sklearn.model_selection import KFold


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%5dm %2ds' % (m, s)


def iterate_minibatch_train(inputs, labels, batchsize, shuffle=False):
    """
    through yield，divide the inputs（reps）into batchsize to output
    inputs = (sample_size, 12 erp, 28 channels, 600 ms/erp)
    labels = (sample_size)
    """
    assert inputs.shape[0] == labels.shape[0]
    indices = np.arange(inputs.shape[0])
    batch_num = int(len(indices) / batchsize)
    batch_plus = len(indices) % batchsize
    if shuffle:
        np.random.shuffle(indices)
        indices = indices[0: batch_num * batchsize]
        print('Train data random reduces %d samples for batch training with total samples %d\n'
              % (batch_plus, (batch_num * batchsize)))
    else:
        if batch_plus > 0:
            indices = indices[0: batch_num * batchsize]
        print('Test data random reduces %d samples for batch predict with total samples %d\n'
              % (batch_plus, (batch_num * batchsize)))

    for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], labels[excerpt]


def iterate_minibatch_eval(inputs, labels, batchsize, shuffle=False):
    """
    through yield，divide the inputs（reps）into batchsize to output
    inputs = (sample_size, 12 erp, 28 channels, 600 ms/erp)
    labels = (sample_size)
    """
    assert inputs.shape[0] == labels.shape[0]
    indices = np.arange(inputs.shape[0])
    batch_num = int(len(indices) / batchsize)
    batch_plus = len(indices) % batchsize
    if shuffle:
        np.random.shuffle(indices)
        indices = indices[0: batch_num * batchsize]
        print('Train data random reduces %d samples for batch training with total samples %d\n'
              % (batch_plus, (batch_num * batchsize)))
    else:
        if batch_plus > 0:
            indices = indices[0: batch_num * batchsize]
        print('Valid data random reduces %d samples for batch predict with total samples %d\n'
              % (batch_plus, (batch_num * batchsize)))

    for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], labels[excerpt]


def iterate_minibatch_test(inputs, batchsize, event_action_all, y_trials_all, shuffle=False):

    assert inputs.shape[0] == y_trials_all.shape[0]
    indices = np.arange(inputs.shape[0])
    batch_num = int(len(indices) / batchsize)
    batch_plus = len(indices) % batchsize
    if shuffle:
        np.random.shuffle(indices)
        indices = indices[0: batch_num * batchsize]
        print('Train data random reduces %d samples for batch training with total samples %d\n'
              % (batch_plus, (batch_num * batchsize)))
    else:
        if batch_plus > 0:
            indices = indices[0: batch_num * batchsize]
        print('Test data random reduces %d samples for batch predict with total samples %d\n'
              % (batch_plus, (batch_num * batchsize)))

    for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], event_action_all[excerpt], y_trials_all[excerpt]


def gen_minibatch_train(x_data_all, y_label_all, mini_batch_size, shuffle=True):
    for x_data, y_label in iterate_minibatch_train(x_data_all, y_label_all, mini_batch_size, shuffle=shuffle):
        if torch.cuda.is_available():
            yield (Variable(torch.from_numpy(x_data), requires_grad=False)).to(torch.float).cuda() \
                , (Variable(torch.from_numpy(y_label), requires_grad=False)).to(torch.long).cuda()
        else:
            yield (Variable(torch.from_numpy(x_data), requires_grad=False)).to(torch.float) \
                , (Variable(torch.from_numpy(y_label), requires_grad=False)).to(torch.long)


def gen_minibatch_eval(x_data_all, y_label_all, mini_batch_size, shuffle=False):
    for x_data, y_label in iterate_minibatch_eval(x_data_all, y_label_all, mini_batch_size, shuffle=shuffle):
        if torch.cuda.is_available():
            yield (Variable(torch.from_numpy(x_data), requires_grad=False)).to(torch.float).cuda() \
                , (Variable(torch.from_numpy(y_label), requires_grad=False)).to(torch.long).cuda()
        else:
            yield (Variable(torch.from_numpy(x_data), requires_grad=False)).to(torch.float) \
                , (Variable(torch.from_numpy(y_label), requires_grad=False)).to(torch.long)


def gen_minibatch_test(x_data_all, event_action_all, y_trials_all, mini_batch_size, shuffle=False):
    for x_data, event_action_all, y_trials_all in iterate_minibatch_test(x_data_all, mini_batch_size, event_action_all,
                                                                         y_trials_all, shuffle=shuffle):
        if torch.cuda.is_available():
            yield (Variable(torch.from_numpy(x_data), requires_grad=False)).to(torch.float).cuda() \
                , (Variable(torch.from_numpy(event_action_all), requires_grad=False)).to(torch.float).cuda() \
                , (Variable(torch.from_numpy(y_trials_all), requires_grad=False)).to(torch.long).cuda()
        else:
            yield (Variable(torch.from_numpy(x_data), requires_grad=False)).to(torch.float) \
                , (Variable(torch.from_numpy(event_action_all), requires_grad=False)).to(torch.float) \
                , (Variable(torch.from_numpy(y_trials_all), requires_grad=False)).to(torch.long)


def train_model(epoch_idx, optimizer, criterion_action, iter_step, model, train_data_gen, logger,
                start_time):
    model.train()
    train_loss_epoch = []
    train_acc_epoch = []
    while True:
        try:
            inputs, labels = train_data_gen.__next__()
            # inputs = (batch_size, 12 erp, 28 channels, 600 ms / erp)
            # labels = (batch_size, 1)
            labels = labels.squeeze(1)

            if config.ADD_ADVERSARIAL_ATTACK:
                if config.USE_CUDA:  # Random initialization every time, Approximately equal to the clean data
                    delta = torch.zeros_like(inputs).uniform_(-config.ATTACK_EPS, config.ATTACK_EPS).cuda()
                else:
                    delta = torch.zeros_like(inputs).uniform_(-config.ATTACK_EPS, config.ATTACK_EPS)
                delta.requires_grad = True
                pred_action = model(inputs + delta, False)
                if torch.cuda.is_available():
                    # pred_action: (batch_size, sent_class)
                    loss_action = criterion_action(pred_action.cuda(), labels)
                else:
                    # pred_action: (batch_size, sent_class)
                    loss_action = criterion_action(pred_action, labels)
                loss_action.backward()
                grad = delta.grad.detach()
                grad = torch.nn.functional.normalize(grad, p=2, dim=3, eps=1e-12)
                delta.data = config.ATTACK_ALPHA * grad
                delta = delta.detach()
                inputs_adv = inputs + delta
                inputs_adv = torch.clamp(inputs_adv, -1, 1)

                pred_action_adv = model(inputs_adv, True)
                pred_action = model(inputs, False)
                if torch.cuda.is_available():
                    # pred_action: (batch_size, sent_class)
                    loss_action_adv = criterion_action(pred_action_adv.cuda(), labels)
                    loss_action = criterion_action(pred_action.cuda(), labels)
                    loss_action = loss_action_adv + loss_action
                else:
                    # pred_action: (batch_size, sent_class)
                    loss_action_adv = criterion_action(pred_action_adv, labels)
                    loss_action = criterion_action(pred_action, labels)
                    loss_action = loss_action_adv + loss_action
            else:
                pred_action = model(inputs, False)
                if torch.cuda.is_available():
                    # pred_action: (batch_size, sent_class)
                    loss_action = criterion_action(pred_action.cuda(), labels)
                else:
                    # pred_action: (batch_size, sent_class)
                    loss_action = criterion_action(pred_action, labels)
            optimizer.zero_grad()
            loss_action.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # pred_action: (batch_size, sent_class)
            action_pred_class = torch.argmax(pred_action, dim=1)
            # action_pred_class: (batch_size, )
            action_pred_class = action_pred_class.data.cpu().numpy()
            labels_class = labels.data.cpu().numpy()
            num_s_correct = sum(action_pred_class == labels_class)
            output_acc = float(num_s_correct) / len(action_pred_class)

            logger.log_training(loss_action.cpu().item(), output_acc, iter_step)

            print('####, Epoch:%05d, Iter:%08d, Time:%s, loss:%+07.5f, acc:%+07.5f' % (
                epoch_idx, iter_step, timeSince(start_time), loss_action.cpu().item(), output_acc))
            train_loss_epoch.append(loss_action.cpu().item())
            train_acc_epoch.append(output_acc)
            iter_step += 1
        except StopIteration:
            break
    return iter_step, np.mean(train_loss_epoch), np.mean(train_acc_epoch)


def valid_model(epoch_idx, criterion_action, model, valid_data_gen, logger, start_time):
    model.eval()
    valid_loss_epoch = []
    valid_acc_epoch = []
    valid_step = 0
    with torch.no_grad():
        while True:
            try:
                inputs, labels = valid_data_gen.__next__()
                # inputs = (batch_size, 12 erp, 28 channels, 600 ms / erp)
                # labels = (batch_size, 1)
                pred_action = model(inputs, False)
                labels = labels.squeeze(1)
                if torch.cuda.is_available():
                    # pred_action: (batch_size, sent_class)
                    loss_action = criterion_action(pred_action.cuda(), labels)
                else:
                    # pred_action: (batch_size, sent_class)
                    loss_action = criterion_action(pred_action, labels)
                # pred_action: (batch_size, sent_class)
                action_pred_class = torch.argmax(pred_action, dim=1)
                # action_pred_class: (batch_size, )
                action_pred_class = action_pred_class.data.cpu().numpy()
                labels_class = labels.data.cpu().numpy()
                num_s_correct = sum(action_pred_class == labels_class)
                output_acc = float(num_s_correct) / len(action_pred_class)
                print('#### Valid:%08d, Time:%s, loss:%+07.5f, acc:%+07.5f' % (
                    valid_step, timeSince(start_time), loss_action.cpu().item(), output_acc))
                valid_loss_epoch.append(loss_action.cpu().item())
                valid_acc_epoch.append(output_acc)
                valid_step += 1
            except StopIteration:
                break
    logger.log_validation(np.mean(valid_loss_epoch), np.mean(valid_acc_epoch), epoch_idx)
    return np.mean(valid_loss_epoch), np.mean(valid_acc_epoch)


def test_model(criterion_action, model, repetitions, test_data_gen, logger, start_time):
    model.eval()
    test_loss_epoch = []
    test_acc_epoch = []
    test_step = 0
    with torch.no_grad():
        while True:
            try:
                inputs, event_action_all, trials = test_data_gen.__next__()
                # inputs = (batch_size, repetitions, 12 stimulus, 28 channels, 600 ms / erp) first-d: trials
                # event_action_all = (batch_size, repetitions, 12 stimulus)
                # trials = (batch_size, 1)
                inputs = inputs.reshape(-1, 12, 28, 600)
                # inputs = (batch_size, 12 stimulus, 28 channels, 600 ms / erp) first-d: reps
                pred_action = model(inputs, False)
                # pred_action: (batch, 12 log_softmax)  first-d: reps
                pred_action = pred_action.reshape(-1, repetitions, 12)
                # pred_action: (batch, reps, 12log_softmax)  first-d: trials

                # integrate all repetitions in a trial for joint voting
                # pred_action_sum：(trials, 12 stimulus_value)
                len_first = event_action_all.shape[0]
                pred_action_sum = np.zeros((len_first, 12))
                pred_action_sum = torch.Tensor(pred_action_sum)  # array -> tensor
                if repetitions == 1:  # no need for joint voting
                    for batch_idx in range(len_first):
                        for compare_itself in range(12):
                            stimulus_value = int(event_action_all[batch_idx, (repetitions - 1), compare_itself])
                            pred_action_sum[batch_idx, (stimulus_value - 1)] = pred_action[
                                batch_idx, (repetitions - 1), compare_itself]
                else:  # joint voting
                    for batch_idx in range(len_first):
                        for rep_idx in range(repetitions - 1):
                            for compare_first in range(12):  # the stimulus label of former rep for comparison
                                for compare_second in range(12):  # the stimulus label of latter rep for comparison
                                    if rep_idx == 0:
                                        if event_action_all[batch_idx, rep_idx, compare_first] == event_action_all[
                                             batch_idx, rep_idx + 1, compare_second]:
                                            stimulus_value = int(
                                                event_action_all[batch_idx, (rep_idx + 1), compare_second])
                                            pred_action_sum[batch_idx, (stimulus_value - 1)] = pred_action[
                                                batch_idx, rep_idx, compare_first]
                                    if event_action_all[batch_idx, rep_idx, compare_first] == event_action_all[
                                         batch_idx, rep_idx + 1, compare_second]:
                                        stimulus_value = int(event_action_all[batch_idx, (rep_idx + 1), compare_second])
                                        pred_action_sum[batch_idx, (stimulus_value - 1)] = pred_action_sum[batch_idx, (
                                                stimulus_value - 1)] + pred_action[batch_idx, (
                                                rep_idx + 1), compare_second]
                                        #  (batch, reps, 12 log_softmax) -> (batch, stimulus_value)

                trials = trials.squeeze(1)
                if torch.cuda.is_available():
                    loss_action = criterion_action(pred_action_sum.cuda(), trials)
                else:
                    loss_action = criterion_action(pred_action_sum, trials)
                # pred_action_sum: (batch, stimulus_value)
                pred_action_class = torch.argmax(pred_action_sum, dim=1)
                # pred_action_class: (batch_size, 1)
                pred_action_class = pred_action_class.data.cpu().numpy()
                trials_class = trials.data.cpu().numpy()
                num_s_correct = sum(pred_action_class == trials_class)
                output_acc = float(num_s_correct) / len(pred_action_class)
                print('#### Test:%08d, Time:%s, loss:%+07.5f, acc:%+07.5f' % (
                    test_step, timeSince(start_time), loss_action.cpu().item(), output_acc))
                test_loss_epoch.append(loss_action.cpu().item())
                test_acc_epoch.append(output_acc)
                test_step += 1
            except StopIteration:
                break
    logger.log_test(np.mean(test_loss_epoch), np.mean(test_acc_epoch), repetitions)
    return np.mean(test_loss_epoch), np.mean(test_acc_epoch)


def shape_for_kfold(x_data_all_x1, y_label_all_x1, x_data_all_x2, y_label_all_x2, event_action_all_x):
    x_data_all_x1 = x_data_all_x1.reshape(-1, 10, 12, 28, 600)
    y_label_all_x1 = y_label_all_x1.reshape(-1, 10, 1)
    x_data_all_x2 = x_data_all_x2.reshape(-1, 10, 12, 28, 600)
    y_label_all_x2 = y_label_all_x2.reshape(-1, 10, 1)
    event_action_all_x = event_action_all_x.reshape(-1, 10, 12)
    return x_data_all_x1, y_label_all_x1, x_data_all_x2, y_label_all_x2, event_action_all_x


def reshape_for_train_eval(train_data, train_label, eval_data, eval_label):
    # for train dataset
    train_data = train_data.reshape(-1, 12, 28, 600)  # [0]reps
    train_label = train_label.reshape(-1, 1)
    # for valid dataset
    eval_data = eval_data.reshape(-1, 12, 28, 600)
    eval_label = eval_label.reshape(-1, 1)
    return train_data, train_label, eval_data, eval_label


def generate_train_valid_test(x_data_all_train, y_label_all_train, x_data_all_test, y_label_all_test,
                              event_action_all, y_trials_all, train_index, test_index):
    train_data = x_data_all_train[train_index]  # train state
    train_label = y_label_all_train[train_index]
    eval_data = x_data_all_test[train_index]  # test state
    eval_label = y_label_all_test[train_index]
    test_data = x_data_all_test[test_index]  # test state
    test_trials = y_trials_all[test_index]
    event_action_test = event_action_all[test_index]

    train_data, unused, unused, eval_data, train_label, unused, unused, eval_label \
        = train_test_split(train_data, eval_data, train_label, eval_label,
                           random_state=config.RANDOM_SEED, train_size=8.0 / 9,
                           test_size=1.0 / 9, shuffle=True)
    train_data, train_label, eval_data, eval_label = \
        reshape_for_train_eval(train_data, train_label, eval_data, eval_label)
    return train_data, train_label, eval_data, eval_label, test_data, event_action_test, test_trials


def train_early_stopping(model, x_data_all_L, x_data_all_M, x_data_all_H, y_label_all_L, y_label_all_M, y_label_all_H,
                         event_action_all_L, event_action_all_M, event_action_all_H, y_trials_all_L, y_trials_all_M,
                         y_trials_all_H, state_train, state_test, train_index, test_index, count, optimizer,
                         criterion_action, params_path_toks, logger, lr_decay=None, early_stop=None):
    start_time = time.time()
    best_val_epoch = -1
    if early_stop is None:
        early_stop = Early_stop(config.ES_PATIENT)
        lr_decay = Adjust_lr(config.INIT_LEARNING_RATE, config.LR_PATIENT, optimizer)
        epoch_idx = 1
        iter_idx = 1
    else:
        epoch_idx = early_stop.get_current_epoch()
        iter_idx = early_stop.get_current_iter()

    print('*' * 40 + 'Begin to train' + '*' * 40)
    lr = config.INIT_LEARNING_RATE
    print('learning rate:{}'.format(lr))

    # build 3 * 3 = 9 mappings
    if state_train == 0 and state_test == 0:
        x_data_all_L, y_label_all_L, x_data_all_L, y_label_all_L, event_action_all_L = shape_for_kfold(
            x_data_all_L, y_label_all_L, x_data_all_L, y_label_all_L, event_action_all_L)
        train_data, train_label, eval_data, eval_label, test_data, event_action_test, test_trials = \
            generate_train_valid_test(x_data_all_L, y_label_all_L, x_data_all_L, y_label_all_L,
                                      event_action_all_L, y_trials_all_L, train_index, test_index)

    if state_train == 0 and state_test == 1:
        x_data_all_L, y_label_all_L, x_data_all_M, y_label_all_M, event_action_all_M = shape_for_kfold(
            x_data_all_L, y_label_all_L, x_data_all_M, y_label_all_M, event_action_all_M)
        train_data, train_label, eval_data, eval_label, test_data, event_action_test, test_trials = \
            generate_train_valid_test(x_data_all_L, y_label_all_L, x_data_all_M, y_label_all_M,
                                      event_action_all_M, y_trials_all_M, train_index, test_index)

    if state_train == 0 and state_test == 2:
        x_data_all_L, y_label_all_L, x_data_all_H, y_label_all_H, event_action_all_H = shape_for_kfold(
            x_data_all_L, y_label_all_L, x_data_all_H, y_label_all_H, event_action_all_H)
        train_data, train_label, eval_data, eval_label, test_data, event_action_test, test_trials = \
            generate_train_valid_test(x_data_all_L, y_label_all_L, x_data_all_H, y_label_all_H,
                                      event_action_all_H, y_trials_all_H, train_index, test_index)

    if state_train == 1 and state_test == 0:
        x_data_all_M, y_label_all_M, x_data_all_L, y_label_all_L, event_action_all_L = shape_for_kfold(
            x_data_all_M, y_label_all_M, x_data_all_L, y_label_all_L, event_action_all_L)
        train_data, train_label, eval_data, eval_label, test_data, event_action_test, test_trials = \
            generate_train_valid_test(x_data_all_M, y_label_all_M, x_data_all_L, y_label_all_L,
                                      event_action_all_L, y_trials_all_L, train_index, test_index)

    if state_train == 1 and state_test == 1:
        x_data_all_M, y_label_all_M, x_data_all_M, y_label_all_M, event_action_all_M = shape_for_kfold(
            x_data_all_M, y_label_all_M, x_data_all_M, y_label_all_M, event_action_all_M)
        train_data, train_label, eval_data, eval_label, test_data, event_action_test, test_trials = \
            generate_train_valid_test(x_data_all_M, y_label_all_M, x_data_all_M, y_label_all_M,
                                      event_action_all_M, y_trials_all_M, train_index, test_index)

    if state_train == 1 and state_test == 2:
        x_data_all_M, y_label_all_M, x_data_all_H, y_label_all_H, event_action_all_H = shape_for_kfold(
            x_data_all_M, y_label_all_M, x_data_all_H, y_label_all_H, event_action_all_H)
        train_data, train_label, eval_data, eval_label, test_data, event_action_test, test_trials = \
            generate_train_valid_test(x_data_all_M, y_label_all_M, x_data_all_H, y_label_all_H,
                                      event_action_all_H, y_trials_all_H, train_index, test_index)

    if state_train == 2 and state_test == 0:
        x_data_all_H, y_label_all_H, x_data_all_L, y_label_all_L, event_action_all_L = shape_for_kfold(
            x_data_all_H, y_label_all_H, x_data_all_L, y_label_all_L, event_action_all_L)
        train_data, train_label, eval_data, eval_label, test_data, event_action_test, test_trials = \
            generate_train_valid_test(x_data_all_H, y_label_all_H, x_data_all_L, y_label_all_L,
                                      event_action_all_L, y_trials_all_L, train_index, test_index)

    if state_train == 2 and state_test == 1:
        x_data_all_H, y_label_all_H, x_data_all_M, y_label_all_M, event_action_all_M = shape_for_kfold(
            x_data_all_H, y_label_all_H, x_data_all_M, y_label_all_M, event_action_all_M)
        train_data, train_label, eval_data, eval_label, test_data, event_action_test, test_trials = \
            generate_train_valid_test(x_data_all_H, y_label_all_H, x_data_all_M, y_label_all_M,
                                      event_action_all_M, y_trials_all_M, train_index, test_index)

    if state_train == 2 and state_test == 2:
        x_data_all_H, y_label_all_H, x_data_all_H, y_label_all_H, event_action_all_H = shape_for_kfold(
            x_data_all_H, y_label_all_H, x_data_all_H, y_label_all_H, event_action_all_H)
        train_data, train_label, eval_data, eval_label, test_data, event_action_test, test_trials = \
            generate_train_valid_test(x_data_all_H, y_label_all_H, x_data_all_H, y_label_all_H,
                                      event_action_all_H, y_trials_all_H, train_index, test_index)

    while True:
        train_data_gen = gen_minibatch_train(train_data, train_label, mini_batch_size=config.BATCH_SIZE)
        iter_idx, train_loss_epoch, train_acc = train_model(epoch_idx, optimizer, criterion_action, iter_idx, model,
                                                            train_data_gen, logger, start_time=start_time)
        # eval the model
        if epoch_idx >= 1 and epoch_idx % config.EVAL_EPOCH == 0:
            with torch.no_grad():
                valid_data_gen = gen_minibatch_eval(eval_data, eval_label, mini_batch_size=config.BATCH_SIZE_EVAL,
                                                    shuffle=False)
                loss_eval, acc_eval = valid_model(epoch_idx, criterion_action, model, valid_data_gen, logger,
                                                  start_time)
                print('this epoch %d train_loss:%f, train_acc:%f, valid_loss:%f, valid_acc:%f, ' %
                      (epoch_idx, train_loss_epoch, train_acc, loss_eval, acc_eval))

            is_stop = early_stop.step(loss_eval, acc_eval, epoch_idx, iter_idx)
            best_val_epoch = early_stop.get_best_epoch()
            best_acc_valid = early_stop.get_best_acc_valid()
            if is_stop:  # have overpassed the valid dataset 20epoch
                print('[Stop Iter]-Best Epoch %d, total time cost (%s): best valid loss: %f and best valid acc: %f; '
                      % (best_val_epoch, timeSince(start_time), early_stop.get_best_loss_valid(),
                         best_acc_valid))

                # save model's params
                if epoch_idx >= 1 and epoch_idx % config.SAVE_EPOCH == 0:
                    print("[Saving Trained Parameters] at epoch %d and best epoch is %d, best valid acc is %f" % (
                        epoch_idx, best_val_epoch, best_acc_valid))
                    model_dir = '%s%d-%d-%d%s' % (params_path_toks[0], state_train, state_test,
                                                  RANDOM_SEED, params_path_toks[1])
                    if config.USE_CUDA and len(config.GPUS) > 1:
                        torch.save(
                            {'state_dict': model.module.state_dict(),
                             'early_stop': early_stop,
                             'lr_decay': lr_decay,
                             }, model_dir)
                    else:
                        torch.save(
                            {'state_dict': model.state_dict(),
                             'early_stop': early_stop,
                             'lr_decay': lr_decay,
                             }, model_dir)
                break
        lr = lr_decay.step(train_loss_epoch)
        logger.log_lr(lr, epoch_idx)
        logger.flush()
        print('this epoch {} learning rate is {}'.format(epoch_idx, lr))
        # when the epoch equal to setting max epoch, end!
        if epoch_idx == config.MAX_EPOCH_NUM:
            print('[Stop Iter]-Best Epoch %d, total time cost (%s): best valid loss: %f and best valid acc: %f '
                  % (best_val_epoch, timeSince(start_time), early_stop.get_best_loss_valid(), best_acc_valid))
            # save model's params
            if epoch_idx >= 1 and epoch_idx % config.SAVE_EPOCH == 0:
                print("[Saving Trained Parameters] at epoch %d and best epoch is %d, best valid acc is %f" % (
                    epoch_idx, best_val_epoch, best_acc_valid))
                model_dir = '%s%d-%d-%d_%d%s' % (params_path_toks[0], state_train, state_test, count,
                                                 RANDOM_SEED, params_path_toks[1])
                if config.USE_CUDA and len(config.GPUS) > 1:
                    torch.save(
                        {'state_dict': model.module.state_dict(),
                         'early_stop': early_stop,
                         'lr_decay': lr_decay,
                         }, model_dir)
                else:
                    torch.save(
                        {'state_dict': model.state_dict(),
                         'early_stop': early_stop,
                         'lr_decay': lr_decay,
                         }, model_dir)
            break
        epoch_idx += 1

    for repetitions in range(1, 11):
        print('The number of used repetitions is %d' % repetitions)
        # for test dataset
        event_action_test_rep, test_data_rep = test_prepare(test_data, event_action_test, repetitions)
        test_data_gen = gen_minibatch_test(test_data_rep, event_action_test_rep, test_trials,
                                           mini_batch_size=config.BATCH_SIZE_TEST, shuffle=False)
        checkpoint = torch.load(model_dir)
        if config.USE_CUDA and len(config.GPUS) > 1:
            model.module.load_state_dict(checkpoint['state_dict'])
            early_stop = checkpoint['early_stop']
        else:
            model.load_state_dict(checkpoint['state_dict'])
            early_stop = checkpoint['early_stop']
        epoch_idx = early_stop.get_current_epoch()
        loss_test, acc_test = test_model(criterion_action, model, repetitions, test_data_gen, logger, start_time)
        print("The BEST test acc in %d train state, %d test state, %d repetitions used, is %f" %
              (state_train, state_test, repetitions, acc_test))


def test_prepare(test_data, event_action_test, repetitions):
    event_action_test = event_action_test[:, 0:repetitions, :]
    test_data = test_data[:, 0:repetitions, :, :, :]
    return event_action_test, test_data


def main():
    print("*" * 80)
    print('Build Model')
    print("*" * 80)

    if config.IS_TRAIN:
        print('Current is TRAIN phase')
        print('Reading eeg singal and Generating erp clips from folders ... ')
        if not config.LOAD_DATA_PICKLE:
            x_data_all_L, x_data_all_M, x_data_all_H, y_label_all_L, y_label_all_M, y_label_all_H, event_action_all_L, event_action_all_M, event_action_all_H, y_trials_all_L, y_trials_all_M, y_trials_all_H = get_data_from_files(
                config.DATA_FOLDER)
            print('')
            print("Please wait for a while patiently, the data is being processed, don't do anything now!")
            if config.SAVE_DATA_PICKLE:
                data_pkl_f = open(config.DATA_PICKLE_FILE, 'wb')
                pickle.dump((x_data_all_L, x_data_all_M, x_data_all_H, y_label_all_L, y_label_all_M, y_label_all_H,
                             event_action_all_L, event_action_all_M, event_action_all_H, y_trials_all_L, y_trials_all_M,
                             y_trials_all_H), data_pkl_f,
                            protocol=4)
                data_pkl_f.close()
        else:
            print('Direct loading erp clips from pickle file')
            print("Please wait for a while patiently, the data is being processed, don't do anything now!")
            data_pkl_f = open(config.DATA_PICKLE_FILE, 'rb')
            x_data_all_L, x_data_all_M, x_data_all_H, y_label_all_L, y_label_all_M, y_label_all_H, event_action_all_L, event_action_all_M, event_action_all_H, y_trials_all_L, y_trials_all_M, y_trials_all_H = pickle.load(
                data_pkl_f)
            data_pkl_f.close()

        for state_train in range(0, 3):  #
            for state_test in range(0, 3):  #
                print('The state_train is %d, The state_test is %d' % (state_train, state_test))
                print("This is the cross-subject task.")

                # Tensorboad's setting
                if os.path.exists(config.TENSORBOARD_LOG_DIR) and (not config.MODEL_SAVE_PATH):
                    shutil.rmtree(config.TENSORBOARD_LOG_DIR)
                    os.mkdir(config.TENSORBOARD_LOG_DIR)
                # logger = Logger(config.TENSORBOARD_LOG_DIR)

                kf = KFold(n_splits=10)
                sum_acc = 0
                count = 0
                for train_index, test_index in kf.split(y_trials_all_L):
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    count += 1
                    best_acc = 0
                    if count in range(1, 11):  # No.count
                        model = ActionPredBasedEEG(win_for_eeg_sig=config.WIN_FOR_EEG_SIGNAL,
                                                   eeg_enc_dim=config.EEG_ENCODER_DIM
                                                   , eeg_fea_dim=config.EEG_FEATURE_DIM,
                                                   rnn_hid_dim=config.RNN_HIDDEN_DIM
                                                   , rnn_layer_num=config.RNN_LAYER_NUM)

                        params_path_toks = config.MODEL_SAVE_PATH.split(',')
                        # load params, load pretrained model
                        early_stop = None
                        lr_decay = None
                        if config.RELOAD_PRETRAIN_MODEL and config.MODEL_SAVE_PATH is not None:
                            params_path = '%s%05d%s' % (
                            params_path_toks[0], config.RELOAD_EVAL_EPOCH, params_path_toks[1])
                            model.load_state_dict(torch.load(params_path)['state_dict'])
                            early_stop = torch.load(params_path)['early_stop']
                            lr_decay = torch.load(params_path)['lr_decay']
                            epoch_idx = early_stop.get_current_epoch()
                            iter_idx = early_stop.get_current_iter()
                            assert epoch_idx == config.RELOAD_EVAL_EPOCH
                            print('[ReLoading Models successfully with epoch %d, iter %d and model path:%s.]\n'
                                  % (epoch_idx, iter_idx, params_path))

                        if config.USE_CUDA:
                            print('The model is running on GPU', str(config.GPUS))
                            # torch.cuda.set_device(config.GPUS[0])
                            model.cuda()
                            if len(config.GPUS) > 1:
                                model = nn.DataParallel(model, device_ids=config.GPUS, dim=0)
                        else:
                            print('The model is running on CPU')

                        # compute total parameters in model
                        param_count = 0
                        for param in model.parameters():
                            param_count += param.view(-1).size()[0]
                        print('Params size in this model is {}'.format(param_count))

                        optimizer = torch.optim.Adam(model.parameters(), lr=config.INIT_LEARNING_RATE)
                        criterion_action = nn.NLLLoss()

                        EVENTSTAMP = "cross_%d-%d_%d" % (state_train, state_test, count)
                        logdir = config.TENSORBOARD_LOG_DIR + EVENTSTAMP
                        if not os.path.exists(logdir):
                            os.mkdir(logdir)
                        logger = Logger(logdir)
                        train_early_stopping(model, x_data_all_L, x_data_all_M, x_data_all_H,
                                             y_label_all_L,
                                             y_label_all_M, y_label_all_H, event_action_all_L,
                                             event_action_all_M,
                                             event_action_all_H, y_trials_all_L, y_trials_all_M,
                                             y_trials_all_H, state_train, state_test,
                                             train_index, test_index, count, optimizer,
                                             criterion_action,
                                             params_path_toks, logger, lr_decay=lr_decay,
                                             early_stop=early_stop)
                        print('This is the %d people' % count)
                    print('**********************************************************')
    else:
        print('Current is INFERENCE phase')
    logger.close()

if __name__ == "__main__":
    RANDOM_SEED = config.RANDOM_SEED
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print('Start, and the random seed is %d' % config.RANDOM_SEED)
    if config.USE_CUDA:
        # assure the certainty and repeatability of the results
        torch.backends.cudnn.deterministic = True
    main()
