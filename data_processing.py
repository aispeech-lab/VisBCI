# -*- coding: utf-8 -*-
# Created on 2021/04
# Author: NZY & XJM

"""
preprocess the data -  read and label the data
"""

import scipy.io as scio
import numpy as np
import os
import pickle

# 28 channels
channel_choose = np.array([2, 10, 19, 28, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 56, 57, 59, 61, 62, 63]) 
# shift for python (idx from 0), matlab' ids is from 1.
channel_choose -= 1 


def get_data_from_files(data_folder):
    et_file_name = 'event_eventtime.mat'
    signal_file_name = 'Signal.mat'

    x_data_all_L = np.zeros((0, 12, 28, 600))  # (sample_size(reps), 12 erp, 28 channels, 600 ms/erp)
    x_data_all_M = np.zeros((0, 12, 28, 600))
    x_data_all_H = np.zeros((0, 12, 28, 600))
    y_label_all_L = np.zeros((0, 1))     # (sample_size(reps), 1)
    y_label_all_M = np.zeros((0, 1))
    y_label_all_H = np.zeros((0, 1))
    event_action_all_L = np.zeros((0, 12))
    event_action_all_M = np.zeros((0, 12))
    event_action_all_H = np.zeros((0, 12))
    y_trials_all_L = np.zeros((0, 1))
    y_trials_all_M = np.zeros((0, 1))
    y_trials_all_H = np.zeros((0, 1))

    # the structure: ./EEG_ERP/H/subject_001_20190525_WYW/01/event_eventtime.mat
    if os.path.exists(data_folder):
        brain_state_folders = os.listdir(data_folder)
        print('###: There are %d brain states in data folder: %s' % (len(brain_state_folders), data_folder))
        for brain_state in brain_state_folders:
            brain_state_root = os.path.join(data_folder, brain_state)  
            subject_folders = os.listdir(brain_state_root)
            print('###: There are %d subjects in brain state folder: %s' % (len(subject_folders), brain_state_root))
            for subject in subject_folders:
                subject_root = os.path.join(brain_state_root, subject)  
                task_folders = os.listdir(subject_root)
                print('###: %s takes %d tasks in the subject folder: %s' % (subject, len(task_folders), subject_root))
                for task in task_folders:
                    task_root = os.path.join(subject_root, task)  
                    print('###: Here is task: %s' % task_root)

                    x_data_L = np.zeros((12 * 10, 12, 28, 600))
                    x_data_M = np.zeros((12 * 10, 12, 28, 600))
                    x_data_H = np.zeros((12 * 10, 12, 28, 600))
                    y_label_L = np.zeros((12 * 10, 1))
                    y_label_M = np.zeros((12 * 10, 1))
                    y_label_H = np.zeros((12 * 10, 1))
                    event_action_L = np.zeros((12 * 10, 12))
                    event_action_M = np.zeros((12 * 10, 12))
                    event_action_H = np.zeros((12 * 10, 12))
                    y_trials_L = np.zeros((12, 1))
                    y_trials_M = np.zeros((12, 1))
                    y_trials_H = np.zeros((12, 1))

                    et_file_path = os.path.join(task_root, et_file_name)
                    event_eventtime = scio.loadmat(et_file_path)
                    event_timestamp = event_eventtime['event_eventtime'][1]
                    event_action_id = event_eventtime['event_eventtime'][0]

                    signal_file_path = os.path.join(task_root, signal_file_name)
                    signal_eeg = scio.loadmat(signal_file_path)
                    signal_eeg = signal_eeg['Signal'][channel_choose, :]

                    for action_target in range(1, 13, 1):
                        if brain_state == 'L':
                            y_trials_L[action_target - 1] = action_target - 1
                        if brain_state == 'M':
                            y_trials_M[action_target - 1] = action_target - 1
                        if brain_state == 'H':
                            y_trials_H[action_target - 1] = action_target - 1
                        for repetition in range(10):  # repetitions
                            sample_idx = (action_target-1)*10 + repetition
                            cur_idx = sample_idx*12
                            for erp_idx in range(12):
                                event_timestamp_start = int(event_timestamp[cur_idx+erp_idx] - 1)
                                if brain_state == 'L':
                                    x_data_L[sample_idx, erp_idx, :, :] = signal_eeg[:, event_timestamp_start:(
                                                event_timestamp_start + 600)]
                                    event_action_L[sample_idx, erp_idx] = event_action_id[
                                        cur_idx + erp_idx]
                                    if event_action_id[cur_idx + erp_idx] == action_target:
                                        y_label_L[sample_idx] = erp_idx
                                if brain_state == 'M':
                                    x_data_M[sample_idx, erp_idx, :, :] = signal_eeg[:, event_timestamp_start:(
                                                event_timestamp_start + 600)]
                                    event_action_M[sample_idx, erp_idx] = event_action_id[
                                        cur_idx + erp_idx]
                                    if event_action_id[cur_idx + erp_idx] == action_target:
                                        y_label_M[sample_idx] = erp_idx
                                if brain_state == 'H':
                                    x_data_H[sample_idx, erp_idx, :, :] = signal_eeg[:, event_timestamp_start:(
                                                event_timestamp_start + 600)]
                                    event_action_H[sample_idx, erp_idx] = event_action_id[
                                        cur_idx + erp_idx]  # stimulus
                                    if event_action_id[cur_idx + erp_idx] == action_target:
                                        y_label_H[sample_idx] = erp_idx

                    if not np.all(x_data_L == 0):
                        x_data_all_L = np.vstack((x_data_all_L, x_data_L))
                    if not np.all(x_data_M == 0):
                        x_data_all_M = np.vstack((x_data_all_M, x_data_M))
                    if not np.all(x_data_H == 0):
                        x_data_all_H = np.vstack((x_data_all_H, x_data_H))
                    if not np.all(y_label_L == 0):
                        y_label_all_L = np.vstack((y_label_all_L, y_label_L))
                    if not np.all(y_label_M == 0):
                        y_label_all_M = np.vstack((y_label_all_M, y_label_M))
                    if not np.all(y_label_H == 0):
                        y_label_all_H = np.vstack((y_label_all_H, y_label_H))
                    if not np.all(event_action_L == 0):
                        event_action_all_L = np.vstack((event_action_all_L, event_action_L))
                    if not np.all(event_action_M == 0):
                        event_action_all_M = np.vstack((event_action_all_M, event_action_M))
                    if not np.all(event_action_H == 0):
                        event_action_all_H = np.vstack((event_action_all_H, event_action_H))
                    if not np.all(y_trials_L == 0):
                        y_trials_all_L = np.vstack((y_trials_all_L, y_trials_L))
                    if not np.all(y_trials_M == 0):
                        y_trials_all_M = np.vstack((y_trials_all_M, y_trials_M))
                    if not np.all(y_trials_H == 0):
                        y_trials_all_H = np.vstack((y_trials_all_H, y_trials_H))
    else:
        print('The datafolder path is empty: %s' % data_folder)
    print('Total number of the samples is %d' % (len(y_label_all_L)+len(y_label_all_M)+len(y_label_all_H)))
    return x_data_all_L, x_data_all_M, x_data_all_H, y_label_all_L, y_label_all_M, y_label_all_H,  event_action_all_L, event_action_all_M, event_action_all_H, y_trials_all_L, y_trials_all_M, y_trials_all_H           
