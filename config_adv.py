# coding:utf-8

# Created on 2021/04
# Author: NZY & XJM

import torch

DEBUG = False

DATASET = 'EEG_ERP'

TENSORBOARD_LOG_DIR = "./tmp_logging/tensorboard/"  # path of saving tensorboard data

SAVE_DATA_PICKLE = False  # save the data into pickle
LOAD_DATA_PICKLE = True  # load the data from pickle

# path of saving or loading pickle data
DATA_PICKLE_FILE = './tmp_logging/DATA_%s.pkl' % DATASET
GPUS = [0, 1, 2, 3]# , 4, 5, 6, 7]
USE_CUDA = torch.cuda.is_available() and len(GPUS) > 0
RANDOM_SEED = 0  # the random seed
RELOAD_PRETRAIN_MODEL = False  # reload pretrain model

# path of saving model params
PARAMS_PATH = './tmp_logging/'
# path of pretrained model
MODEL_SAVE_PATH = './tmp_logging/model_20210211_,.pt'

ADD_ADVERSARIAL_ATTACK = True
ATTACK_EPS = 0.01
ATTACK_ALPHA = 3 * ATTACK_EPS

IS_TRAIN = True

EEG_SAMPLE_RATE = 1000
if DEBUG:
    DATA_FOLDER = "./../EEG_ERP/"
    MAX_EPOCH_NUM = 20
    BATCH_SIZE = 4
    BATCH_SIZE_EVAL = 2
    BATCH_SIZE_TEST = 2
    WIN_FOR_EEG_SIGNAL = 16
    EEG_ENCODER_DIM = 32  # 2  # 16
    EEG_FEATURE_DIM = 16  # 2  # 8
    RNN_HIDDEN_DIM = 32  # 2   # 20
    RNN_LAYER_NUM = 1
    LOCAL_RNN_STEP = int((600/1000)*EEG_SAMPLE_RATE // (WIN_FOR_EEG_SIGNAL // 2))
    GLOBAL_RNN_STEP = 12
    EARLY_STOP_EPOCH_GAP = 10
    RELOAD_EVAL_EPOCH = 20  # 821
else:
    DATA_FOLDER = "/home/Brain_Machine_Interface/EEG_ERP/"
    MAX_EPOCH_NUM = 150
    BATCH_SIZE = 16
    BATCH_SIZE_EVAL = 16  # 设置一个合适的评估batch_size，使得测试数据不被丢弃
    BATCH_SIZE_TEST = 4
    WIN_FOR_EEG_SIGNAL = 32
    EEG_ENCODER_DIM = 256
    EEG_FEATURE_DIM = 64
    RNN_HIDDEN_DIM = 128  # 50
    RNN_LAYER_NUM = 3
    LOCAL_RNN_STEP = int((600/1000)*EEG_SAMPLE_RATE // (WIN_FOR_EEG_SIGNAL // 2))
    GLOBAL_RNN_STEP = 12
    EARLY_STOP_EPOCH_GAP = 100  # the epoch step of reload pretrain model
    RELOAD_EVAL_EPOCH = 821

INIT_LEARNING_RATE = 0.001
EVAL_EPOCH = 1  # num of epochs to eval
SAVE_EPOCH = 1  # num of epochs to save model
LR_PATIENT = 2  # patient for lr decay
ES_PATIENT = 20   # patient for early stop
