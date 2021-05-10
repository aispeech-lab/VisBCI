import random
import torch
from tensorboardX import SummaryWriter
import numpy as np


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_training(self, loss, acc, iteration):
        self.add_scalar("training.loss", loss, iteration)
        self.add_scalar("training.acc", acc, iteration)

    def log_validation(self, loss, acc, epoch):
        self.add_scalar("validation.loss", loss, epoch)
        self.add_scalar("validation.acc", acc, epoch)

    def log_test(self, loss, acc, repetitions):
        self.add_scalar("test.loss", loss, repetitions)
        self.add_scalar("test.acc", acc, repetitions)

    def log_best_acc_average(self, acc, repetitions):
        self.add_scalar("Average validation.acc", acc, repetitions)

    def log_lr(self, lr, iteration):
        self.add_scalar("model.lr", lr, iteration)

    def writer_flush(self):
        self.flush()
