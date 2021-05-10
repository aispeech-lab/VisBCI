# coding:utf-8


class Adjust_lr(object):
    def __init__(self, init_lr, lr_patient, optimizer):
        self.best_loss = 1e5
        self.now_lr = init_lr
        self.lr_patient = lr_patient
        self.count = 0
        self.optimizer = optimizer

    def step(self, loss):
        # if loss < self.best_loss:
        #     self.best_loss = loss
        #     self.count = 0
        # else:
        #     self.count += 1
        self.count += 1
        if self.count == self.lr_patient:
            lr = self.now_lr * 0.99
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.now_lr = lr
            self.count = 0  
            self.best_loss = loss
        return self.now_lr


class Early_stop(object):
    def __init__(self, es_patient):
        self.best_loss_valid = -1
        self.best_loss_test = -1
        self.best_acc_valid = 0.0000001
        self.best_acc_test = 0
        self.best_epoch = -1
        self.current_epoch = -1
        self.current_iter = -1
        self.es_patient = es_patient
        self.count = 0

    def step(self, loss, acc, epoch_idx, iter_idx):
        self.current_epoch = epoch_idx
        self.current_iter = iter_idx
        if acc > self.best_acc_valid:
            self.best_loss_valid = loss
            self.best_acc_valid = acc
            self.best_epoch = epoch_idx
            self.count = 0
        else:
            self.count += 1
        if self.count == self.es_patient:
            # early stop
            return True
        else:
            # continue
            return False

    def is_best_val_loss(self):
        if self.count == 0:
            return True
        else:
            return False

    def get_best_loss_valid(self):
        return self.best_loss_valid

    def get_best_acc_valid(self):
        return self.best_acc_valid

    def get_best_epoch(self):
        return self.best_epoch

    def get_current_epoch(self):
        return self.current_epoch

    def get_current_iter(self):
        return self.current_iter
