# -*- coding: utf-8 -*-
# Created on 2021/04
# Author: NZY & XJM

"""
build the 1dCNN+BiLSTM model
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, win=16, fea_dim=256, in_dim=28):
        super(Encoder, self).__init__()
        self.win, self.fea_dim, self.in_dim = win, fea_dim, in_dim
        # 50% overlap
        self.conv1d_U = nn.Conv1d(self.in_dim, self.fea_dim, kernel_size=self.win, stride=self.win // 2, bias=False)

    def forward(self, in_samples):
        """
        Args:
            in_samples: [B, C, T], B is batch size, C is channels, T is #samples
        Returns:
            fea: [B, N, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
        """
        out_fea = F.relu(self.conv1d_U(in_samples))  # -> [B, N, L]
        return out_fea


class ActionPredBasedEEG(nn.Module):
    def __init__(self, win_for_eeg_sig, eeg_enc_dim, eeg_fea_dim, rnn_hid_dim, rnn_layer_num, dropout=0.5):
        super(ActionPredBasedEEG, self).__init__()
        self.win_for_eeg_sig = win_for_eeg_sig  # 16
        self.eeg_enc_dim = eeg_enc_dim  # 256
        self.eeg_fea_dim = eeg_fea_dim  # 64
        self.rnn_hid_dim = rnn_hid_dim  # 128
        self.dropout = dropout
        self.num_layers = rnn_layer_num
        self.bidirectional = True

        self.erp_encoder = Encoder(self.win_for_eeg_sig, self.eeg_enc_dim, in_dim=28)
        self.erp_enc_LayerNorm = nn.GroupNorm(1, self.eeg_enc_dim, eps=1e-8)  # [B E L]-->[B E L]
        # bottleneck
        self.bottle_neck = nn.Conv1d(self.eeg_enc_dim, self.eeg_fea_dim, 1, bias=False)
        # encoding erp
        self.erp_local_rnn = nn.LSTM(self.eeg_fea_dim, self.rnn_hid_dim, self.num_layers, dropout=dropout
                                     , batch_first=True, bidirectional=self.bidirectional)
        self.erp_global_rnn = nn.LSTM(self.rnn_hid_dim*(int(self.bidirectional)+1), self.rnn_hid_dim, self.num_layers
                                      , dropout=dropout, batch_first=True, bidirectional=self.bidirectional)
        self.pred_proj = nn.Linear(self.rnn_hid_dim*(int(self.bidirectional)+1), 1)

    def forward(self, eeg_erp_in):
        """
        :param eeg_erp_in: shape (batch, reps, erp_clips:12, channels:28, seq_len:600 ms / erp)
        """
        self.erp_local_rnn.flatten_parameters()
        self.erp_global_rnn.flatten_parameters()

        batch_size, erp_clips, eeg_channels, samples_len = eeg_erp_in.shape

        # (batch, erp_clips, channels,seq_len)->(batch * erp_clips, channels, seq_len)
        eeg_erp_in = eeg_erp_in.reshape(-1, eeg_channels, samples_len)

        # (batch * erp_clips, channels, seq_len) -> (batch * erp_clips, fea_dim, seq_len)
        eeg_erp_fea = self.erp_encoder(eeg_erp_in)
        eeg_erp_fea = self.erp_enc_LayerNorm(eeg_erp_fea)
        eeg_erp_fea = self.bottle_neck(eeg_erp_fea)

        # (batch * erp_clips, fea_dim, seq_len) -> (batch * erp_clips, seq_len, fea_dim)
        eeg_erp_local_fea, hidden_states = self.erp_local_rnn(eeg_erp_fea.transpose(1, 2))
        # (batch * erp_clips, seq_len, fea_dim) -> (batch * erp_clips, fea_dim) through pooling to aggregate 600ms
        eeg_erp_local_fea = torch.mean(eeg_erp_local_fea, dim=1)  # mean pooling
        # (batch * erp_clips, fea_dim) -> (batch, erp_clips, fea_dim)
        eeg_erp_local_fea = eeg_erp_local_fea.reshape(batch_size, erp_clips, -1)
        eeg_erp_global_fea, hidden_states = self.erp_global_rnn(eeg_erp_local_fea)
        # (batch, erp_clips, fea_dim) -> (batch, erp_clips, 1)
        eeg_erp_pred = self.pred_proj(eeg_erp_global_fea)
        # (batch, erp_clips, 1) -> (batch, erp_clips)
        action_pred = F.log_softmax(eeg_erp_pred.squeeze(2), dim=1)
        return action_pred


