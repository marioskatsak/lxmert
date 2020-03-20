# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch
from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 25


class ROSMIModel(nn.Module):
    def __init__(self, num_bearings):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        self.hid_dim = self.lxrt_encoder.dim
        print(self.hid_dim)
        self.distance_fc = nn.Sequential(
            # nn.Linear(self.hid_dim, self.hid_dim*2),
            # GeLU(),
            GeLU(),
            BertLayerNorm(self.hid_dim*2, eps=1e-12),
            nn.Linear(self.hid_dim*2, 1)
        )
        self.bearing_fc = nn.Sequential(
            # nn.Linear(self.hid_dim, self.hid_dim*2),
            # GeLU(),
            GeLU(),
            BertLayerNorm(self.hid_dim*2, eps=1e-12),
            nn.Linear(self.hid_dim*2, 1)
        )
        self.land_fc = nn.Sequential(
            # nn.Linear(self.hid_dim, self.hid_dim*2),
            # GeLU(),
            GeLU(),
            BertLayerNorm(self.hid_dim*2, eps=1e-12),
            nn.Linear(self.hid_dim*2, 4)
        )
        # ROSMI Pred heads
        self.logit_fc = nn.Sequential(
            # nn.Linear(68 * self.hid_dim* 3, self.hid_dim),
            nn.Linear(self.hid_dim*2, self.hid_dim*4),
            GeLU(),
            BertLayerNorm(self.hid_dim*4, eps=1e-12),
            nn.Linear(self.hid_dim*4, 4)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.land_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.bearing_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.distance_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, feat_mask, pos, names, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param names:  (b, o, max_seq_length)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        if args.n_ent:
            x = self.lxrt_encoder(sent, (feat, pos, names),visual_attention_mask = feat_mask)
        else:
            x = self.lxrt_encoder(sent, (feat, pos, names))
        # print(x)
        # print((x.shape))
        # input(torch.mean(x))
        # x = x.view(-1, 68 * self.hid_dim* 3)
        # print(x.shape)
        logit = self.logit_fc(x)
        dist_ = self.distance_fc(x)
        landmark_ = self.land_fc(x)
        bearing_ = self.bearing_fc(x)
        return logit, (dist_, landmark_, bearing_)
