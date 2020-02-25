# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class ROSMIModel(nn.Module):
    def __init__(self, num_bearings):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        print(hid_dim)

        # ROSMI Pred heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            # GeLU(),
            # BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 4)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

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

        logit = self.logit_fc(x)

        return logit