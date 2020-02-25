# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_det_obj_tsv, calc_iou_individual
from lxrt.entry import convert_sents_to_features

from lxrt.tokenization import BertTokenizer


# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000
# Max length including <bos> and <eos>
MAX_SENT_LENGTH = 20
# The path to data and image features.
# VQA_DATA_ROOT = '/scratch/mmk11/data/vqa/'
# IMGFEAT_ROOT = '/scratch/mmk11/data/rosmi/'
SPLIT2NAME = {
    'train': 'train',
    'valid': 'val',
    'test': 'test',
}
# SPLIT2NAME = {
#     'train': 'train2014',
#     'valid': 'val2014',
#     'minival': 'val2014',
#     'nominival': 'val2014',
#     'test': 'test2015',
# }

class ROSMIDataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }

    ROSMI data example in json file

        {
      "img_id": "3G5F9DBFOS5RDFXHAP1AIEBZCHJVHO_5",
      "image_filename": "3G5F9DBFOS5RDFXHAP1AIEBZCHJVHO_5.png",
      "scenario_items": "scenario3.json" <--- contains all items of the map
      "landmarks": [
        {
          "name": "husky17",
          "distance": "118",
          "bearing": "0",
          "confidence": "2",
          "raw_gps": [],
          "id": "3G5F9DBFOS5RDFXHAP1AIEBZCHJVHO_5_husky17",
          "keywords": "husky robot",
          "g_type": "Point",
          "landmark_gps": [],
          "human_gps": [],
          "landmark_pixels": [ ],
          "human_pixels": [],
          "raw_pixels": []
        }
      ],
      "dynamo_obj": [],
      "gold_coordinates": [],
      "sentid": 279,
      "sentence": {
        "raw": "send husky17 118m in north",
        "imgid": "3G5F9DBFOS5RDFXHAP1AIEBZCHJVHO_5",
        "tokens": [    ]
      },
      "gold_pixels": [ ]
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(os.path.join(args.data_path,"%s.json" % SPLIT2NAME[split]))))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['sentid']: datum
            for datum in self.data
        }

        # Answers
        self.bearing2label = json.load(open(os.path.join(args.data_path,"trainval_bearing2label.json")))
        self.label2bearing = json.load(open(os.path.join(args.data_path,"trainval_label2bearing.json")))
        self.convert2bearing = json.load(open(os.path.join(args.data_path,"convert_bearing_values.json")))
        assert len(self.bearing2label) == len(self.label2bearing)

    @property
    def num_bearings(self):
        return len(self.bearing2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class ROSMITorchDataset(Dataset):
    def __init__(self, dataset: ROSMIDataset):
        super().__init__()
        self.raw_dataset = dataset
        self.max_seq_length = MAX_SENT_LENGTH

        if args.n_ent:
            self.named_entities = True
        else:
            self.named_entities = False
        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        IMGFEAT_ROOT = args.data_path
        # Loading detection features to img_data
        img_data = []
        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_det_obj_tsv(
                os.path.join(IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        sent_id = datum['sentid']
        sent = datum['sentence']['raw']
        target = torch.tensor(datum['gold_pixels'])

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        names = img_info['names'].copy()


        feat_mask = 0



        if self.named_entities:


            feats = torch.from_numpy(feats)
            boxes = torch.from_numpy(boxes)
            # if names:
            names_ids = []
            names_segment_ids = []
            names_mask = []
            for obj in names:
                # for obj in img:
                # input(f"objects to tokenize: {obj}")
                names_features = convert_sents_to_features(
                    obj, self.max_seq_length, self.tokenizer)

                # input(names_features[0].input_ids)
                # for f in names_features
                names_ids.append(torch.tensor(names_features[0].input_ids, dtype=torch.long))
                names_segment_ids.append(torch.tensor(names_features[0].segment_ids, dtype=torch.long))
                names_mask.append(torch.tensor(names_features[0].input_mask, dtype=torch.long))

            # names_ids = torch.stack(names_ids)
            # input(len(names_ids))

            # sentence = convert_sents_to_features(
            #     [sent], self.max_seq_length, self.tokenizer)
            #
            # names_ids.append(torch.tensor(sentence[0].input_ids, dtype=torch.long))
            # names_segment_ids.append(torch.tensor(sentence[0].segment_ids, dtype=torch.long))
            # names_mask.append(torch.tensor(sentence[0].input_mask, dtype=torch.long))

            if (100 - len(names_ids)) > 0:
                # Zero-pad up to the sequence length.
                padding = (100 - len(names_ids))*[torch.zeros(self.max_seq_length, dtype=torch.long)]

                feats_vis_padding = torch.zeros(((100 - feats.shape[0]),feats.shape[1]), dtype=torch.double)
                box_vis_padding = torch.zeros(((100 - boxes.shape[0]),boxes.shape[1]), dtype=torch.double)
                feats = torch.cat((feats,feats_vis_padding))
                boxes = torch.cat((boxes,box_vis_padding))

                names_ids = torch.stack(names_segment_ids + padding)
                names_segment_ids = torch.stack(names_segment_ids + padding)
                names_mask = torch.stack(names_mask + padding)

                    # bert hidden_size = 768
                feat_mask = torch.ones(feats.shape[0], dtype=torch.double)
                feats_padding = torch.zeros((100 - feats.shape[0]), dtype=torch.double)
                feat_mask = torch.cat((feat_mask,feats_padding))
            else:

                # feats_vis_padding = torch.zeros(((100 - feats.shape[0]),feats.shape[1]), dtype=torch.double)
                # box_vis_padding = torch.zeros(((100 - boxes.shape[0]),boxes.shape[1]), dtype=torch.double)
                # feats = torch.cat((feats,feats_vis_padding))
                # boxes = torch.cat((boxes,box_vis_padding))
                names_ids = torch.stack(names_segment_ids)
                names_segment_ids = torch.stack(names_segment_ids)
                names_mask = torch.stack(names_mask)

            _names = (names_ids, names_segment_ids, names_mask)
        else:
            _names = 0

            assert obj_num == len(boxes) == len(feats)

        # print(feats.shape)
        # print(boxes.shape)
        # print(names_ids.shape)
        # print(names_segment_ids.shape)
        # print(feat_mask.shape)
        # input(names_mask.shape)


        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        return sent_id, feats, feat_mask, boxes, _names, sent, target#bearing
            # else:
            #     return ques_id, feats, boxes, ques


class ROSMIEvaluator:
    def __init__(self, dataset: ROSMIDataset):
        self.dataset = dataset

    def evaluate(self, sentid2ans: dict):
        score = 0.
        for sentid, pred_box in sentid2ans.items():
            datum = self.dataset.id2datum[sentid]
            print(pred_box,datum['gold_pixels'])
            iou = calc_iou_individual(pred_box, datum['gold_pixels'])
            if iou > 0.4:
            # if ans in label:
                score += 1
        return score / len(sentid2ans)

    def dump_result(self, sentid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param sentid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for sent_id, ans in sentid2ans.items():
                result.append({
                    'sentence_id': sent_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)
