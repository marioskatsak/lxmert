# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import *
from lxrt.entry import convert_sents_to_features

from lxrt.tokenization import BertTokenizer


SCALES = {
            '0':25,
            '1':25,
            '2':4,
            '3':12,
            '4':4,
            '5':4,
            '6':4
        }

ZOOMS = {
            0:18,
            1:18,
            2:15,
            3:17,
            4:15,
            5:15,
            6:15
        }
GOLD_SIZES = {
            0:25,
            1:25,
            2:3,
            3:12,
            4:3,
            5:3,
            6:3
        }

BEAR2NUMS = {
    "None": -1,
    "North": 0,
    "South": 180,
    "West": 270,
    "East": 90,
    "North West": 315,
    "North East": 45 ,
    "South West": 225,
    "South East": 135

}
#  centers in lat , lon
CENTRES = {
            0:[37.73755663692416, -122.19795016945281],
            1:[32.58577585559755, -117.09164085240766],
            2:[32.61748188924153, -117.14119088106783],
            3:[32.60760476678458, -117.08442647549721],
            4:[37.694753719037756, -122.19294177307802],
            5:[37.71336706451458, -122.19060472858666],
            6:[32.59795016014067, -117.11036626803674]
        }
# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000
# Max length including <bos> and <eos>
MAX_SENT_LENGTH = 25
MAX_BOXES = 68
# The path to data and image features.
# VQA_DATA_ROOT = '/scratch/mmk11/data/vqa/'
# IMGFEAT_ROOT = '/scratch/mmk11/data/rosmi/'
SPLIT2NAME = {
    'train': 'easy_train',
    'valid': 'easy_val',
    'test': 'test',
    'mini_train': 'mini_train',
    'mini_valid': 'mini_val',
    'mini_test': 'mini_test',
    '0_easy_train':'0_easy_train',
    '0_easy_val':'0_easy_val',
    '1_easy_train':'1_easy_train',
    '1_easy_val':'1_easy_val',
    '2_easy_train':'2_easy_train',
    '2_easy_val':'2_easy_val',
    '3_easy_train':'3_easy_train',
    '3_easy_val':'3_easy_val',
    '4_easy_train':'4_easy_train',
    '4_easy_val':'4_easy_val',
    '5_easy_train':'5_easy_train',
    '5_easy_val':'5_easy_val',
    '6_easy_train':'6_easy_train',
    '6_easy_val':'6_easy_val',
    '7_easy_train':'7_easy_train',
    '7_easy_val':'7_easy_val',

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
        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
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
            # img_data.extend(load_obj_tsv(
            #     os.path.join(IMGFEAT_ROOT, 'train_obj36.tsv'),
            #     topk=load_topk))

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

        # with open('val_vocab.json') as training:
        #     train_dict = json.load(training)

        img_id = datum['img_id']
        sent_id = datum['sentid']
        sent = datum['sentence']['raw']
        # dist = torch.tensor([int(datum['landmarks'][0]['distance'])])
        # bearing = torch.tensor([int(datum['landmarks'][0]['bearing'])])
        if datum['landmarks'][0]['g_type'] == 'Point':
            landmark = torch.tensor(datum['landmarks'][0]['raw_pixels'])
        else:
            landmark = torch.tensor(datum['landmarks'][0]['landmark_pixels'])

        # print(datum['landmarks'][0]['raw_pixels'])
        # print(torch.tensor(datum['landmarks'][0]['raw_pixels']))
        # input(landmark)
        target = torch.tensor(datum['gold_pixels'])



        bearing = torch.zeros(self.raw_dataset.num_bearings)
        bearing[self.raw_dataset.bearing2label[self.raw_dataset.convert2bearing[datum['landmarks'][0]['bearing']]]] = 1

        # start and end id of distance
        tokens = ["[CLS]"] + self.tokenizer.tokenize(sent.strip()) + ["[SEP]"]
        # print(tokens)
        # for tks in tokens:
        #     if tks in train_dict.keys():
        #         train_dict[tks] += 1
        #     else:
        #         train_dict[tks] = 1
        #
        # with open('val_vocab.json', 'w') as trv:
        #     json.dump(train_dict, trv)
        dists = torch.zeros(MAX_SENT_LENGTH)
        diste = torch.zeros(MAX_SENT_LENGTH)
        if datum['landmarks'][0]['distance'] != '0':
            t_distance = self.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())
            # print(len(t_distance))

            # dist = torch.tensor([0,0],dtype=torch.int)
            # dist[0] = int(tokens.index(t_distance[0]))
            # dist[1] = int(tokens.index(t_distance[-1]))

            dists[int(tokens.index(t_distance[0]))]  = 1
            diste[int(tokens.index(t_distance[-1]))]  = 1
        else:
            # dist = torch.tensor([-1,-1], dtype=torch.int)

            # dist = torch.zeros(2,MAX_SENT_LENGTH)
            dists[-1]  = 1
            diste[-1]  = 1

        # input(tokens)
        # print(dist)
        # input(bearing)
        # dist = torch.tensor([int(datum['landmarks'][0]['distance'])])
        # bearing = torch.tensor([int(datum['landmarks'][0]['bearing'])])
        # print(target)
        # print(dist)
        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        # obj_num = img_info['t_num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        # boxes = img_info['t_boxes'].copy()
        # target = torch.tensor(datum['landmarks'][0]['raw_pixels'])
        # target = torch.tensor(boxes[-1]).float()
        # print(boxes)
        #
        # print(target)
        feat_mask = 0

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()

        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        if self.named_entities:


            names = img_info['names'].copy()
            # names = img_info['t_names'].copy()
            # input(names)
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

            if (MAX_BOXES - len(names_ids)) > 0:
                print("Zerppp")
                # Zero-pad up to the sequence length.
                padding = (MAX_BOXES - len(names_ids))*[torch.zeros(self.max_seq_length, dtype=torch.long)]

                feats_vis_padding = torch.zeros(((MAX_BOXES - feats.shape[0]),feats.shape[1]), dtype=torch.double)
                box_vis_padding = torch.zeros(((MAX_BOXES - boxes.shape[0]),boxes.shape[1]), dtype=torch.double)
                feats = torch.cat((feats,feats_vis_padding))
                boxes = torch.cat((boxes,box_vis_padding))

                names_ids = torch.stack(names_segment_ids + padding)
                names_segment_ids = torch.stack(names_segment_ids + padding)
                names_mask = torch.stack(names_mask + padding)

                    # bert hidden_size = 768
                feat_mask = torch.ones(feats.shape[0], dtype=torch.double)
                feats_padding = torch.zeros((MAX_BOXES - feats.shape[0]), dtype=torch.double)
                feat_mask = torch.cat((feat_mask,feats_padding))
            else:
                # print("No mask")
                # feats_vis_padding = torch.zeros(((100 - feats.shape[0]),feats.shape[1]), dtype=torch.double)
                # box_vis_padding = torch.zeros(((100 - boxes.shape[0]),boxes.shape[1]), dtype=torch.double)
                # feats = torch.cat((feats,feats_vis_padding))
                # boxes = torch.cat((boxes,box_vis_padding))
                names_ids = torch.stack(names_ids)
                names_segment_ids = torch.stack(names_segment_ids)
                names_mask = torch.stack(names_mask)
            # input(names_ids.shape)
            # feat_mask = torch.ones(names_ids.shape[0], dtype=torch.double)
            # feats_padding = torch.zeros((68 - feats.shape[0]), dtype=torch.double)
            # # input(feats_padding.shape)
            # feat_mask = torch.cat((feat_mask,feats_padding))

            _names = (names_ids, names_segment_ids, names_mask)
        else:
            _names = 0

            # assert obj_num == len(boxes) == len(feats)



        # x = np.random.randn(N, D_in)



        # boxes = np.random.rand(boxes.shape[0],boxes.shape[1])
        # feats = np.random.rand(feats.shape[0],feats.shape[1])

        # print(np.mean(feats))


        return sent_id, feats, feat_mask, boxes, _names, sent,dists, diste,landmark, bearing, target#bearing
            # else:
            #     return ques_id, feats, boxes, ques


class ROSMIEvaluator:
    def __init__(self, dataset: ROSMIDataset):
        self.dataset = dataset

    def evaluate(self, sentid2ans: dict):
        score = 0.
        score2 = 0.
        mDist = 0.
        for sentid, (pred_box, diss,dise, ln, br) in sentid2ans.items():
            siou2 = 0
            # datum = self.dataset.id2datum[sentid]
            # gold = torch.tensor(self.dataset.imgid2img[datum['img_id']]['boxes'][-1])
            datum = self.dataset.id2datum[sentid]
            # input(gold)
            filename = os.path.join('/home/marios/experiments/gps_prediction/ROSMI/ROSMI_dataset','images', datum["image_filename"])
            iou = calc_iou_individual(pred_box, datum['gold_pixels'])
            _scale = 25/SCALES[datum['scenario_items'].split('rio')[1].split('.json')[0]]
            siou = iou*_scale
            # iou2 = 1 - iou_loss(pred_box, datum['gold_pixels'])
            # if iou > 0:
            print("Stats:---------------")
            print(datum['sentence']['raw'])
            print(pred_box,datum['gold_pixels'])
            print(diss,dise, datum['landmarks'][0]['distance'])
            print(br, datum['landmarks'][0]['bearing'])

            if datum['landmarks'][0]['g_type'] == 'Point':
                print(ln, datum['landmarks'][0]['raw_pixels'])
            else:
                print(ln, datum['landmarks'][0]['landmark_pixels'])
            print(iou, siou)


            sn_id = int(datum['scenario_items'].split('rio')[1].split('.j')[0])

            centre = calculateTiles(CENTRES[sn_id],ZOOMS[sn_id])

            pred_coords = getPointLatLng(pred_box[0] + (pred_box[2] - pred_box[0])/2, pred_box[1] +(pred_box[3] - pred_box[1])/2,  \
                            CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)


            # print(GOLD_SIZES[sn_id], (ln[2] - ln[0])/2)
            # input()

            pred_land_coords = getPointLatLng(ln[0] + (ln[2] - ln[0])/2, ln[1] + (ln[3] - ln[1])/2,  \
                            CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)


            # if datum['landmarks'][0]['g_type'] == 'Point':
            #     print(pred_land_coords, datum['landmarks'][0]['raw_gps'])
            # else:
            #     print(pred_land_coords, datum['landmarks'][0]['landmark_gps'])
            # # print(pred_land_coords)
            # print(pred_coords, datum['gold_coordinates'])
            bearing = BEAR2NUMS[br]
            # start and end id of distance
            tokens = ["[CLS]"] + self.dataset.tokenizer.tokenize(datum['sentence']['raw'].strip()) + ["[SEP]"]
            # print(tokens)
            tmp_pixs = None
            if datum['landmarks'][0]['distance'] != '0':
                t_distance = self.dataset.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())

                if diss == int(tokens.index(t_distance[0])) and dise == int(tokens.index(t_distance[-1])):
                    _distance = int(datum['landmarks'][0]['distance'])

                    final_coord = destination([pred_land_coords[1], pred_land_coords[0]] , _distance, bearing)
                    # final_coord = destination([datum['landmarks'][0]['raw_gps'][0], datum['landmarks'][0]['raw_gps'][1]] , datum['landmarks'][0]['distance'], datum['landmarks'][0]['bearing'])

                    tmp_ob = {'g_type':'Point'}
                    tmp_ob['coordinates'] = final_coord
                    tmp_pixs = generatePixel(tmp_ob,centre,ZOOMS[sn_id],[ 700, 500], GOLD_SIZES[sn_id])

            else:

                if diss == (MAX_SENT_LENGTH-1) and dise == (MAX_SENT_LENGTH-1):
                    _distance = int(datum['landmarks'][0]['distance'])

                    final_coord = destination([pred_land_coords[1], pred_land_coords[0]] , _distance, bearing)
                    # final_coord = destination([datum['landmarks'][0]['raw_gps'][0], datum['landmarks'][0]['raw_gps'][1]] , datum['landmarks'][0]['distance'], datum['landmarks'][0]['bearing'])

                    tmp_ob = {'g_type':'Point'}
                    tmp_ob['coordinates'] = final_coord
                    tmp_pixs = generatePixel(tmp_ob,centre,ZOOMS[sn_id],[ 700, 500], GOLD_SIZES[sn_id])


            if tmp_pixs:
                px = tmp_pixs["points_x"]
                py = tmp_pixs["points_y"]
                new_bbox = [np.min(px), np.min(py), np.max(px), np.max(py)]
            # input(new_bbox)


                # try:
                #     img = Image.open(filename)
                # except Exception as e:
                #     print(e)
                #     continue



                # img_dim = [img.getbbox()[2],img.getbbox()[3]]
                # print(img_dim)

                # drawItem(['gold_pixels','predicted_pixels','landmark'],filename,pixels_bb=[datum['gold_pixels'],new_bbox,ln])
                iou = calc_iou_individual(new_bbox, datum['gold_pixels'])
                _scale = 25/SCALES[datum['scenario_items'].split('rio')[1].split('.json')[0]]
                siou2 = iou*_scale
                print(iou, siou2)
                # input("checking...")
            if siou > 0.65:
                # print("ONE CORRECT")
            # if ans in label:
                score2 += 1
            # gold_coords = getPointLatLng(datum['gold_pixels'][0]+GOLD_SIZES[sn_id], datum['gold_pixels'][1]+GOLD_SIZES[sn_id],  \
            #                 CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)
            # print(datum['gold_coordinates'])
            # print(gold_coords)
            # print(haversine(gold_coords[1],gold_coords[0],datum['gold_coordinates'][0],datum['gold_coordinates'][1])*1000)
            distance = haversine(pred_coords[1],pred_coords[0],datum['gold_coordinates'][0],datum['gold_coordinates'][1])*1000
            print(f"Distance is {distance}m")
            mDist += distance
            if siou > 0.65 or siou2 > 0.65:
                # print("ONE CORRECT")
            # if ans in label:
                score += 1
        # if score >=50:
        #     input("?")
            # if score2 > score:
            #     print("Better!!")
            #     score = score2

        return score / len(sentid2ans), mDist / len(sentid2ans), score2 / len(sentid2ans)

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
