# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os, sys, math, ast
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import *
from lxrt.entry import convert_sents_to_features

from lxrt.tokenization import BertTokenizer
from transformers import BertTokenizer as hBertToken

SCALES = [25,25,4,12,4,4,4]
SCALES2 = [1,1,0.12486,0.49958,0.12486,0.12486,0.12486]
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
#  centers in lat, lon
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
MAX_BOXES = 10
# The path to data and image features.
# VQA_DATA_ROOT = '/scratch/mmk11/data/vqa/'
# IMGFEAT_ROOT = '/scratch/mmk11/data/rosmi/'

class ROSMIDataset:
    """
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
        self.htokenizer = hBertToken.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(os.path.join(args.data_path,"%s.json" % split))))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # input(self.data[0])
        # Convert list to dict (for evaluation)
        # self.id2datum = {
        #     id_k : datum
        #     for id_k, datum in enumerate(self.data)
        # }
        # input(self.id2datum[0])
        # if args.tiny:
        #     topk = TINY_IMG_NUM
        # elif args.fast:
        #     topk = FAST_IMG_NUM
        # else:
        #     topk = None


        IMGFEAT_ROOT = args.data_path
        # Loading detection features to img_data
        img_data = []

        img_data.extend(load_det_obj_tsv(
                os.path.join(IMGFEAT_ROOT, f'{split}_obj36.tsv')))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            # c = list(zip(img_datum['t_names'].tolist(), img_datum['t_boxes'].tolist()))
            # random.shuffle(c)
            # a, b = zip(*c)
            # img_datum['t_names'] = np.array(a,dtype='<U100')
            # img_datum['t_boxes'] = np.array(b)

            self.imgid2img[img_datum['img_id']] = img_datum

        self.id2datum = {}
        # input(self.data[0])
        for id_k, datum in enumerate(self.data):
            try:
                # print(datum['instruction'])
                # print(self.imgid2img[str(id_k)]['box_order'])
                # print(datum['block_moved_index'])
                landmark_id = datum['block_moved_index']
                # landmark_id = np.where(self.imgid2img[str(id_k)]['box_order'] == datum['block_moved_index'])[0][0]
                # print(type(landmark_id))
                # input(landmark_id)
                # last is reserved for landmarks that do not appear in the input feat
                landmark_id_ = torch.zeros(MAX_BOXES)
                landmark_id_[int(landmark_id)] = 1
                datum['target'] = landmark_id_
                datum['sent_id'] = str(id_k)
                # input(datum)
                self.id2datum[str(id_k)] = datum
            except Exception as e:
                # print(datum)
                # print(e)
                # print(self.id2datum)
                pass
        self.data = self.id2datum.values()
        print(f"Data remaining: {len(self.id2datum)}")

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
        self.htokenizer = hBertToken.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # # Convert img list to dict
        self.imgid2img = self.raw_dataset.imgid2img

        # Only kept the data with loaded image features
        self.data = []
        for k_ids,datum in enumerate(self.raw_dataset.data):
            # print(self.imgid2img)
            if datum['sent_id'] in self.imgid2img:
                self.data.append(datum)
            else:
                print(self.imgid2img)
                input(k_ids)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        # print(item)
        img_id = datum['sent_id']
        sent_id = datum['sent_id']
        sent = datum['instruction']

        # if datum['landmarks'][0]['g_type'] != 'LineString':
        #     landmark = torch.tensor(datum['landmarks'][0]['raw_pixels'])
        # else:
        #     landmark = torch.tensor(datum['landmarks'][0]['landmark_pixels'])


        # target = torch.tensor(datum['gold_pixels'])
        target = torch.tensor([1,1,1,1])
        landmark = torch.tensor([1,1,1,1])
        #
        #
        #
        bearing = torch.zeros(self.raw_dataset.num_bearings)
        bearing[self.raw_dataset.bearing2label[datum['bearing_to_target_enum']]] = 1

        distance = torch.tensor([datum['distance_to_target']])
        world_st = torch.tensor(ast.literal_eval(datum['world_state']))
        eu_dist_ed_co = torch.cat(( torch.tensor(ast.literal_eval(datum['euclidean_distance_to_edges'])), torch.tensor(ast.literal_eval(datum['euclidean_distance_to_corners']))), 1)
        # input(eu_dist_ed_co.shape)
        is_stacked = torch.tensor(ast.literal_eval(datum['is_stacked'])).unsqueeze(1)
        # print(is_stacked)
        # input(is_stacked.shape)
        ew_split = torch.tensor(ast.literal_eval(datum['EW_split'])).unsqueeze(1)
        ns_split = torch.tensor(ast.literal_eval(datum['NS_split'])).unsqueeze(1)
        # distance = torch.tensor([datum['EW_split']])
        # print(ns_split)
        # input(ns_split.shape)
        # distance = torch.tensor([datum['NS_split']])
        stacked_ew_ns = torch.cat(( ns_split, ew_split, is_stacked), 1)
        # stacked_ew_ns = torch.cat(( stacked_ew_ns, is_stacked), 1)
        # print(stacked_ew_ns)
        # print(type(stacked_ew_ns))
        # input(stacked_ew_ns.shape)


        # # start and end id of distance
        # tokens = ["[CLS]"] + self.tokenizer.tokenize(sent.strip()) + ["[SEP]"]
        #
        dists = torch.zeros(MAX_SENT_LENGTH)
        diste = torch.zeros(MAX_SENT_LENGTH)
        # if datum['landmarks'][0]['distance'] != '0':
        #     t_distance = self.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())
        #     dists[int(tokens.index(t_distance[0]))]  = 1
        #     diste[int(tokens.index(t_distance[-1]))]  = 1
        # else:
        #     dists[-1]  = 1
        #     diste[-1]  = 1

        # input(self.imgid2img.keys())
        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        # obj_num = img_info['t_num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        # print(type(boxes))
        # boxes = np.array(ast.literal_eval(datum['bounding_boxes']))
        names = img_info['names'].copy()

        # print(datum['bounding_boxes'])
        # input(boxes)

        # print(img_info['box_order'])
        # names = img_info['t_names'].copy()
        # boxes = img_info['t_boxes'].copy()
        # print(datum['block_moved_index'])
        # print(img_info['box_order'])
        # landmark_id = np.where(img_info['box_order'] == datum['block_moved_index'])[0][0]
        # print(type(landmark_id))
        # print(landmark_id)
        # # last is reserved for landmarks that do not appear in the input feat
        # landmark_id_ = torch.zeros(MAX_BOXES)
        # landmark_id_[int(landmark_id)] = 1
        landmark_id_ = datum['target']
        # input(landmark_id_)
        feat_mask = 0

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()

        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        feats = torch.from_numpy(feats)
        boxes = torch.from_numpy(boxes)

        _names = 0
        landmark_start = 0
        landmark_end = 0
        # print(input_ids)
        # input(_names[0])


        if self.named_entities:


            names_ids = []
            names_segment_ids = []
            names_mask = []
            for obj in names:
                names_features = convert_sents_to_features(
                    obj, self.max_seq_length, self.tokenizer)

                # for f in names_features
                names_ids.append(torch.tensor(names_features[0].input_ids, dtype=torch.long))
                names_segment_ids.append(torch.tensor(names_features[0].segment_ids, dtype=torch.long))
                names_mask.append(torch.tensor(names_features[0].input_mask, dtype=torch.long))



            if (MAX_BOXES - boxes.shape[0]) > 0:
                feat_mask = torch.ones(boxes.shape[0], dtype=torch.double)
                feats_padding = torch.zeros((MAX_BOXES - boxes.shape[0]), dtype=torch.double)
                feat_mask = torch.cat((feat_mask,feats_padding))
                # Zero-pad up to the sequence length.
                padding = (MAX_BOXES - boxes.shape[0])*[torch.zeros(self.max_seq_length, dtype=torch.long)]

                feats_vis_padding = torch.zeros(((MAX_BOXES - feats.shape[0]),feats.shape[1]), dtype=torch.double)
                box_vis_padding = torch.zeros(((MAX_BOXES - boxes.shape[0]),boxes.shape[1]), dtype=torch.double)
                feats = torch.cat((feats,feats_vis_padding))
                boxes = torch.cat((boxes,box_vis_padding))

                names_ids = torch.stack(names_ids + padding)
                names_segment_ids = torch.stack(names_segment_ids + padding)
                names_mask = torch.stack(names_mask + padding)

                    # bert hidden_size = 768
            else:

                names_ids = torch.stack(names_ids)
                names_segment_ids = torch.stack(names_segment_ids)
                names_mask = torch.stack(names_mask)
                # input(names_ids.shape)
                feat_mask = torch.ones(boxes.shape[0], dtype=torch.double)
                feats_padding = torch.zeros((MAX_BOXES - boxes.shape[0]), dtype=torch.double)
                # # input(feats_padding.shape)
                feat_mask = torch.cat((feat_mask,feats_padding))

            _names = (names_ids, names_segment_ids, names_mask)
        else:
            if (MAX_BOXES - boxes.shape[0]) > 0:
                feat_mask = torch.ones(boxes.shape[0], dtype=torch.double)
                feats_padding = torch.zeros((MAX_BOXES - boxes.shape[0]), dtype=torch.double)
                feat_mask = torch.cat((feat_mask,feats_padding))
                # Zero-pad up to the sequence length.
                # padding = (MAX_BOXES - len(boxes))*[torch.zeros(self.max_seq_length, dtype=torch.long)]

                feats_vis_padding = torch.zeros(((MAX_BOXES - feats.shape[0]),feats.shape[1]), dtype=torch.double)
                box_vis_padding = torch.zeros(((MAX_BOXES - boxes.shape[0]),boxes.shape[1]), dtype=torch.double)
                world_st_padding = torch.zeros(((MAX_BOXES - world_st.shape[0]),world_st.shape[1]), dtype=torch.double)
                eu_dist_ed_co_padding = torch.zeros(((MAX_BOXES - eu_dist_ed_co.shape[0]),eu_dist_ed_co.shape[1]), dtype=torch.double)
                stacked_ew_ns_padding = torch.zeros(((MAX_BOXES - stacked_ew_ns.shape[0]),stacked_ew_ns.shape[1]), dtype=torch.double)

                feats = torch.cat((feats,feats_vis_padding))
                boxes = torch.cat((boxes,box_vis_padding))
                world_st = torch.cat((world_st,world_st_padding))
                eu_dist_ed_co = torch.cat((eu_dist_ed_co,eu_dist_ed_co_padding))
                stacked_ew_ns = torch.cat((stacked_ew_ns,stacked_ew_ns_padding))


            else:
                feat_mask = torch.ones(boxes.shape[0], dtype=torch.double)
                feats_padding = torch.zeros((MAX_BOXES - boxes.shape[0]), dtype=torch.double)
                # input(feats_padding.shape)
                feat_mask = torch.cat((feat_mask,feats_padding))
            # _names = 0



        return sent_id, feats, feat_mask, boxes, _names, sent,dists, diste,landmark, landmark_id_, bearing, world_st, eu_dist_ed_co, stacked_ew_ns, landmark_start,landmark_end, distance#bearing
        # return sent_id, feats, feat_mask, boxes, _names, sent,dists, diste,landmark, torch.tensor([landmark_id]), bearing, target#bearing
            # else:
            #     return ques_id, feats, boxes, ques


class ROSMIEvaluator:
    def __init__(self, dataset: ROSMIDataset):
        self.dataset = dataset



    def evaluate(self, sentid2ans: dict):
        target_score = 0.
        meta_score = 0.
        block_distance = []
        source_distance = []
        br_score = 0.
        dist_score = 0.
        tagging_score = 0.
        meanDist = []
        pixDiff = []
        mDist = 0.
        lands = 0
        counterDist = 0
        thres = 0.50
        # {id:'', sentence:'',gold:[a,b,c],pred:[a,b,c],outcome:True }
        examples = []
        # scenarios = {'scenario0.json':[0,0],'scenario1.json':[0,0],'scenario2.json':[0,0],'scenario3.json':[0,0],'scenario4.json':[0,0],'scenario5.json':[0,0],'scenario6.json':[0,0]}
        for sentid, (distance, diss,dise, ln, ln_, br, l_s,l_e) in sentid2ans.items():



            siou = 0
            siou3 = 0
            # distance2 = None



            datum = self.dataset.id2datum[sentid]
            img_info = self.dataset.imgid2img[sentid]
            # scenarios[datum['scenario_items']][1] += 1
            # obj_num = img_info['num_boxes']
            # # obj_num = img_info['t_num_boxes']
            feats = img_info['features'].copy()
            boxes = img_info['boxes'].copy()
            names = img_info['names'].copy()
            # boxes = img_info['t_boxes'].copy()
            # names = img_info['t_names'].copy()
            sent = datum['instruction']
            # landmark_id_ = 0
            # landmark_id_ = random.randint(0,67)
            # for ipd, name_box in enumerate(names):
            #     if "".join(datum['landmarks'][0]['name'].split(" ")).lower()  == "".join(name_box[0].split(" ")).lower():
            #         landmark_id_ = ipd
            #         break
            #
            #
            #
            # sn_id = int(datum['scenario_items'].split('rio')[1].split('.j')[0])
            # # filename = os.path.join('/home/marios/experiments/gps_prediction/ROSMI/ROSMI_dataset','images', datum["image_filename"])
            # iou = calc_iou_individual(pred_box, datum['gold_pixels'])
            # _scale = 25/SCALES[sn_id]
            # siou = iou*_scale
            # # iou2 = 1 - iou_loss(pred_box, datum['gold_pixels'])
            # # if iou > 0:


            # landmark_id_ = img_info['box_order'][datum['block_moved_index']]

            _, landmark_id_ = datum['target'].max(0)
            print(landmark_id_)
            # start and end id of distance
            # tokens = ["[CLS]"] + self.dataset.tokenizer.tokenize(datum['sentence']['raw'].strip()) + ["[SEP]"]

            # dists = torch.zeros(MAX_SENT_LENGTH)
            # diste = torch.zeros(MAX_SENT_LENGTH)
            # if datum['landmarks'][0]['distance'] != '0':
            #     # t_distance = self.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())
            #     t_distance = self.dataset.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())
            #
            #     start_ = int(tokens.index(t_distance[0]))
            #     dists[start_]  = 1
            #     diste[int(tokens[start_:].index(t_distance[-1]))+start_]  = 1
            # else:
            #     dists[-1]  = 1
            #     diste[-1]  = 1
            #
            #
            # dists = np.argmax(dists).item()
            # diste = np.argmax(diste).item()
            # landmark_id =img_info['box_order'][ln_]
            # tmp_dist = torch.tensor([datum['distance_to_target']])
            print("Stats:---------------")
            print(datum['instruction'])
            print(distance,datum['distance_to_target'])
            # print(diss,dise, datum['landmarks'][0]['distance'], dists, diste)
            print(br, datum['bearing_to_target_enum'])
            # print(ln, datum['landmarks'][0]['raw_pixels'])
            try:
                print(f"Landmark ids: {landmark_id_} {names[int(landmark_id_)]} - {ln_} {names[int(ln_)]} - {boxes[landmark_id_]} - {boxes[ln_]}")
            except Exception as e:
                print(f"Cannot print stats because {e}")




            # # gold landmark
            # ln_ = landmark_id_

            # # gold bearing
            # br = datum['bearing_to_target_enum']
            #
            # centre = calculateTiles(CENTRES[sn_id],ZOOMS[sn_id])
            # print(br)
            # print(type(br))
            # print(distance)
            # print(type(distance))
            # print(tmp_dist)
            # input(type(tmp_dist))
            # correct_distance =  math.sqrt( (ast.literal_eval(datum['world_state'])[int(img_info['box_order'][ln_])][2] - datum['block_moved_to_position'][2])**2 + (ast.literal_eval(datum['world_state'])[int(img_info['box_order'][ln_])][0] - datum['block_moved_to_position'][0])**2 )

            correct_distance = distance[0]
            # gold distance:
            # correct_distance = datum['distance_to_target']
            # print(f"Distance from gold and source should be: {correct_distance}")
            # print(distance[0])
            # print(math.sin(math.radians(BEAR2NUMS[br])))
            # print(f"World state with id {img_info['box_order'][ln_]} is {ast.literal_eval(datum['world_state'])[int(img_info['box_order'][ln_])]}")
            # calculate new target point x and y. datum['world_state'] = xyz y is constant and is the height!
            # new_x = ast.literal_eval(datum['world_state'])[int(img_info['box_order'][ln_])][0]
            new_x = ast.literal_eval(datum['world_state'])[int(ln_)][0]
            new_z = ast.literal_eval(datum['world_state'])[int(ln_)][2]
            new_y = ast.literal_eval(datum['world_state'])[int(ln_)][1]
            new_x = math.sin(math.radians(BEAR2NUMS[br])) * correct_distance + new_x
            new_z = math.cos(math.radians(BEAR2NUMS[br])) * correct_distance + new_z
            # print(f"Gold: {datum['block_moved_to_position']}, predicted: [{new_x},{ast.literal_eval(datum['world_state'])[int(img_info['box_order'][ln_])][1]},{new_z}]")
            # print(f"distance from gold target is { math.sqrt( (new_z - datum['block_moved_to_position'][2])**2 + (new_x - datum['block_moved_to_position'][0])**2 )}")
            # input(f"distance from gold and source is { math.sqrt( (ast.literal_eval(datum['world_state'])[int(img_info['box_order'][ln_])][2] - datum['block_moved_to_position'][2])**2 + (ast.literal_eval(datum['world_state'])[int(img_info['box_order'][ln_])][0] - datum['block_moved_to_position'][0])**2 )}")

            block_distance += [math.sqrt( (new_z - datum['block_moved_to_position'][2])**2 + (new_x - datum['block_moved_to_position'][0])**2 )]
            source_distance += [math.sqrt( (ast.literal_eval(datum['world_state'])[int(ln_)][2] - ast.literal_eval(datum['world_state'])[int(landmark_id_)][2])**2 + (ast.literal_eval(datum['world_state'])[int(ln_)][0] - ast.literal_eval(datum['world_state'])[int(landmark_id_)][0])**2 )]
            # if abs(distance[0] - datum['distance_to_target']) < 0.05:
            dist_score += abs(distance[0] - datum['distance_to_target'])
            if br == datum['bearing_to_target_enum']:
                br_score += 1
            if landmark_id_ == ln_:
                lands += 1
                meta_score +=1
            try:

                print(boxes[landmark_id_],boxes[ln_])

                # pred_cland_coords = getPointLatLng(boxes[ln_][0] + (boxes[ln_][2] - boxes[ln_][0])/2, boxes[ln_][1] + (boxes[ln_][3] - boxes[ln_][1])/2,  \
                #                         CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)
            except:
                pred_cland_coords = None

            # print(iou, siou)


            # pred_coords = getPointLatLng(pred_box[0] + (pred_box[2] - pred_box[0])/2, pred_box[1] +(pred_box[3] - pred_box[1])/2,  \
            #                 CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)


            # pred_land_coords = getPointLatLng(ln[0] + (ln[2] - ln[0])/2, ln[1] + (ln[3] - ln[1])/2,  \
            #                 CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)

            # bearing = BEAR2NUMS[br]
            # tmp_pixs2 = None
            # final_coord2 = None


            # if datum['landmarks'][0]['distance'] != '0':
                # t_distance = self.dataset.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())

                # if diss == int(tokens.index(t_distance[0])) and dise == int(tokens.index(t_distance[-1])):
            # if diss == dists and dise == diste:
            #     _distance = int(datum['landmarks'][0]['distance'])
            #
            #     if pred_cland_coords:
            #         final_coord2 = destination([pred_cland_coords[1], pred_cland_coords[0]] , _distance, bearing)
            #         # final_coord = destination([datum['landmarks'][0]['raw_gps'][0], datum['landmarks'][0]['raw_gps'][1]] , datum['landmarks'][0]['distance'], datum['landmarks'][0]['bearing'])
            #
            #         tmp_ob = {'g_type':'Point'}
            #         tmp_ob['coordinates'] = final_coord2
            #         tmp_pixs2 = generatePixel(tmp_ob,centre,ZOOMS[sn_id],[ 700, 500], GOLD_SIZES[sn_id])

            # if final_coord2:
            #     distance2 = haversine(final_coord2[0],final_coord2[1],datum['gold_coordinates'][0],datum['gold_coordinates'][1])*1000
            #     if distance2 < 1:
            #         scenarios[datum['scenario_items']][0] += 1

            # if distance2:
            #     mDist += distance2
            #     distance2 = distance2*SCALES2[sn_id]
            #     meanDist.append(distance2)
            #
            # else:
            #     counterDist +=1
            #
            # print(f"Distance is {distance2}m")


            # if tmp_pixs2:
            #     px = tmp_pixs2["points_x"]
            #     py = tmp_pixs2["points_y"]
            #     new_bbox2 = [np.min(px), np.min(py), np.max(px), np.max(py)]
            #
            #     # try:
            #     #     img = Image.open(filename)
            #     # except Exception as e:
            #     #     print(e)
            #     #     continue
            #
            #     prd_center = [new_bbox2[0] + (new_bbox2[2] - new_bbox2[0])/2, new_bbox2[1] + (new_bbox2[3] - new_bbox2[1])/2]
            #     gold_center = [datum['gold_pixels'][0] + (datum['gold_pixels'][2] - datum['gold_pixels'][0])/2, datum['gold_pixels'][1] + (datum['gold_pixels'][3] - datum['gold_pixels'][1])/2]
            #
            #
            #     pixDiff.append(sqrt((int(prd_center[1]-gold_center[1]))**2 + (int(prd_center[0]-gold_center[0]))**2))
            #
            #     iou = calc_iou_individual(new_bbox2, datum['gold_pixels'])
            #     _scale = 25/SCALES[sn_id]
            #     # siou3 = iou*_scale
            #     siou3 = iou/SCALES2[sn_id]
            #     print(iou*_scale)
            #     print(siou3)
            #     # input(iou/SCALES2[datum['scenario_items'].split('rio')[1].split('.json')[0]])
            #     if siou3 > thres:
            #         # print("ONE CORRECT")
            #     # if ans in label:
            #         meta_score += 1
            #     # drawItem(['gold_pixels','predicted_pixels','landmark'],filename,pixels_bb=[datum['gold_pixels'],new_bbox,ln])

            # if siou > thres:
            #     target_score += 1
            # gold_coords = getPointLatLng(datum['gold_pixels'][0]+GOLD_SIZES[sn_id], datum['gold_pixels'][1]+GOLD_SIZES[sn_id],  \
            #                 CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)
            # print(datum['gold_coordinates'])
            # print(gold_coords)
            # print(haversine(gold_coords[1],gold_coords[0],datum['gold_coordinates'][0],datum['gold_coordinates'][1])*1000)
            # distance = haversine(pred_coords[1],pred_coords[0],datum['gold_coordinates'][0],datum['gold_coordinates'][1])*1000


            # try:
            #     save_land = str(names[ln_])
            # except Exception as e:
            #     print(f"No examples because {e}")
            #     save_land = str(None)
            # examples.append({ 'id':sentid, 'img_id':datum['img_id'], 'sentence':sent, 'gold':[str(names[landmark_id_]),str(datum['landmarks'][0]['distance'])+' '+str(dists)+ ' '+str(diste),str(datum['landmarks'][0]['bearing'])], 'pred':[save_land,str(diss)+ ' '+str(dise),str(br)], 'outcome': str(siou3 > thres), 'distance':distance2 })
            #

        print(f"Target Score: {target_score / len(sentid2ans)}, Meta Score: {meta_score / len(sentid2ans)}, Bearing Score: {br_score / len(sentid2ans)},  \
                    Mean Distance: {np.mean(block_distance)/ 0.1524}")
        # if len(pixDiff) > 0.2*len(sentid2ans):
        #     # meanD =  mDist / (len(sentid2ans) - counterDist)
        #     pixMean = int(np.mean(pixDiff))
        #     # variance = int(np.var(pixDiff))
        #     pixsd_ = int(np.std(pixDiff))
        #     distMean = int(np.mean(meanDist))
        #     # variance = int(np.var(pixDiff))
        #     distsd_ = int(np.std(meanDist))
        # else:
        #     pixMean = 99999999
        #     distMean = 99999999
        #     distsd_ = 99999999
        #     pixsd_ = 99999999
        print(len(sentid2ans))
        print(lands/len(sentid2ans))
        # print(f"Mean distance , Mean pix : {distMean} [{distsd_}] , {pixMean} [{pixsd_}]")
        return target_score / len(sentid2ans), (np.mean(block_distance)/ 0.1524, np.median(block_distance)/ 0.1524,np.std(block_distance)/ 0.1524),(br_score / len(sentid2ans), dist_score/len(sentid2ans)),(meta_score / len(sentid2ans), np.mean(source_distance)/ 0.1524, np.median(source_distance)/ 0.1524,np.std(source_distance)/ 0.1524)
        # return target_score / len(sentid2ans), (distMean,distsd_,pixMean,pixsd_,scenarios,examples),tagging_score / len(sentid2ans),meta_score / len(sentid2ans)
