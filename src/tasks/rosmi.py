# coding=utf-8
# Copyleft 2019 project LXRT.

import os, time
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.rosmi_model import ROSMIModel
from tasks.rosmi_data import ROSMIDataset, ROSMITorchDataset, ROSMIEvaluator
from torch.utils.tensorboard import SummaryWriter
from utils import iou_loss, giou_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = ROSMIDataset(splits)
    tset = ROSMITorchDataset(dset)
    evaluator = ROSMIEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class ROSMI:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=args.batch_size,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None


        self.writer = SummaryWriter(f'snap/rosmi/logging_rosmi_{args.n_ent}_names/{os.uname()[1]}.{time.time()}')
        # Model
        self.model = ROSMIModel(self.train_tuple.dataset.num_bearings)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.SmoothL1Loss()

        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        self.scheduler = ReduceLROnPlateau(self.optim, 'min')
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)


        # self.writer.add_graph(self.model,loader)
        # self.writer.close()
        best_valid = 0.
        best_train = 0.
        loss = 99999
        n_iter = 0
        for epoch in tqdm(range(args.epochs)):
            sentid2ans = {}
            for i, (sent_id, feats, feat_mask, boxes, names, sent, target) in iter_wrapper(enumerate(loader)):

                # input("lol")
                self.model.train()
                self.optim.zero_grad()

                # print(feats.shape)
                # print(names[0].squeeze(2).cuda())
                # print(names[1].shape)
                # input(names[2].shape)
                if args.n_ent:
                    names = (names[0].squeeze(2).cuda(), \
                                  names[1].squeeze(2).cuda(), \
                                  names[2].squeeze(2).cuda())
                else:
                    names = None

                feats, feat_mask, boxes, target = feats.cuda(), feat_mask.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats.float(), feat_mask.float(), boxes.float(), names, sent)

                # assert logit.dim() == target.dim() == 2
                loss = self.mse_loss(logit, target)
                print(logit)
                print(target)
                # loss = iou_loss(logit, target)
                iou,loss2 = giou_loss(logit, target)
                # print(iou,giou)
                # input(loss)
                # if not loss:
                #     print("Not ready yet")
                #     loss = self.mse_loss(logit, target)
                # print(loss)
                # print(loss2)
                # if loss > 100:
                #     loss = loss + loss2
                # else:
                #     loss = loss2
                loss = loss + loss2
                print(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                label = logit
                # input(logit)
                # score, label = logit.max(1)
                for sid, l in zip(sent_id, label.cpu().detach().numpy()):
                    # ans = dset.label2ans[l]
                    sentid2ans[sid.item()] = l


                self.writer.add_scalar('Loss/train', loss, n_iter)

                n_iter += 1
                # writer.add_scalar('Loss/test', np.random.random(), n_iter)

            # self.scheduler.step(loss)
            log_str = f"\nEpoch {epoch}: Loss {loss}\n"
            tmp_acc = evaluator.evaluate(sentid2ans)
            log_str += f"\nEpoch {epoch}: Train {tmp_acc * 100.}%\n"
            self.writer.add_scalar('Accuracy/train [IoU=0.5]', tmp_acc * 100., n_iter)
            # awlf.writer.close()

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")
                if tmp_acc > best_train:
                    best_train = tmp_acc

                self.writer.add_scalar('Accuracy/valid [IoU=0.5]', valid_score * 100., n_iter)
                # awlf.writer.close()
                log_str += f"Epoch {epoch}: Valid {valid_score * 100.}%\n" + \
                           f"Epoch {epoch}: Best Train {best_train * 100.}%\n" + \
                           f"Epoch {epoch}: Best Val {best_valid * 100.}%\n"

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        sentid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, feat_mask, boxes, names, sent = datum_tuple[:6]   # Avoid seeing ground truth
            with torch.no_grad():
                if args.n_ent:
                    names = (names[0].squeeze(2).cuda(), \
                                  names[1].squeeze(2).cuda(), \
                                  names[2].squeeze(2).cuda())
                else:
                    names = None
                feats, feat_mask, boxes = feats.cuda(),feat_mask.cuda(), boxes.cuda()
                logit = self.model(feats.float(), feat_mask.float(), boxes.float(), names, sent)
                label = logit
                for qid, l in zip(ques_id, label.cpu().detach().numpy()):
                    # ans = dset.label2ans[l]
                    sentid2ans[qid.item()] = l
        if dump is not None:
            evaluator.dump_result(sentid2ans, dump)
        return sentid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        sentid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(sentid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        sentid2ans = {}

        for i, (ques_id, feats, feat_mask, boxes, names, sent, target) in enumerate(loader):
            # input(target)
            label = target
            for qid, l in zip(ques_id, label.cpu().numpy()):
                # ans = dset.label2ans[l]
                sentid2ans[qid.item()] = l
        return evaluator.evaluate(sentid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    rosmi = ROSMI()
    # Load ROSMI model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        rosmi.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            rosmi.predict(
                get_data_tuple(args.test, bs=args.batch_size,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = rosmi.evaluate(
                get_data_tuple('val', bs=args.batch_size,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'val_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', rosmi.train_tuple.dataset.splits)
        if rosmi.valid_tuple is not None:
            print('Splits in Valid data:', rosmi.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (rosmi.oracle_score(rosmi.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        rosmi.train(rosmi.train_tuple, rosmi.valid_tuple)
