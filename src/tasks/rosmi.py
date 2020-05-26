# coding=utf-8
# Copyleft 2019 project LXRT.

import os, time
import collections

import torch, json

import torch.nn as nn
import numpy as np
from torch.autograd import Variable
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

MAX_VQA_LENGTH = 25
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
        if args.test:
            self.test_tuple = get_data_tuple(
                args.test, bs=args.batch_size,
                shuffle=False, drop_last=False
            )
        else:
            self.test_tuple = None

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
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.SmoothL1Loss()

        if 'bert' in args.optim:
            # input("bert")
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            # t_total = -1
            # batch 24 when 20 and epochs 3000 = 72000
            # input(int(batch_per_epoch * args.epochs))
            t_total = 72000
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
            # input("AdamW")
            # self.optim = torch.optim.AdamW(self.model.parameters(), args.lr, weight_decay=0.1,amsgrad=True)

        # self.scheduler = ReduceLROnPlateau(self.optim, 'min')
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
        best_acc2 = 0.
        best_tacc = 0
        best_acc3 = 0
        best_test_acc = 0
        best_mDist = [99999,99999,99999,99999]
        best_testDist = [99999,99999,99999,99999]
        n_iter = 0
        for epoch in tqdm(range(args.epochs)):
            sentid2ans = {}
            for i, (sent_id, feats, feat_mask, boxes, names, sent, dists, diste, land_, cland_, bear_ ,l_start,l_end, target) in iter_wrapper(enumerate(loader)):

                total_loss = 0
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
                elif args.qa:
                    names = (names[0].cuda(), \
                                  names[1].cuda(), \
                                  names[2].cuda())
                else:
                    names = None

                feats, feat_mask, boxes, target, dists, diste, land_, cland_, bear_, l_start, l_end = feats.cuda(), feat_mask.cuda(), boxes.cuda(), target.cuda(), dists.cuda(), diste.cuda(), \
                                                                                land_.cuda(), cland_.cuda(), bear_.cuda(), l_start.cuda(), l_end.cuda()
                logit, auxilaries = self.model(feats.float(), feat_mask.float(), boxes.float(), names, sent)
                # print(names.shape)
                # input(sent.shape)
                # print(type(sent))
                # if i == 0:
                #
                #     # input(sent)
                #     # input(len(sent))
                #     tmpInd = torch.ones(len(sent),MAX_VQA_LENGTH)
                #     tmpNames = torch.ones(len(sent),MAX_VQA_LENGTH)
                #     # print(type(tmpInd))
                #     # print(type(tmpNames))
                #     # input(sent)
                #     # input(type(names))
                #     self.writer.add_graph(self.model, (feats.float(), feat_mask.float(), boxes.float(),names,tmpInd ))
                # assert logit.dim() == target.dim() == 2

                # target_loss = self.mse_loss(logit, target)
                # self.writer.add_scalar('target loss', target_loss, n_iter)
                # total_loss += target_loss*logit.size(1)*4
                # print(logit.size(1))
                # print(target)
                # total_loss += iou_loss(logit, target)

                iou,loss2 = giou_loss(logit, target)
                self.writer.add_scalar('giou loss', loss2, n_iter)
                # total_loss += loss2

                p_dist_s, p_dist_e, p_land, p_cland, p_bear, p_start, p_end = auxilaries

                assert logit.dim() == target.dim() == 2
                bear_loss = self.bce_loss(p_bear,bear_.float())
                self.writer.add_scalar('Bearing loss', bear_loss, n_iter)

                total_loss += bear_loss* p_bear.size(1) * 2

                dists_loss = self.bce_loss(p_dist_s,dists.float())
                self.writer.add_scalar('distance Start loss', dists_loss, n_iter)
                total_loss += dists_loss* p_dist_s.size(1) * 2

                diste_loss = self.bce_loss(p_dist_e,diste.float())
                self.writer.add_scalar('distance End loss', diste_loss, n_iter)
                total_loss += diste_loss* p_dist_e.size(1) * 2

                # print(p_cland)
                # print(cland_)
                # print(p_cland.shape)
                # input(cland_.shape)
                if args.qa:
                    cland_loss = self.bce_loss(p_start,l_start) + self.bce_loss(p_end,l_end)
                    total_loss += cland_loss* p_start.size(1) * 2
                else:
                    cland_loss = self.bce_loss(p_cland,cland_)

                    self.writer.add_scalar('Cls Landmark loss', cland_loss, n_iter)
                    total_loss += cland_loss* p_cland.size(1) * 4


                # total_loss /=4
                # loss += self.mse_loss(p_dist,dist.float())#*p_dist.size(1)
                # print(land_)
                # input(land_.float())
                # land_loss = self.mse_loss(p_land,land_.float())#*p_land.size(1)
                # self.writer.add_scalar('landmark loss', land_loss, n_iter)
                # total_loss += land_loss*p_land.size(1)

                # total_loss += self.mse_loss(p_bear,bear_.float())#*p_bear.size(1)

                # print(p_dist,torch.Tensor([[int(di)]for di in dist]))
                # input(total_loss)
                # if not total_loss:
                #     print("Not ready yet")
                #     total_loss = self.mse_loss(logit, target)
                # print(loss)
                # print(loss2)
                # if land_loss < 50 and diste_loss < 0.01:
                #
                #     total_loss += self.mse_loss(logit, target)*logit.size(1)
                    # total_loss = loss + loss2
                # else:
                #     total_loss = loss2
                # total_loss = loss + loss2

                self.writer.add_scalar('total loss', total_loss, n_iter)
                # print(loss)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                label = logit
                # input(logit)
                bear_score, bear_label = p_bear.max(1)
                # if args.qa:
                _, p_start = p_start.max(1)
                _, p_end = p_end.max(1)
                _, p_cland = p_cland.max(1)
                _, dist_e = p_dist_e.max(1)
                _, dist_s = p_dist_s.max(1)
                # score, label = logit.max(1)
                for sid,diss,dise,ln,ln_,br,l_s,l_e, l in zip(sent_id, dist_s.cpu().detach().numpy(), \
                                                dist_e.cpu().detach().numpy(), \
                                                p_land.cpu().detach().numpy(), \
                                                p_cland.cpu().detach().numpy(), \
                                                bear_label.cpu().detach().numpy(), \
                                                p_start.cpu().detach().numpy(), \
                                                p_end.cpu().detach().numpy(), \
                                                    label.cpu().detach().numpy()):

                    br = dset.label2bearing[br]
                    # print(ans)
                    sentid2ans[sid.item()] = (l, diss,dise, ln,ln_, br,l_s,l_e)



                n_iter += 1
                # writer.add_scalar('Loss/test', np.random.random(), n_iter)

            # self.scheduler.step(loss)
            log_str = f"\nEpoch {epoch}: Total Loss {total_loss}\n"
            tmp_acc, mDist, acc2, acc3, tmpAcc = evaluator.evaluate(sentid2ans)
            log_str += f"\nEpoch {epoch}: Train {tmpAcc * 100.}%\n"
            # log_str += f"\nEpoch {epoch}: Training Av. Distance {mDist}m\n"
            self.writer.add_scalar('Accuracy/train [IoU=0.5]', tmpAcc * 100., n_iter)
            # awlf.writer.close()

            if self.valid_tuple is not None:  # Do Validation
                valid_score, m_dist, acc2, acc3, tAcc = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")
                if tmpAcc > best_train:
                    best_train = tmpAcc
                if acc2 > best_acc2:
                    best_acc2 = acc2
                if acc3 > best_acc3:
                    best_acc3 = acc3
                if tAcc > best_tacc:
                    best_tacc = tAcc
                if m_dist[0] < best_mDist[0]:
                    best_mDist = m_dist

                self.writer.add_scalar('Accuracy/valid [IoU=0.5]', valid_score * 100., n_iter)
                # awlf.writer.close()
                log_str += f"Epoch {epoch}: Best Valid dist/pixel [SD] {best_mDist[0]} [{best_mDist[1]}] / {best_mDist[2]} [{best_mDist[1]}] \n" + \
                           f"Epoch {epoch}: Best Train {best_train * 100.}%\n" + \
                           f"Epoch {epoch}: Best Val3 {best_acc3 * 100.}%\n" + \
                           f"Epoch {epoch}: T-Best Val {best_tacc * 100.}%\n"

            if self.test_tuple is not None:  # Do Validation
                _, test_dist, _, _, test_acc = self.evaluate(self.test_tuple)
                print("test")

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                if test_dist[0] < best_testDist[0]:
                    best_testDist = test_dist
                log_str += f"Epoch {epoch}: Test {test_acc * 100.}%\n" + \
                        f"Epoch {epoch}: Best Test dist/pixel [SD] {best_testDist[0]} [{best_testDist[1]}] / {best_testDist[2]} [{best_testDist[1]}]\n" + \
                            f"Epoch {epoch}: Best Test {best_test_acc * 100.}%\n"
            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")
        return best_acc3, best_mDist

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
            ques_id, feats, feat_mask, boxes, names, sent, g_ds, g_de, land_,cland_, bear_ = datum_tuple[:11]   # Avoid seeing ground truth
            with torch.no_grad():
                if args.n_ent:
                    names = (names[0].squeeze(2).cuda(), \
                                  names[1].squeeze(2).cuda(), \
                                  names[2].squeeze(2).cuda())
                elif args.qa:
                    names = (names[0].cuda(), \
                                  names[1].cuda(), \
                                  names[2].cuda())
                else:
                    names = None
                feats, feat_mask, boxes = feats.cuda(),feat_mask.cuda(), boxes.cuda()
                label, aux  = self.model(feats.float(), feat_mask.float(), boxes.float(), names, sent)
                dist_s, dist_e, lnd,clnd, brng, land_start, land_end = aux

                bear_score, bear_label = brng.max(1)
                _, dist_e = dist_e.max(1)
                _, dist_s = dist_s.max(1)
                _, land_start = land_start.max(1)
                _, land_end = land_end.max(1)
                _, clnd = clnd.max(1)
                for qid,diss,dise, ln,cln, br,l_s,l_e, l in zip(ques_id,dist_s.cpu().detach().numpy(), \
                                                dist_e.cpu().detach().numpy(), \
                                                lnd.cpu().detach().numpy(), \
                                                clnd.cpu().detach().numpy(), \
                                                bear_label.cpu().detach().numpy(), \
                                                land_start.cpu().detach().numpy(), \
                                                land_end.cpu().detach().numpy(), \
                                                    label.cpu().detach().numpy()):
                    # ans = dset.label2ans[l]
                    # input(br)
                    br = dset.label2bearing[br]
                    sentid2ans[qid.item()] = (l, diss, dise, ln,cln, br, l_s, l_e)
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

        for i, (ques_id, feats, feat_mask, boxes, names, sent,dists,diste,land_,cland_, bear_,land_s,land_e, target) in enumerate(loader):
            # input(target)
            label = target
            for qid,diss,dise, ln,cln, br,l_s,l_e, l in zip(ques_id,dists.cpu().detach().numpy(), \
                                                diste.cpu().detach().numpy(), \
                                                land_.cpu().detach().numpy(), \
                                                cland_.cpu().detach(), \
                                                bear_.cpu().detach().numpy(), \
                                                land_s.cpu().detach().numpy(), \
                                                land_e.cpu().detach().numpy(), \
                                                    label.cpu().detach().numpy()):

                br = np.argmax(br)
                diss = np.argmax(diss)
                dise = np.argmax(dise)
                l_s = np.argmax(l_s)
                l_e = np.argmax(l_e)
                cln = np.argmax(cln)
                br = dset.label2bearing[br]
                sentid2ans[qid.item()] = (l, diss,dise, ln,cln, br,l_s,l_e)
        valid_score, m_dist, acc2, acc3, tAcc = evaluator.evaluate(sentid2ans)
        return acc3, m_dist

    def save(self, name, k = ''):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, f"{k}{name}.pth"))

    def load(self, path, k = ''):
        print(f"Load model from {k}{path}")
        state_dict = torch.load(f"{k}{path}.pth")
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":

    scores = []
    scores2 = []
    distances = [[],[],[],[]]
    oracle_distances = [[],[],[],[]]
    scenarios = []
    t_scores = []
    oracle_scores = []
    # for k in range(7):
    # for k in range(10):
    for k in range(0,1):
        print(f"{k} on cross")
        # args.train = f'{k}_easy_train'
        # args.valid = f'{k}_easy_val'
        # args.train = f'{k}_train'
        # args.valid = f'{k}_val'
        args.train = '440_train'
        args.valid = '55_val'
        # Build Class
        rosmi = ROSMI()
        # Load ROSMI model weights
        # Note: It is different from loading LXMERT pre-trained weights.
        if args.load is not None:
            rosmi.load(args.load)

        # Test or Train
        if args.test is not None and False:
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
            # rosmi.oracle_score(rosmi.train_tuple)
            # rosmi.oracle_score(rosmi.valid_tuple)
            # input("??")
            if rosmi.valid_tuple is not None:
                print('Splits in Valid data:', rosmi.valid_tuple.dataset.splits)
                tmpA, dis = rosmi.oracle_score(rosmi.valid_tuple)

                oracle_distances[0].append(dis[0])
                oracle_distances[1].append(dis[1])
                oracle_distances[2].append(dis[2])
                oracle_distances[3].append(dis[3])
                oracle_scores.append(tmpA)
                # input(dis[4])
                with open(f'{args.abla}_oracle_scores.json', 'w') as scores_out:
                    json.dump(oracle_scores, scores_out)

                with open(f'{args.abla}_oracle_distances.json', 'w') as scores_out:
                    json.dump(oracle_distances, scores_out)
                print("Valid Oracle: %0.2f" % (tmpA * 100))
            else:
                print("DO NOT USE VALIDATION")
            # input()
            if rosmi.test_tuple is not None:
                print('Splits in Valid data:', rosmi.test_tuple.dataset.splits)

                tmpA, dis = rosmi.oracle_score(rosmi.test_tuple)
                oracle_distances = [[],[],[],[]]
                oracle_scores = []

                oracle_distances[0].append(dis[0])
                oracle_distances[1].append(dis[1])
                oracle_distances[2].append(dis[2])
                oracle_distances[3].append(dis[3])
                oracle_scores.append(tmpA)
                with open(f'{args.abla}_t_oracle_scores.json', 'w') as scores_out:
                    json.dump(oracle_scores, scores_out)

                with open(f'{args.abla}_t_oracle_distances.json', 'w') as scores_out:
                    json.dump(oracle_distances, scores_out)
                print("Test Oracle: %0.2f" % (tmpA * 100))
            # input()
            best_tacc, best_mDist = rosmi.train(rosmi.train_tuple, rosmi.valid_tuple)

            distances[0].append(best_mDist[0])
            distances[1].append(best_mDist[1])
            distances[2].append(best_mDist[2])
            distances[3].append(best_mDist[3])
            scenarios.append(best_mDist[4])
            t_scores.append(best_tacc)
            with open(f'{args.abla}_t_scores.json', 'w') as scores_out:
                json.dump(t_scores, scores_out)

            with open(f'{args.abla}_distances.json', 'w') as scores_out:
                json.dump(distances, scores_out)

            with open(f'{args.abla}_scenarios.json', 'w') as scores_out:
                json.dump(scenarios, scores_out)
        # input("???")
    # print(f"Best scores: {scores, scores2, scores3, t_scores}")
    # print(f"Mean 6-fold accuracy 1 {sum(scores) / len(scores)}")
    # print(f"Mean 6-fold accuracy 2 {sum(scores2) / len(scores2)}")
    # print(f"Mean 6-fold accuracy 3 {sum(scores3) / len(scores3)}")
    # print(f"Mean 6-fold total accuracy {sum(t_scores) / len(t_scores)}")
