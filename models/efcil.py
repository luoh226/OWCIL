import os
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from models.base import BaseLearner
from utils.inc_net import EFCILNet, DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy, accuracy
from utils.autoaugment import CIFAR10Policy

EPSILON = 1e-8

init_epoch = 1  # 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 1  # 170
lrate = 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
weight_decay = 2e-4
num_workers = 1
T = 2


class EFCIL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._current_classes = None
        self._network = EFCILNet(args, False)
        self._loss_args = {
            "loss_type": args["loss_type"],
            "activate_type": args["activate_type"],
            "a1": args["a1"],
            "a2": args["a2"],
            "coef_start": args["coef_start"],
            "coef_type": args["coef_type"]
        }

    def after_task(self):
        self._known_classes = self._total_classes
        if not self._eval_only:
            self.save_checkpoint(self._save_path)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self.args['auto_aug'] and self.args['dataset'] == "cifar100":
            data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ]
        self._increments = data_manager._increments
        self._current_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._current_classes
        self._network.update_fc(self._current_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # 固定1~K-1阶段的backbone和fc layers
        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False
                # 1~K-1的分类器也固定
                for p in self._network.fcs[i].parameters():
                    p.requires_grad = False
                # 如果有额外分类头也固定
                if self._separate_head:
                    for p in self._network.fcs_cnn[i].parameters():
                        p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._eval_only:
            self._network.load_state_dict(
                torch.load("{}_{}.pkl".format(self._save_path, self._cur_task))["model_state_dict"])
            self._network.to(self._device)
        else:
            self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1:
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network

        self._network_module_ptr.convnets[-1].train()
        self._network_module_ptr.fcs[-1].train()
        if self._separate_head:
            self._network_module_ptr.fcs_cnn[-1].train()

        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()
                self._network_module_ptr.fcs[i].eval()
                if self._separate_head:
                    self._network_module_ptr.fcs_cnn[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        # print(self._network)
        # for name, param in self._network.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape)
        #     else:
        #         print('Freeze!   ', name, param.shape)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        info = ''
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses_edl = 0.0
            losses_cnn = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                # 只更新current task
                logits = outputs["logits"][-1]
                loss_edl = self.edl_loss(logits, targets, epoch, self._current_classes, init_epoch, self._device,
                                         **self._loss_args) * self.args["b1"]
                losses_edl += loss_edl.item()
                loss = loss_edl
                if self._separate_head:
                    logits_cnn = outputs["logits_cnn"][-1]
                    loss_ce = F.cross_entropy(logits_cnn, targets.long()) * self.args["b2"]
                    losses_cnn += loss_ce.item()
                    loss += loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # CIL pred
                if not self._separate_head:
                    _, preds = torch.max(logits, dim=1)
                else:
                    _, preds = torch.max(logits_cnn, dim=1)

                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_edl {:.3f}, Loss_cnn {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    (losses_edl + losses_cnn) / len(train_loader),
                    losses_edl / len(train_loader),
                    losses_cnn / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_edl {:.3f}, Loss_cnn {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    (losses_edl + losses_cnn) / len(train_loader),
                    losses_edl / len(train_loader),
                    losses_cnn / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses_edl = 0.0
            losses_cnn = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                targets = targets - self._known_classes  # 50, 51, ..., 59 ==> 0, 1, ..., 9
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                # 只更新current task
                logits = outputs["logits"][-1]
                loss_edl = self.edl_loss(logits, targets, epoch, self._current_classes, epochs, self._device,
                                         **self._loss_args) * self.args["b1"]
                losses_edl += loss_edl.item()
                loss = loss_edl
                if self._separate_head:
                    logits_cnn = outputs["logits_cnn"][-1]
                    loss_ce = F.cross_entropy(logits_cnn, targets.long()) * self.args["b2"]
                    losses_cnn += loss_ce.item()
                    loss += loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # eval train
                preds_list = []
                uncertainty_list = []
                label_offset = 0
                for j in range(len(outputs["logits"])):
                    # calculate uncertainty
                    evidence = self.get_evidence(outputs["logits"][j], self._loss_args["activate_type"])
                    alpha = evidence + 1
                    uncertainty = self._increments[j] / torch.sum(alpha, dim=1, keepdim=False)
                    uncertainty_list.append(uncertainty)
                    # get preds label
                    if not self._separate_head:
                        _, preds = torch.max(outputs["logits"][j], dim=1)
                    else:
                        _, preds = torch.max(outputs["logits_cnn"][j], dim=1)
                    preds += label_offset
                    preds_list.append(preds)
                    label_offset += self._increments[j]

                uncertaintys = torch.stack(uncertainty_list).T  # (128, 2/3/.../10)
                pred_task_id = torch.min(uncertaintys, 1)[1]
                result = torch.zeros_like(pred_task_id)
                for j in range(len(result)):
                    result[j] = preds_list[pred_task_id[j]][j]

                correct += result.eq(targets.expand_as(result)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_edl {:.3f}, Loss_cnn {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    (losses_edl + losses_cnn) / len(train_loader),
                    losses_edl / len(train_loader),
                    losses_cnn / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_edl {:.3f}, Loss_cnn {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    (losses_edl + losses_cnn) / len(train_loader),
                    losses_edl / len(train_loader),
                    losses_cnn / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)
            logits = outputs["logits"]
            if self._separate_head:
                logits_cnn = outputs["logits_cnn"]

            preds_list = []
            uncertainty_list = []
            label_offset = 0
            for j in range(len(logits)):
                # calculate uncertainty
                evidence = self.get_evidence(logits[j], self._loss_args["activate_type"])
                alpha = evidence + 1
                uncertainty = self._increments[j] / torch.sum(alpha, dim=1, keepdim=False)
                uncertainty_list.append(uncertainty)
                # get preds label
                if self._separate_head:
                    _, preds = torch.max(logits_cnn[j], 1)
                else:
                    _, preds = torch.max(logits[j], 1)
                preds += label_offset
                if self._eval_ccil:
                    preds[uncertainty >= self._ood_thresh] = -1
                preds_list.append(preds)
                label_offset += self._increments[j]

            uncertaintys = torch.stack(uncertainty_list).T  # (128, 2/3/.../10)
            pred_task_id = torch.min(uncertaintys, 1)[1]
            predicts = torch.zeros_like(pred_task_id)
            for j in range(len(predicts)):
                predicts[j] = preds_list[pred_task_id[j]][j]

            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def eval_task(self):
        # 1. EDL pred
        y_pred, y_true, task_pred, task_true = self._eval_edl(self.test_loader)
        edl_accy = self._evaluate(y_pred, y_true)

        # 2. TASK pred
        task_accy = self._evaluate(task_pred, task_true, pred_task=True)

        if self._eval_ccil:
            return edl_accy, None, edl_accy, task_accy
        else:
            # 3. CNN pred
            y_pred, y_true = self._eval_cnn(self.test_loader)
            cnn_accy = self._evaluate(y_pred, y_true)

            return cnn_accy, None, edl_accy, task_accy

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                if self._separate_head:
                    outputs = self._network(inputs)["logits_cnn"]  # [[512, 10], [512, 10]]
                else:
                    outputs = self._network(inputs)["logits"]  # [[512, 10], [512, 10]]
            outputs = torch.cat(outputs, dim=1)
            # print(outputs.shape)
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_edl(self, loader):
        self._network.eval()
        y_pred, y_true, task_pred, task_true, target2taskid = [], [], [], [], []
        for i, inc_len in enumerate(self._increments):
            if i <= self._cur_task:
                target2taskid.extend([i for _ in range(inc_len)])
            else:
                target2taskid.extend([-1 for _ in range(inc_len)])
        target2taskid = np.array(target2taskid, dtype=np.int64)
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)
            logits = outputs["logits"]
            if self._separate_head:
                logits_cnn = outputs["logits_cnn"]

            preds_list = []
            uncertainty_list = []
            label_offset = 0
            for j in range(len(logits)):
                # calculate uncertainty
                evidence = self.get_evidence(logits[j], self._loss_args["activate_type"])
                alpha = evidence + 1
                uncertainty = self._increments[j] / torch.sum(alpha, dim=1, keepdim=False)
                uncertainty_list.append(uncertainty)
                # get preds label
                if self._separate_head:
                    preds = torch.topk(logits_cnn[j], k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
                else:
                    preds = torch.topk(logits[j], k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
                preds += label_offset
                if self._eval_ccil:
                    preds[uncertainty >= self._ood_thresh, 0] = -1
                preds_list.append(preds)
                label_offset += self._increments[j]

            uncertaintys = torch.stack(uncertainty_list).T  # (128, 2/3/.../10)
            pred_task_id = torch.min(uncertaintys, 1)[1]  # [bs, 1]
            predicts = torch.zeros(pred_task_id.shape[0], self.topk, dtype=torch.int64)
            for j in range(predicts.shape[0]):
                predicts[j] = preds_list[pred_task_id[j]][j]
                if self._eval_ccil and preds_list[pred_task_id[j]][j][0] == -1:
                    pred_task_id[j] = -1

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            task_pred.append(pred_task_id.cpu().numpy())
            task_true.append(target2taskid[targets])

        return np.concatenate(y_pred), np.concatenate(y_true), np.concatenate(task_pred), np.concatenate(
            task_true)  # [N, topk]

    def _evaluate(self, y_pred, y_true, pred_task=False):
        ret = {}
        if pred_task:
            grouped = accuracy(y_pred, y_true, self._cur_task, 1)
            ret["grouped"] = grouped
            ret["top1"] = grouped["total"]
        else:
            grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
            ret["grouped"] = grouped
            ret["top1"] = grouped["total"]
            ret["top{}".format(self.topk)] = np.around(
                (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
                decimals=2,
            )
        return ret

