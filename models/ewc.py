import logging
import numpy as np
import copy
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from models.podnet import pod_spatial_loss
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 180
lrate = 0.1
milestones = [70, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
T = 2
lamda = 1000
fishermax = 0.0001


class EWC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.fisher = None
        self._network = IncrementalNet(args, False)

    def after_task(self, fold='final'):
        self._known_classes = self._total_classes
        if not self._eval_only and not self._get_thresh:
            self.save_checkpoint(self._save_path, fold)

    def incremental_train(self, data_manager, fold='final'):
        self.data_manager = data_manager
        self._cur_task += 1
        self._current_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._current_classes
        self._network.update_fc(self._total_classes)
        self._fold = fold

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        train_dataset = data_manager.get_dataset_Kfold(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            fold=fold,
        )
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.args["num_workers"])
        val_dataset = data_manager.get_dataset_Kfold(
            np.arange(self._known_classes, self._total_classes),
            source="val",
            fold=fold,
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.args["num_workers"])
        test_dataset = data_manager.get_dataset_wood(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._eval_only:
            self.load_checkpoint(self._save_path, fold)
        else:
            self._train(self.train_loader, self.test_loader)

            if self.fisher is None:
                self.fisher = self.getFisherDiagonal(self.train_loader)
            else:
                alpha = self._known_classes / self._total_classes
                new_finsher = self.getFisherDiagonal(self.train_loader)
                for n, p in new_finsher.items():
                    new_finsher[n][: len(self.fisher[n])] = (
                            alpha * self.fisher[n]
                            + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
                    )
                self.fisher = new_finsher
            self.mean = {
                n: p.clone().detach()
                for n, p in self._network.named_parameters()
                if p.requires_grad
            }

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
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
                self._network.parameters(),
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
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                loss = F.cross_entropy(logits, targets.long())
                if self._separate_head:
                    loss = loss * self.args["b2"]
                    logits_edl = outputs["logits_edl"]
                    loss_edl = self.edl_loss(logits_edl, targets, epoch, self._current_classes, init_epoch, self._device,
                                             **self._loss_args) * self.args["b1"]
                    loss += loss_edl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                targets = targets - self._known_classes  # 50, 51, ..., 59 ==> 0, 1, ..., 9
                inputs, targets = inputs.to(self._device), targets.to(self._device).long()
                outputs = self._network(inputs)
                logits = outputs["logits"]
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes:], targets
                )
                loss_ewc = self.compute_ewc()
                loss = loss_clf + lamda * loss_ewc

                if self._separate_head:
                    loss = loss * self.args["b2"]
                    logits_edl = outputs["logits_edl"]
                    loss_edl = self.edl_loss(logits_edl[:, self._known_classes:], targets, epoch, self._current_classes,
                                             epochs, self._device, **self._loss_args) * self.args["b1"]
                    loss += loss_edl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def compute_ewc(self):
        loss = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.module.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                        torch.sum(
                            (self.fisher[n])
                            * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                        )
                        / 2
                    )
        else:
            for n, p in self._network.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                        torch.sum(
                            (self.fisher[n])
                            * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                        )
                        / 2
                    )
        return loss

    def getFisherDiagonal(self, train_loader):
        fisher = {
            n: torch.zeros(p.shape).to(self._device)
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }
        self._network.train()
        optimizer = optim.SGD(self._network.parameters(), lr=lrate)
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs)["logits"]
            loss = torch.nn.functional.cross_entropy(logits, targets.long())
            optimizer.zero_grad()
            loss.backward()
            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher
