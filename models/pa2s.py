import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, FOSTERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.augment import *

EPSILON = 1e-8


class PASS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
        if not self._eval_only:
            self._protos = []
        self._radius = 0
        self._radiuses = []

    def after_task(self, fold='final'):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network, "module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        if not self._eval_only and not self._get_thresh:
            self.save_checkpoint(self._save_path, fold)

    def incremental_train(self, data_manager, fold='final'):
        self.data_manager = data_manager
        self._cur_task += 1
        self._current_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._current_classes
        self._network.update_fc(self._total_classes * 4, edl_nb_classes=self._total_classes)
        self._network_module_ptr = self._network
        self._fold = fold

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset_Kfold(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            fold=fold,
        )
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"])
        val_dataset = data_manager.get_dataset_Kfold(
            np.arange(self._known_classes, self._total_classes),
            source="val",
            fold=fold,
        )
        self.val_loader = DataLoader(val_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
        test_dataset = data_manager.get_dataset_wood(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._eval_only:
            self.load_checkpoint(self._save_path, fold)
        else:
            self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        self._epoch_num = self.args["epochs"]
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["lr"],
                                     weight_decay=self.args["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"],
                                                    gamma=self.args["gamma"])
        self._train_function(train_loader, test_loader, optimizer, scheduler)
        self._build_protos()

    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                idx_dataset = self.data_manager.get_dataset_Kfold(np.arange(class_idx, class_idx + 1),
                                                                  source='train', fold=self._fold, mode='test')
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)
                cov = np.cov(vectors.T)
                self._radiuses.append(np.trace(cov) / vectors.shape[1])
            self._radius = np.sqrt(np.mean(self._radiuses))

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        info = ''
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            losses_clf, losses_fkd, losses_proto = 0., 0., 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                b, c, h, w = inputs.shape
                inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                inputs = inputs.view(-1, c, h, w)
                tmp = [targets * 4 + k for k in range(4)]
                targets = torch.stack(tmp, 1).view(-1)
                outputs, loss_clf, loss_fkd, loss_proto = self._compute_pass_loss(inputs, targets.long())
                loss = loss_clf + loss_fkd + loss_proto

                if self._separate_head:
                    loss = loss * self.args["b2"]
                    logits_edl = outputs["logits_edl"]
                    loss_edl = self.edl_loss(logits_edl[::4, self._known_classes:], targets[::4] / 4 - self._known_classes,
                                             epoch, self._current_classes, self._epoch_num, self._device,
                                             **self._loss_args) * self.args["b1"]
                    loss += loss_edl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_fkd += loss_fkd.item()
                losses_proto += loss_proto.item()
                _, preds = torch.max(outputs['logits'], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct) * 100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self._epoch_num, losses / len(train_loader),
                                    losses_clf / len(train_loader), losses_fkd / len(train_loader),
                                    losses_proto / len(train_loader), train_acc)
            else:

                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self._epoch_num, losses / len(train_loader),
                                    losses_clf / len(train_loader), losses_fkd / len(train_loader),
                                    losses_proto / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def _compute_pass_loss(self, inputs, targets):
        outputs = self._network(inputs)
        logits = outputs["logits"]
        loss_clf = F.cross_entropy(logits / self.args["temp"], targets)

        if self._cur_task == 0:
            return outputs, loss_clf, torch.tensor(0.), torch.tensor(0.)

        features = self._network_module_ptr.extract_vector(inputs)
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        loss_fkd = self.args["lambda_fkd"] * torch.dist(features, features_old, 2)

        # index = np.random.choice(range(self._known_classes),size=self.args["batch_size"],replace=True)

        index = np.random.choice(range(self._known_classes), size=self.args["batch_size"] * int(
            self._known_classes / (self._total_classes - self._known_classes)), replace=True)
        # print(index)
        # print(np.concatenate(self._protos))
        proto_features = np.array(self._protos)[index]
        # print(proto_features)
        proto_targets = 4 * index
        proto_features = proto_features + np.random.normal(0, 1, proto_features.shape) * self._radius
        proto_features = torch.from_numpy(proto_features).float().to(self._device, non_blocking=True)
        proto_targets = torch.from_numpy(proto_targets).to(self._device, non_blocking=True)

        proto_logits = self._network_module_ptr.fc(proto_features)["logits"]
        loss_proto = self.args["lambda_proto"] * F.cross_entropy(proto_logits / self.args["temp"], proto_targets.long())

        return outputs, loss_clf, loss_fkd, loss_proto

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader, val=False):
        self._network.eval()
        y_pred, y_true, y_prob = [], [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # if isinstance(loader.dataset, FeatureDataset) or isinstance(loader.dataset, OodFeatureDataset):
                #     outputs = self._network.fc(inputs)
                # else:
                outputs = self._network(inputs)
            prob = torch.softmax(outputs["logits"][:, ::4], dim=1)
            max_prob, predicts = torch.topk(prob, k=self.topk, dim=1, largest=True, sorted=True)  # [bs, topk]

            # OOD method
            if self._ood_method != "None":
                if "MSP" == self._ood_method:
                    ood_prob = 1 - max_prob[:, 0]
                elif "ENERGY" == self._ood_method:
                    ood_prob = -torch.logsumexp(outputs["logits"][:, ::4], dim=1)
                elif "EDL" == self._ood_method:
                    evidence = self.get_evidence(outputs["logits_edl"], self._loss_args["activate_type"])
                    alpha = evidence + 1
                    ood_prob = self._total_classes / torch.sum(alpha, dim=1, keepdim=False)
                elif "MSP_CE" == self._ood_method:
                    sfprob = max_prob[:, 0]
                    task_entropy = self.cal_task_entropy(prob.clone(), predicts[:, 0].clone())
                    ood_prob = -1.0 * sfprob / task_entropy

                prob = torch.cat((prob, ood_prob.view(-1, 1)), dim=-1)
                # predicts[ood_prob >= self._ood_thresh, 0] = -1

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            y_prob.append(prob.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)

        if val:
            return y_pred[:, 0], y_true, y_prob

        multi_th_pred = []
        if self._ood_method == "None":
            multi_th_pred.append(y_pred)
        else:
            for th in self._ood_thresh:
                tmp_y_pred = y_pred.copy()
                tmp_y_pred[y_prob[:, -1] >= th, 0] = -1
                multi_th_pred.append(tmp_y_pred)

        return multi_th_pred, y_true, y_prob  # [N, topk]
