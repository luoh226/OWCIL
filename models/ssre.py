import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, FOSTERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.augment import *
from utils.autoaugment import CIFAR10Policy,ImageNetPolicy
from utils.ops import Cutout
from torchvision import datasets, transforms

EPSILON = 1e-8


class SSRE(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
        if not self._eval_only:
            self._protos = []

    def after_task(self, fold='final'):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        if not self._eval_only and not self._get_thresh:
            self.save_checkpoint(self._save_path, fold)

    def incremental_train(self, data_manager, fold='final'):
        self.data_manager = data_manager
        self._cur_task += 1
        self._current_classes = self.data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._current_classes
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        self._fold = fold

        if not self._eval_only and not self._get_thresh:
            logging.info("Model Expansion!")
            self._network_expansion()

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))

        if self._cur_task == 0:
            batch_size = 64
        else:
            batch_size = self.args["batch_size"]

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

        if not self._eval_only and not self._get_thresh:
            logging.info("Model Compression!")
            self._network_compression()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        self._epoch_num = self.args["epochs"]
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters(
        )), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"])
        self._train_function(train_loader, test_loader, optimizer, scheduler)
        self._build_protos()

    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                idx_dataset = self.data_manager.get_dataset_Kfold(np.arange(class_idx, class_idx+1),
                                                                  source='train', fold=self._fold, mode='test')
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)

    def train(self):
        if self._cur_task > 0:
            self._network.eval()
            return
        self._network.train()

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        info = ''
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            losses_clf, losses_fkd, losses_proto = 0., 0., 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                outputs, loss_clf, loss_fkd, loss_proto = self._compute_ssre_loss(inputs, targets.long())
                loss = loss_clf + loss_fkd + loss_proto

                if self._separate_head:
                    loss = loss * self.args["b2"]
                    logits_edl = outputs["logits_edl"]
                    loss_edl = self.edl_loss(logits_edl[:, self._known_classes:], targets - self._known_classes,
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
                _, preds = torch.max(outputs["logits"], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_clf/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_clf/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def _compute_ssre_loss(self,inputs, targets):
        if self._cur_task == 0:
            outputs = self._network(inputs)
            logits = outputs["logits"]
            loss_clf = F.cross_entropy(logits/self.args["temp"], targets)
            return outputs, loss_clf, torch.tensor(0.), torch.tensor(0.)

        features = self._network_module_ptr.extract_vector(inputs) # N D

        with torch.no_grad():
            features_old = self.old_network_module_ptr.extract_vector(inputs)

        protos = torch.from_numpy(np.array(self._protos)).to(self._device) # C D
        with torch.no_grad():
            weights = F.normalize(features,p=2,dim=1,eps=1e-12) @ F.normalize(protos,p=2,dim=1,eps=1e-12).T
            weights = torch.max(weights,dim=1)[0]
            # mask = weights > self.args["threshold"]
            mask = weights

        outputs = self._network(inputs)
        logits = outputs["logits"]
        loss_clf = F.cross_entropy(logits/self.args["temp"],targets,reduction="none")
        # loss_clf = torch.mean(loss_clf * ~mask)
        loss_clf =  torch.mean(loss_clf * (1-mask))

        loss_fkd = torch.norm(features - features_old, p=2, dim=1)
        loss_fkd = self.args["lambda_fkd"] * torch.sum(loss_fkd * mask)

        index = np.random.choice(range(self._known_classes),size=self.args["batch_size"],replace=True)

        proto_features = np.array(self._protos)[index]
        proto_targets = index
        proto_features = proto_features
        proto_features = torch.from_numpy(proto_features).float().to(self._device,non_blocking=True)
        proto_targets = torch.from_numpy(proto_targets).to(self._device,non_blocking=True)


        proto_logits = self._network_module_ptr.fc(proto_features)["logits"]
        loss_proto = self.args["lambda_proto"] * F.cross_entropy(proto_logits/self.args["temp"], proto_targets.long())
        return outputs, loss_clf, loss_fkd, loss_proto

    def _network_expansion(self):
        if self._cur_task > 0:
            for p in self._network.convnet.parameters():
                p.requires_grad = True
            for k, v in self._network.convnet.named_parameters():
                if 'adapter' not in k:
                    v.requires_grad = False
        # self._network.convnet.re_init_params() # do not use!
        self._network.convnet.switch("parallel_adapters")

    def _network_compression(self):

        model_dict = self._network.state_dict()
        for k, v in model_dict.items():
            if 'adapter' in k:
                k_conv3 = k.replace('adapter', 'conv')
                if 'weight' in k:
                    model_dict[k_conv3] = model_dict[k_conv3] + F.pad(v, [1, 1, 1, 1], 'constant', 0)
                    model_dict[k] = torch.zeros_like(v)
                elif 'bias' in k:
                    model_dict[k_conv3] = model_dict[k_conv3] + v
                    model_dict[k] = torch.zeros_like(v)
                else:
                    assert 0
        self._network.load_state_dict(model_dict)
        self._network.convnet.switch("normal")
