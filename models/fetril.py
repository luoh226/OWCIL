'''

results on CIFAR-100: 
               
           |   Reported  Resnet18        |  Reproduced Resnet32 
Protocols  |  Reported FC | Reported SVM |  Reproduced FC | Reproduced SVM |  

T = 5      |   64.7       |  66.3        |  65.775        | 65.375         |

T = 10     |   63.4       |  65.2        |  64.91         | 65.10          |

T = 60     |   50.8       |  59.8        |  62.09         | 61.72          |

'''
import warnings
import copy
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
from sklearn.svm import LinearSVC
from utils.augment import *
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy, ImageNetPolicy
from utils.ops import Cutout

EPSILON = 1e-8


class FeTrIL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
        self._means = []

    def after_task(self, fold='final'):
        self._known_classes = self._total_classes
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

        if self._cur_task > 0:
            for p in self._network.convnet.parameters():
                p.requires_grad = False

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
            if self._cur_task == 0:
                self.load_checkpoint(self._save_path, fold)
                self._compute_means()
                self._build_feature_set()
            else:
                self._compute_means()
                self._compute_relations()
                self._build_feature_set()
                self.load_checkpoint(self._save_path, fold)
        else:
            self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            self._epoch_num = self.args["init_epochs"]
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
            )), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"])
            self._train_function(train_loader, test_loader, optimizer, scheduler)
            self._compute_means()
            self._build_feature_set()
        else:
            self._epoch_num = self.args["epochs"]
            self._compute_means()
            self._compute_relations()
            self._build_feature_set()

            train_loader = DataLoader(self._feature_trainset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
            optimizer = optim.SGD(self._network_module_ptr.fc.parameters(),momentum=0.9,lr=self.args["lr"],weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max = self.args["epochs"])
            
            self._train_function(train_loader, test_loader, optimizer, scheduler)
        # self._train_svm(self._feature_trainset,self._feature_testset)

        
    def _compute_means(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                idx_dataset = self.data_manager.get_dataset_Kfold(np.arange(class_idx, class_idx + 1),
                                                                  source='train', fold=self._fold, mode='test')
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._means.append(class_mean)
                # self._covss.append(np.cov(vectors.T))

    def _compute_relations(self):
        old_means = np.array(self._means[:self._known_classes])
        new_means = np.array(self._means[self._known_classes:])
        self._relations=np.argmax((old_means/np.linalg.norm(old_means,axis=1)[:,None])@(new_means/np.linalg.norm(new_means,axis=1)[:,None]).T,axis=1)+self._known_classes
    def _build_feature_set(self):
        # train
        self.vectors_train = []
        self.labels_train = []
        for class_idx in range(self._known_classes, self._total_classes):
            idx_dataset = self.data_manager.get_dataset_Kfold(np.arange(class_idx, class_idx + 1),
                                                              source='train', fold=self._fold, mode='test')
            idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
            vectors, _ = self._extract_vectors(idx_loader)
            self.vectors_train.append(vectors)
            self.labels_train.append([class_idx]*len(vectors))
        for class_idx in range(0,self._known_classes):
            new_idx = self._relations[class_idx]
            self.vectors_train.append(self.vectors_train[new_idx-self._known_classes]-self._means[new_idx]+self._means[class_idx])
            self.labels_train.append([class_idx]*len(self.vectors_train[-1]))

        self.vectors_train = np.concatenate(self.vectors_train)
        self.labels_train = np.concatenate(self.labels_train)
        self._feature_trainset = FeatureDataset(self.vectors_train, self.labels_train)

        # val
        self.vectors_val = []
        self.labels_val = []
        if self._fold == 'final':
            self.vectors_val = np.array([])
            self.labels_val = np.array([])
        else:
            for class_idx in range(self._known_classes, self._total_classes):
                idx_dataset = self.data_manager.get_dataset_Kfold(np.arange(class_idx, class_idx + 1),
                                                                  source='val', fold=self._fold)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
                vectors, _ = self._extract_vectors(idx_loader)
                self.vectors_val.append(vectors)
                self.labels_val.append([class_idx]*len(vectors))
            self.vectors_val = np.concatenate(self.vectors_val)
            self.labels_val = np.concatenate(self.labels_val)
        self._feature_valset = FeatureDataset(self.vectors_val,self.labels_val)

        # test
        self.vectors_test = []
        self.labels_test = []
        idx_dataset = self.data_manager.get_dataset_wood(np.arange(0, self._total_classes), source="test", mode="test")
        idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
        vectors, targets = self._extract_vectors(idx_loader)
        self.vectors_test = np.array(vectors)
        self.labels_test = np.array(targets)
        self._feature_testset = FeatureDataset(self.vectors_test, self.labels_test)

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        info = ''
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            if self._cur_task == 0:
                self._network.train()
            else:
                self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                if self._cur_task == 0:
                    outputs = self._network(inputs)
                    logits = outputs['logits']
                else:
                    logits = self._network_module_ptr.fc(inputs)['logits']
                loss = F.cross_entropy(logits, targets.long())

                # if self._separate_head:
                #     loss = loss * self.args["b2"]
                #     if self._cur_task == 0:
                #         logits_edl = outputs['logits_edl']
                #     else:
                #         logits_edl = self._network_module_ptr.fc_edl(inputs)["logits"]
                #     loss_edl = self.edl_loss(logits_edl, targets,
                #                              epoch, self._total_classes, self._epoch_num, self._device,
                #                              **self._loss_args) * self.args["b1"]
                #     loss += loss_edl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct) * 100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self._epoch_num, losses / len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self._epoch_num, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
        logging.info(info)

    # def get_ood_loader(self, loader, aug_type):
    #     id_set = copy.deepcopy(loader.dataset)
    #     if aug_type == 'AID_AVG':
    #         ood_set = AvgOfPair(loader.dataset)
    #     # elif type == 'AID_GEO':
    #     #     return DataLoader(GeomMeanOfPair(loader.dataset), batch_size=128, shuffle=False)
    #     elif 'AID_GEN' in aug_type:
    #         if self._cur_task == 0:
    #             ood_set = AvgOfPair(loader.dataset)
    #         else:
    #             id_set, ood_set = self._build_ood_feature_set(id_set, aug_type)
    #
    #     return DataLoader(id_set, batch_size=128, shuffle=False), DataLoader(ood_set, batch_size=128, shuffle=False)

    def _get_ood_thresh(self, loader):
        y_preds, y_true, id_y_prob = self._eval_svm(self._feature_trainset, self._feature_valset, loader, val=True)
        id_score = id_y_prob[:, -1].copy()

        recall_level = [i / 100.0 for i in range(0, 101)]
        multi_ood_thresh = self.get_measures(-id_score, np.array([]), recall_level, only_thresh=True)
        for i, thresh in enumerate(multi_ood_thresh):
            id_recall = np.sum(id_score < thresh) / len(id_score)
            if abs(id_recall - recall_level[i]) > 1e-2:
                warnings.warn(
                    'wrong ood threshold = {}, id_recall = {} < {}'.format(thresh, id_recall, recall_level[i]))

        if self._ood_thresh_type == 'TPR95':
            ood_thresh = [multi_ood_thresh[95]]
        elif self._ood_thresh_type == 'ALL':
            ood_thresh = multi_ood_thresh
        elif self._ood_thresh_type == 'MID':
            misclassified_mask = y_preds != y_true
            cunt = np.sum(misclassified_mask)
            self._ood_thresh = multi_ood_thresh
            y_preds, y_true, id_y_prob = self._eval_svm(self._feature_trainset, self._feature_valset, loader)
            y_true[misclassified_mask] = -1
            # multi_acc = []
            multi_de = []
            for y_pred in y_preds:
                # acc = self._balanced_accuracy(y_pred[:, 0], y_true)
                # multi_acc.append(acc)
                de = self._evaluate_DE(y_pred[:, 0], y_true)
                multi_de.append(de)
            # best_thresh_idx = np.argmax(multi_acc)
            best_thresh_idx = np.argmin(multi_de)
            ood_thresh = [multi_ood_thresh[best_thresh_idx]]
        # elif 'AID' in self._ood_thresh_type:
        #     id_loader, ood_loader = self.get_ood_loader(copy.deepcopy(loader), self._ood_thresh_type)
        #     self._ood_thresh = multi_ood_thresh
        #     with torch.no_grad():
        #         if isinstance(id_loader.dataset, FeatureDataset):
        #             id_feature_set = id_loader.dataset
        #         else:
        #             id_feature_set = FeatureDataset(*self._extract_vectors(id_loader))
        #         if isinstance(ood_loader.dataset, OodFeatureDataset):
        #             ood_feature_set = ood_loader.dataset
        #         else:
        #             ood_feature_set = FeatureDataset(*self._extract_vectors(ood_loader))
        #     id_y_preds, id_y_true, id_y_prob = self._eval_svm(self._feature_trainset, id_feature_set, id_loader)
        #     ood_y_preds, ood_y_true, ood_y_prob = self._eval_svm(self._feature_trainset, ood_feature_set, ood_loader)
        #     y_true = np.concatenate((id_y_true, ood_y_true), axis=0)
        #     # multi_acc = []
        #     multi_de = []
        #     for i in range(len(id_y_preds)):
        #         y_pred = np.concatenate((id_y_preds[i][:, 0], ood_y_preds[i][:, 0]), axis=0)
        #         de = self._evaluate_DE(y_pred, y_true)
        #         multi_de.append(de)
        #         # acc = self._balanced_accuracy(y_pred, y_true)
        #         # multi_acc.append(acc)
        #     # best_thresh_idx = np.argmax(multi_acc)
        #     best_thresh_idx = np.argmin(multi_de)
        #     ood_thresh = [multi_ood_thresh[best_thresh_idx]]

        return ood_thresh

    def _eval_svm(self, train_set, test_set, test_loader, val=False):
        self._network.eval()
        train_features = train_set.features.numpy()
        train_labels = train_set.labels.numpy()
        test_features = test_set.features.numpy()
        # test_labels = test_set.labels.numpy()
        train_features = train_features / np.linalg.norm(train_features, axis=1)[:, None]
        test_features = test_features / np.linalg.norm(test_features, axis=1)[:, None]
        svm_classifier = LinearSVC(random_state=42)
        svm_classifier.fit(train_features, train_labels)
        # svm inference
        scores = svm_classifier.decision_function(test_features)
        # prob = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))  # 最值归一化
        # max_prob = np.amax(prob, axis=1)
        y_pred = np.argsort(-scores, axis=1)[:, :self.topk]

        # cnn inference
        y_true, y_prob = [], []
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # if isinstance(test_loader.dataset, FeatureDataset) or isinstance(test_loader.dataset, OodFeatureDataset):
                #     outputs = self._network.fc(inputs)
                # else:
                outputs = self._network(inputs)
            prob = torch.softmax(outputs["logits"], dim=1)
            max_prob, predicts = torch.topk(prob, k=self.topk, dim=1, largest=True, sorted=True)  # [bs, topk]

            # OOD method
            if self._ood_method != "None":
                if "MSP" == self._ood_method:
                    ood_prob = 1 - max_prob[:, 0]
                elif "ENERGY" == self._ood_method:
                    ood_prob = -torch.logsumexp(outputs["logits"], dim=1)
                elif "EDL" == self._ood_method:
                    evidence = self.get_evidence(outputs["logits_edl"], self._loss_args["activate_type"])
                    alpha = evidence + 1
                    ood_prob = self._total_classes / torch.sum(alpha, dim=1, keepdim=False)
                elif "MSP_CE" == self._ood_method:
                    sfprob = max_prob[:, 0]
                    task_entropy = self.cal_task_entropy(prob.clone(), predicts[:, 0].clone())
                    ood_prob = -1.0 * sfprob / task_entropy

                prob = torch.cat((prob, ood_prob.view(-1, 1)), dim=-1)

            y_true.append(targets.cpu().numpy())
            y_prob.append(prob.cpu().numpy())

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

    def eval_task(self):
        y_preds, y_true, y_prob = self._eval_svm(self._feature_trainset, self._feature_testset, self.test_loader)

        multi_acc = []
        for y_pred in y_preds:
            # acc = self._balanced_accuracy(y_pred[:, 0], y_true)
            acc = self._accuracy(y_pred[:, 0], y_true)
            multi_acc.append(acc)

        id_num, _ = self.get_id_num()
        y_pred = y_preds[-1]
        # ACC Metrics
        # acc = self._balanced_accuracy(y_pred[:, 0], y_true)
        # id_acc = self._balanced_accuracy(y_pred[:id_num, 0], y_true[:id_num])
        acc = self._accuracy(y_pred[:, 0], y_true)
        id_acc = self._accuracy(y_pred[:id_num, 0], y_true[:id_num])

        # OOD Metrics
        if id_num != len(y_true):
            ood_score = round(self._eval_OOD(y_pred, y_true), 4)
        else:
            ood_score = 0.0

        # forgetting, plasticity
        forget_ratio, f1_score, overall_score, task_level_recall, task_level_precision = self._eval_cls(y_pred, y_true)

        return (acc, id_acc, ood_score, forget_ratio, f1_score, overall_score,
                task_level_recall, task_level_precision, multi_acc)

