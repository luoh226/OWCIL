import copy
import logging
import warnings

import numpy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from utils.augment import *
from scipy.spatial.distance import cdist
import os
from sklearn.metrics import precision_recall_curve, auc
import sklearn.metrics as sk
import matplotlib.pyplot as plt

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

        self._separate_head = args["separate_head"]

        self._save_path = args["save_path"]
        self._ood_method = args["ood_method"]
        self._ood_thresh = [1.0]
        self._ood_thresh_type = args["ood_thresh_type"]
        self._eval_only = args.get("eval_only", False)
        self._get_thresh = args["get_thresh"]
        self._cm = []
        self._cm_ood = []
        self._precisions = []
        self._recalls = []
        # self._means = []
        # self._covss = []

        self._loss_args = {
            "loss_type": args["loss_type"],
            "activate_type": args["activate_type"],
            "a1": args["a1"],
            "a2": args["a2"],
            "coef_start": args["coef_start"],
            "coef_type": args["coef_type"]
        }

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def save_checkpoint(self, filename, fold):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}/trained_model/{}_{}.pkl".format(filename, self._cur_task, fold))

    def load_checkpoint(self, filename, fold):
        trained_model = torch.load("{}/trained_model/{}_{}.pkl".format(filename, self._cur_task, fold))
        self._network.load_state_dict(trained_model["model_state_dict"])
        self._network.to(self._device)

        if not self._get_thresh and fold == 'final' and self._ood_method != "None":
            with open("{}/{}/ood_thresh_{}.txt".format(filename, self._ood_method, self._ood_thresh_type)) as f:
                ood_threshs = f.read().split('\n')
                ood_threshs = ood_threshs[self._cur_task]
                self._ood_thresh = [float(ood_threshs)]

    def save_ood_thresh(self, filename):
        if self._ood_method != "None":
            with open("{}/{}/ood_thresh_{}.txt".format(filename, self._ood_method, self._ood_thresh_type), 'a') as f:
                for t in self._ood_thresh:
                    f.write(str(t) + ' ')
                f.write('\n')

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def _balanced_accuracy(self, y_pred, y_true):
        id_acc_list = []
        _, task_sampels = self.get_id_num()
        for i in range(self._cur_task + 1):
            id_acc_list.append(np.around((y_pred[i*task_sampels:(i+1)*task_sampels] == y_true[i*task_sampels:(i+1)*task_sampels]).sum() / task_sampels, decimals=4))

        ood_mask = y_true == -1
        if np.sum(ood_mask) > 0:
            ood_acc = np.around((y_pred[ood_mask] == y_true[ood_mask]).sum() / len(y_true[ood_mask]), decimals=4)
            id_acc_list.append(ood_acc)

        id_acc_list = np.array(id_acc_list)
        balanced_accuracy = np.sum(id_acc_list) / len(id_acc_list)

        return balanced_accuracy

    def _accuracy(self, y_pred, y_true):
        acc = np.around((y_pred == y_true).sum() / len(y_true), decimals=4)
        return acc

    def _evaluate_DE(self, y_pred, y_true):
        pred_ood = y_pred == -1
        pred_id = ~pred_ood
        true_ood = y_true == -1
        true_id = ~true_ood

        id_error = true_id & pred_ood
        ood_error = true_ood & pred_id

        assert np.sum(true_id) > 0
        assert np.sum(true_ood) > 0

        id_error_ratio = np.sum(id_error) / np.sum(true_id)
        ood_error_ratio = np.sum(ood_error) / np.sum(true_ood)

        return 0.5 * id_error_ratio + 0.5 * ood_error_ratio

    def drwa_pr_curve(self, recall, precision, AP):
        plt.figure()
        plt.step(recall, precision, where='post')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Average precision score, macro-averaged over all classes: AP={0:0.3f}'.format(AP))
        plt.show()

    def _evaluate_mAP(self, y_prob, y_true):
        y_label = np.unique(y_true).tolist()
        AP_dict = {}

        for class_id in y_label:
            binary_y_true = y_true.copy()
            binary_y_true[y_true == class_id] = 1
            binary_y_true[~(y_true == class_id)] = 0
            y_pred = y_prob[:, class_id].copy()

            precision, recall, thresholds = precision_recall_curve(binary_y_true, y_pred)
            AP = auc(recall, precision)
            AP_dict[str(class_id)] = AP

            # self.drwa_pr_curve(recall, precision, AP)

        mAPk = 0.0
        for k, v in AP_dict.items():
            if k != '-1':
                mAPk += v
            # print("class_id: {}, AP: {}".format(k, v))
        mAPk = mAPk / ((len(y_label) - 1) if '-1' in AP_dict else len(y_label))
        if '-1' in AP_dict:
            mAPu = AP_dict['-1']
        else:
            mAPu = mAPk

        amAP = (mAPk + mAPu) / 2
        gmAP = (mAPk * mAPu) ** 0.5

        mAP_dict = {"mAPk": mAPk, "mAPu": mAPu, "amAP": amAP, "gmAP": gmAP}
        return mAP_dict

    def get_measures(self, _pos, _neg, recall_level, only_thresh=False):
        pos = np.array(_pos[:]).reshape((-1, 1))
        neg = np.array(_neg[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1

        if only_thresh:
            thresh = fpr_and_thresh_at_recall(labels[:len(pos)], examples[:len(pos)], recall_level,
                                              only_thresh=only_thresh)
            return -thresh

        auroc = sk.roc_auc_score(labels, examples)
        aupr = sk.average_precision_score(labels, examples)
        fpr = fpr_and_thresh_at_recall(labels, examples, recall_level)
        return auroc, aupr, fpr

    # def _compute_means(self):
    #     with torch.no_grad():
    #         for class_idx in range(self._known_classes, self._total_classes):
    #             idx_dataset = self.data_manager.get_dataset_Kfold(np.arange(class_idx, class_idx + 1),
    #                                                               source='train', fold=self._fold, mode='test')
    #             idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False)
    #             vectors, _ = self._extract_vectors(idx_loader)
    #             class_mean = np.mean(vectors, axis=0)
    #             self._means.append(class_mean)
    #             self._covss.append(np.cov(vectors.T))

    # def _compute_relations(self):
    #     old_means = np.array(self._means[:self._known_classes])
    #     new_means = np.array(self._means[self._known_classes:])
    #     self._relations = np.argmax((old_means / np.linalg.norm(old_means, axis=1)[:, None]) @ (
    #                 new_means / np.linalg.norm(new_means, axis=1)[:, None]).T, axis=1) + self._known_classes

    # def _build_ood_feature_set(self, id_set, aug_type):
    #     vectors_new = []
    #     label_new = []
    #     vectors_old = []
    #     label_old = []
    #     vectors_ood = []
    #     # get new feature
    #     with torch.no_grad():
    #         for class_idx in range(self._known_classes, self._total_classes):
    #             idx_dataset = self.data_manager.get_dataset_Kfold(np.arange(class_idx, class_idx + 1),
    #                                                               source='val', fold=self._fold, mode='test')
    #             idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False)
    #             vectors, _ = self._extract_vectors(idx_loader)
    #             vectors_new.append(vectors)
    #             label_new.append([class_idx] * len(vectors))
    #     # generate old feature
    #     for class_idx in range(0, self._known_classes):
    #         new_idx = self._relations[class_idx]
    #         vectors_old.append(vectors_new[new_idx - self._known_classes] - self._means[new_idx] + self._means[class_idx])
    #         label_old.append([class_idx] * len(vectors))
    #     vectors_id = vectors_old + vectors_new
    #     label_id = label_old + label_new
    #     for i in range(len(vectors_id)):
    #         vectors_id[i] = torch.from_numpy(vectors_id[i])
    #         label_id[i] = torch.from_numpy(numpy.array(label_id[i]))
    #     # generate ood feature
    #     if aug_type == 'AID_GEN_1' or aug_type == 'AID_GEN_2':
    #         generate_num_classes = self._known_classes  # only generate old classes' ood
    #     elif aug_type == 'AID_GEN_3' or aug_type == 'AID_GEN_4':
    #         generate_num_classes = self._total_classes  # generate id classes' ood
    #     # for index in range(generate_num_classes):
    #     #     if index == 0:
    #     #         X = vectors_id[index] - vectors_id[index].mean(0)
    #     #         mean_feature_id = vectors_id[index].mean(0).view(1, -1)
    #     #     else:
    #     #         X = torch.cat((X, vectors_id[index] - vectors_id[index].mean(0)), 0)
    #     #         mean_feature_id = torch.cat((mean_feature_id, vectors_id[index].mean(0).view(1, -1)), 0)
    #     # covariance_matrix = torch.from_numpy(np.cov(X.t().numpy())).to(torch.float32)
    #     if aug_type == 'AID_GEN_1' or aug_type == 'AID_GEN_3':
    #         id_num = len(id_set)
    #     elif aug_type == 'AID_GEN_2' or aug_type == 'AID_GEN_4':
    #         id_num = 0
    #         for v in vectors_id:
    #             id_num += v.shape[0]
    #     samples_num = int(id_num / generate_num_classes)
    #     for index in range(generate_num_classes):
    #         checker = torch.distributions.constraints._PositiveDefinite()
    #         covariance_matrix = self._covss[index]
    #         avg_eye = np.average(np.diagonal(covariance_matrix))
    #         for i in range(covariance_matrix.shape[0]):
    #                 if covariance_matrix[i][i] <= 0:
    #                     covariance_matrix[i][i] = avg_eye
    #         covariance_matrix = torch.from_numpy(covariance_matrix)
    #         if not checker.check(covariance_matrix):
    #             covariance_matrix = torch.eye(self._covss[index].shape[0]) * avg_eye
    #         new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
    #             torch.from_numpy(self._means[index]).to(torch.float32),
    #             covariance_matrix=covariance_matrix.to(torch.float32))
    #         ood_features = new_dis.rsample((100000,))
    #         prob_density = new_dis.log_prob(ood_features)
    #         # tmp = prob_density.numpy()
    #         # keep the data in the low density area.
    #         cur_samples, index_prob = torch.topk(- prob_density, samples_num)
    #         if index == 0:
    #             vectors_ood = ood_features[index_prob]
    #         else:
    #             vectors_ood = torch.cat((vectors_ood, ood_features[index_prob]), 0)
    #
    #     ood_feature_set = OodFeatureDataset(vectors_ood)
    #     if aug_type == 'AID_GEN_1' or aug_type == 'AID_GEN_3':
    #         # print(len(id_set), len(vectors_ood))
    #         return id_set, ood_feature_set
    #     else:
    #         # print(len(vectors_id), len(vectors_ood))
    #         id_feature_set = FeatureDataset(torch.cat(vectors_id), torch.cat(label_id))
    #         return id_feature_set, ood_feature_set

    # def get_ood_loader(self, loader, aug_type):
    #     id_set = copy.deepcopy(loader.dataset)
    #     if aug_type == 'AID_AVG':
    #         ood_set = AvgOfPair(loader.dataset)
    #     # elif type == 'AID_GEO':
    #     #     return DataLoader(GeomMeanOfPair(loader.dataset), batch_size=128, shuffle=False)
    #     elif 'AID_GEN' in aug_type:
    #         if self._cur_task == 0:
    #             self._compute_means()
    #             ood_set = AvgOfPair(loader.dataset)
    #         else:
    #             self._compute_means()
    #             self._compute_relations()
    #             id_set, ood_set = self._build_ood_feature_set(id_set, aug_type)
    #
    #     return DataLoader(id_set, batch_size=128, shuffle=False), DataLoader(ood_set, batch_size=128, shuffle=False)

    def _get_ood_thresh(self, loader):
        y_preds, y_true, id_y_prob = self._eval_cnn(loader, val=True)
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
            y_preds, y_true, id_y_prob = self._eval_cnn(loader)
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
        #     id_y_preds, id_y_true, id_y_prob = self._eval_cnn(id_loader)
        #     ood_y_preds, ood_y_true, ood_y_prob = self._eval_cnn(ood_loader)
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

    def _eval_OOD_old(self, y_prob, y_true):
        id_mask = y_true != -1
        ood_score = y_prob[:, -1].copy()
        id_ood_score = ood_score[id_mask]
        ood_ood_score = ood_score[~id_mask]
        auroc, aupr, fpr = self.get_measures(-id_ood_score, -ood_ood_score, [0.95])

        # validate thresh that make ID Recall=0.95
        # pred_id = id_ood_score <= thresh
        # id_recall = np.sum(pred_id) / len(id_ood_score)

        return auroc, aupr, fpr

    def P_R_CM_per_class(self, y_pred, y_true):
        y_label = np.unique(y_true).tolist()
        if -1 in y_label:
            y_label = y_label[1:] + [y_label[0]]

        for class_id in y_label:
            # if class_id == -1:
            #     continue
            binary_y_true = y_true.copy()
            class_mask = y_true == class_id
            binary_y_true[class_mask] = 1
            binary_y_true[~class_mask] = 0
            binary_y_pred = y_pred[:, 0].copy()
            class_mask = y_pred[:, 0] == class_id
            binary_y_pred[class_mask] = 1
            binary_y_pred[~class_mask] = 0

            tn, fp, fn, tp = sk.confusion_matrix(binary_y_true, binary_y_pred).ravel()
            cm = np.array([[tp, fn], [fp, tn]])
            if cm[0][0] == 0:
                precision, recall = 0.0, 0.0
            else:
                precision = sk.precision_score(binary_y_true, binary_y_pred)
                recall = sk.recall_score(binary_y_true, binary_y_pred)

            if class_id == -1:
                self._cm_ood.append(cm)
            else:
                if class_id + 1 > len(self._cm):
                    self._cm.append([None] * self._cur_task + [cm])
                    self._precisions.append([None] * self._cur_task + [precision])
                    self._recalls.append([None] * self._cur_task + [recall])
                else:
                    self._cm[class_id].append(cm)
                    self._precisions[class_id].append(precision)
                    self._recalls[class_id].append(recall)

    def avg_metric(self, measures, mode='class_task_avg', level='class'):
        ratio_to_init_task = []

        for class_id, measure in enumerate(measures):
            init_i = 0
            _ratio_to_init_task = []
            for i in range(1, len(measure)):
                if measure[i - 1] is None:
                    init_i = i
                    continue
                _ratio_to_init_task.append(measure[init_i] - measure[i])

            if level == 'class':
                placeholder = [None] * int((class_id / self._current_classes))
            else:
                placeholder = [None] * class_id
            ratio_to_init_task.append(placeholder + _ratio_to_init_task)

        if mode == 'task_class_avg':
            class_avg = []
            for class_id in range(self._known_classes if level == 'class' else self._cur_task):
                class_ratio = []
                for task_id in range(self._cur_task):
                    if ratio_to_init_task[class_id][task_id] is not None:
                        class_ratio.append(ratio_to_init_task[class_id][task_id])
                class_avg.append(sum(class_ratio) / len(class_ratio))
            IFR = sum(class_avg) / len(class_avg)
        elif mode == 'class_task_avg':
            task_avg = []
            for task_id in range(self._cur_task):
                task_ratio = []
                for class_id in range((task_id + 1) * self._current_classes) if level == 'class' else range(
                        task_id + 1):
                    task_ratio.append(ratio_to_init_task[class_id][task_id])
                task_avg.append(sum(task_ratio) / len(task_ratio))
            IFR = sum(task_avg) / len(task_avg)
        else:
            raise 'ERROR: mode {} is not defined!'.format(mode)

        return IFR

        # if len(self._cm_ood) == 0:
        #     return IFR_ID
        #
        # ood_recall = []
        # for i in range(self._cur_task + 1):
        #     cm_ood = self._cm_ood[i]
        #     ood_recall.append(cm_ood[0][0] / (cm_ood[0][0] + cm_ood[0][1]))
        # ood_forget = []
        # for i in range(1, self._cur_task + 1):
        #     ood_forget.append(ood_recall[0] - ood_recall[i])
        # IFR_OOD = sum(ood_forget) / len(ood_forget)
        #
        # print(IFR_ID)
        # print(IFR_OOD)
        # print((IFR_ID + IFR_OOD) / 2)
        # return (IFR_ID + IFR_OOD) / 2

    def get_task_level_metric(self, cms, mode='precision'):
        measures = []
        for i in range(self._cur_task + 1):
            num = 0
            den = 0
            for class_id in range(self._total_classes):
                if class_id < (i + 1) * self._current_classes:
                    cm = cms[class_id][i]
                    num = num + cm[0][0]
                    if mode == 'precision':
                        den = den + cm[0][0] + cm[1][0]
                    elif mode == 'recall':
                        den = den + cm[0][0] + cm[0][1]
                    else:
                        raise 'Error: mode {} is illegal!'.format(mode)
                    if (class_id + 1) % self._current_classes == 0:
                        if den != 0:
                            task_level_measure = num / den
                        else:
                            task_level_measure = 0.0
                        if mode == 'recall' and len(self._cm_ood) > 0:
                            cm_ood = self._cm_ood[i]
                            ood_recall = cm_ood[0][0] / (cm_ood[0][0] + cm_ood[0][1])
                            task_level_measure = (task_level_measure + ood_recall) / 2
                        # elif mode == 'precision' and len(self._cm_ood) > 0:
                        #     cm_ood = self._cm_ood[i]
                        #     pred_num = cm_ood[0][0] + cm_ood[1][0]
                        #     if pred_num == 0:
                        #         ood_precision = 0.0
                        #     else:
                        #         ood_precision = cm_ood[0][0] / pred_num
                        #     task_level_measure = (task_level_measure + ood_precision) / 2
                        num = 0
                        den = 0
                        if i == 0:
                            measures.append([task_level_measure])
                        else:
                            measures[int((class_id + 1) / self._current_classes) - 1].append(task_level_measure)
                else:
                    if i == 0:
                        for j in range(self._cur_task - i):
                            measures.append([None])
                    else:
                        for j in range(i + 1, self._cur_task + 1):
                            measures[j].append(None)
                    break

        return measures

    def get_f1_score(self, precision, recall, level='task'):
        cur = self._current_classes if level == 'class' else 1
        cur_precision = np.array(precision)[-cur:, -1]
        cur_recall = np.array(recall)[-cur:, -1]
        cur_f1 = []
        for i in range(len(cur_precision)):
            cur_f1.append(f1(cur_precision[i], cur_recall[i]))

        avg_cur_f1 = sum(cur_f1) / len(cur_f1)

        return avg_cur_f1

    def get_task_level_f1_score(self, precision, recall):
        task_level_f1 = []
        for i in range(len(precision)):
            cls_f1 = []
            for j in range(len(precision)):
                cls_f1.append(None if precision[i][j] is None else f1(precision[i][j], recall[i][j]))
            task_level_f1.append(cls_f1)
        return task_level_f1

    def _eval_cls(self, y_pred, y_true):
        self.P_R_CM_per_class(y_pred, y_true)

        task_level_precision = self.get_task_level_metric(self._cm, mode='precision')
        task_level_recall = self.get_task_level_metric(self._cm, mode='recall')
        # task-level F1 score
        task_level_f1_score = self.get_f1_score(task_level_precision, task_level_recall)

        if self._cur_task == 0:
            return None, round(task_level_f1_score, 4), None, [None], [None]

        # task-level forgetting ratio for known classes
        IFR = self.avg_metric(task_level_recall, level='task')

        # forgetting+plasticity
        overall_score = ((1 - IFR) * task_level_f1_score) ** 0.5

        return (round(IFR, 4), round(task_level_f1_score, 4),
                round(overall_score, 4), task_level_recall, task_level_precision)

    def _eval_OOD(self, y_pred, y_true):
        id_ood_f1 = []
        for i in range(2):
            binary_y_true = y_true.copy()
            class_mask = y_true != -1 if i == 0 else y_true == -1
            binary_y_true[class_mask] = 1
            binary_y_true[~class_mask] = 0
            binary_y_pred = y_pred[:, 0].copy()
            class_mask = y_pred[:, 0] != -1 if i == 0 else y_pred[:, 0] == -1
            binary_y_pred[class_mask] = 1
            binary_y_pred[~class_mask] = 0

            tn, fp, fn, tp = sk.confusion_matrix(binary_y_true, binary_y_pred).ravel()
            cm = np.array([[tp, fn], [fp, tn]])
            if cm[0][0] == 0:
                precision, recall = 0.0, 0.0
            else:
                precision = sk.precision_score(binary_y_true, binary_y_pred)
                recall = sk.recall_score(binary_y_true, binary_y_pred)
            f1_score = f1(precision, recall)
            id_ood_f1.append(f1_score)
        return sum(id_ood_f1) / len(id_ood_f1)

    def eval_task(self):
        # self._ood_thresh = [i / 100.0 for i in range(0, 101)]
        y_preds, y_true, y_prob = self._eval_cnn(self.test_loader)

        multi_acc = []
        for i, y_pred in enumerate(y_preds):
            # acc = self._balanced_accuracy(y_pred[:, 0], y_true)
            acc = self._accuracy(y_pred[:, 0], y_true)
            # print(self._ood_thresh[i], acc)
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

    def get_id_num(self):
        if self.args['dataset'] == 'tinyimagenet200' or self.args['dataset'] == 'imagenet100':
            cls_num = 50
        elif self.args['dataset'] == 'cifar100':
            cls_num = 100
        else:
            raise 'dataset is not define!'
        return cls_num * (self._total_classes), cls_num * (self._total_classes - self._known_classes)

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
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

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def get_annealing_coef(self, epoch_num, annealing_step, coef_start=0.01, coef_type='exp'):
        if coef_type == 'linear':
            annealing_coef = torch.tensor(epoch_num / annealing_step, dtype=torch.float32)
        elif coef_type == 'exp':
            coef_start = torch.tensor(coef_start, dtype=torch.float32)
            annealing_coef = coef_start * torch.exp(-torch.log(coef_start) / (annealing_step) * epoch_num)
        else:
            raise 'coef_type: {} is not define!'.format(coef_type)

        annealing_coef = torch.min(
            annealing_coef.float(),
            torch.tensor(1.0, dtype=torch.float32)
        )
        return annealing_coef

    def kl_divergence(self, alpha, num_classes, device=None):
        if not device:
            device = self._device
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

    def loglikelihood_loss(self, y, alpha, device=None):
        if not device:
            device = self._device
        y = y.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood

    def _mse_loss(self, func, y, alpha, epoch_num, num_classes, annealing_step, device=None,
                  a1=0.9, a2=0.1, coef_start=0.01, coef_type='exp'):
        if not device:
            device = self._device
        y = y.to(device)
        alpha = alpha.to(device)
        loglikelihood = self.loglikelihood_loss(y, alpha, device=device)

        annealing_coef = self.get_annealing_coef(epoch_num, annealing_step, coef_start, coef_type)

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes, device=device)
        return a1 * loglikelihood + a2 * kl_div

    def _edl_loss(self, func, y, alpha, epoch_num, num_classes, annealing_step, device=None,
                  a1=0.9, a2=0.1, coef_start=0.01, coef_type='exp'):
        y = y.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

        annealing_coef = self.get_annealing_coef(epoch_num, annealing_step, coef_start, coef_type)

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes, device=device)
        return a1 * A + a2 * kl_div

    def edl_loss(
            self, output, target, epoch_num, num_classes, annealing_step, device=None,
            loss_type='digamma', activate_type='relu', a1=0.9, a2=0.1, coef_start=0.01, coef_type='exp'
    ):
        if not device:
            device = self._device

        if loss_type == 'ce':
            return F.cross_entropy(output, target.long())
        elif loss_type == 'digamma':
            base_loss = self._edl_loss
            func = torch.digamma
        elif loss_type == 'log':
            base_loss = self._edl_loss
            func = torch.log
        elif loss_type == 'mse':
            base_loss = self._mse_loss
            func = None
        else:
            raise 'loss_type: {} is not define!'.format(loss_type)

        alpha = self.get_evidence(output, activate_type) + 1  # (1000, 10)

        y = one_hot_embedding(target.long(), num_classes, device).long()

        loss = torch.mean(
            base_loss(
                func, y, alpha, epoch_num, num_classes, annealing_step, device, a1, a2, coef_start, coef_type
            )
        )
        return loss

    def get_evidence(self, y, activate_type='exp'):
        if activate_type == 'exp':
            return exp_evidence(y)
        elif activate_type == 'relu':
            return relu_evidence(y)
        else:
            raise 'activate_type: {} is not define!'.format(activate_type)

    def cal_task_entropy(self, prob, pred):
        inc_size = self.args['increment']
        task_prob = []
        for i, p in enumerate(pred):
            p = int(p / inc_size)
            task_prob.append(prob[i, p * inc_size:(p + 1) * inc_size])
        task_prob = torch.stack(task_prob)
        # Calculate entropy using base of number of classes
        task_entropy = -torch.sum(task_prob * torch.log(task_prob + 1e-9) / torch.log(torch.tensor(inc_size).float()),
                                  dim=-1)
        return task_entropy


def one_hot_embedding(labels, num_classes=10, device=None):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes, device=device)
    return y[labels]


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y, min=-10, max=10):
    return torch.exp(torch.clamp(y, min, max))


def fpr_and_thresh_at_recall(y_true, y_score, recall_level=0.95, pos_label=None, only_thresh=False):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = []
    for r in recall_level:
        cutoff.append(np.argmin(np.abs(recall - r)))
    cutoff = np.array(cutoff)

    if only_thresh:
        return thresholds[cutoff]

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def f1(p, r):
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)
