import logging
import warnings
import numpy as np
from tqdm import tqdm
import torch

from torch import nn
from torch import optim
from torch import linalg as LA
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy
from utils.maha_utils import compute_common_cov, compute_new_common_cov, compute_new_cov

EPSILON = 1e-8


class FeCAM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = CosineIncrementalNet(args, False)
        self._protos = []
        self._init_protos = []
        self._common_cov = None
        self._cov_mat = []
        self._diag_mat = []
        self._common_cov_shrink = None
        self._cov_mat_shrink = []
        self._norm_cov_mat = []

    def after_task(self, fold='final'):
        self._known_classes = self._total_classes
        if not self._eval_only and not self._get_thresh and self._cur_task == 0:
            self.save_checkpoint(self._save_path, fold)

    def incremental_train(self, data_manager, fold='final'):
        self.data_manager = data_manager
        self._cur_task += 1
        if self.args['dataset'] == "cifar100":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ]
        elif self.args['dataset'] == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        elif self.args['dataset'] == "imagenet100":
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        self._current_classes = self.data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._current_classes
        self._network.update_fc(self._total_classes, self._cur_task)
        self._network_module_ptr = self._network
        self._fold = fold

        if self._cur_task > 0:  # Freezing the network
            for p in self._network.convnet.parameters():
                p.requires_grad = False

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = self.data_manager.get_dataset_Kfold(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            fold=fold,
        )
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"])
        val_dataset = self.data_manager.get_dataset_Kfold(
            np.arange(self._known_classes, self._total_classes),
            source="val",
            fold=fold,
        )
        self.val_loader = DataLoader(val_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
        test_dataset = self.data_manager.get_dataset_wood(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            if self._eval_only:
                self.load_checkpoint(self._save_path, self._fold)
            else:
                self._epoch_num = self.args["init_epochs"]
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
                )), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=self.args["init_epochs"])
                self._train_function(train_loader, test_loader, optimizer, scheduler)

            self._build_base_protos()
            self._build_protos()
            if self.args["full_cov"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    if self.args["shrink"]:  # we apply covariance shrinkage 2 times to obtain better estimates of matrices
                        for cov in self._cov_mat:
                            self._cov_mat_shrink.append(self.shrink_cov(cov))
                    if self.args["norm_cov"]:
                        self._norm_cov_mat = self.normalize_cov()
                else:
                    self._common_cov = compute_common_cov(train_loader, self)
            elif self.args["diagonal"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    for cov in self._cov_mat:
                        self._cov_mat_shrink.append(self.shrink_cov(cov))
                    for cov in self._cov_mat_shrink:
                        cov = self.normalize_cov2(cov)
                        self._diag_mat.append(self.diagonalization(cov))
        else:
            if self._eval_only:
                self.load_checkpoint(self._save_path, self._fold, only_thresh=True)
            self._cov_mat_shrink, self._norm_cov_mat, self._diag_mat = [], [], []
            # if "EDL" in self._ood_method and not self._eval_only:
            #     self._epoch_num = int(self.args["init_epochs"] / 4)
            #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
            #     )), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])
            #     scheduler = optim.lr_scheduler.CosineAnnealingLR(
            #         optimizer=optimizer, T_max=self.args["init_epochs"])
            #     self._train_function(train_loader, test_loader, optimizer, scheduler)

            self._build_protos()
            self._update_fc()

            if self.args["full_cov"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    if self.args["shrink"]:
                        for cov in self._cov_mat:
                            self._cov_mat_shrink.append(self.shrink_cov(cov))
                    if self.args["norm_cov"]:
                        self._norm_cov_mat = self.normalize_cov()
                else:
                    self._common_cov = compute_new_common_cov(train_loader, self)
            elif self.args["diagonal"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    for cov in self._cov_mat:
                        self._cov_mat_shrink.append(self.shrink_cov(cov))
                    for cov in self._cov_mat_shrink:
                        cov = self.normalize_cov2(cov)
                        self._diag_mat.append(self.diagonalization(cov))

    def _build_base_protos(self):
        for class_idx in range(self._known_classes, self._total_classes):
            class_mean = self._network.fc.weight.data[class_idx]
            self._init_protos.append(class_mean)

    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                idx_dataset = self.data_manager.get_dataset_Kfold(np.arange(class_idx, class_idx + 1),
                                                                  source='train', fold=self._fold, mode='test')
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(torch.tensor(class_mean).to(self._device))

    def _update_fc(self):
        self._network.fc.fc2.weight.data = torch.stack(self._protos[-self.args["increment"]:], dim=0).to(self._device)
        # for cosine incremental fc layer

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
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                if self._cur_task == 0:
                    logits = self._network(inputs)['logits']
                else:
                    logits = self._network_module_ptr.fc(inputs)['logits']
                # if self._separate_head:
                #     logits_edl = outputs["logits_edl"]
                #     loss_edl = self.edl_loss(logits_edl[:, self._known_classes:], targets - self._known_classes,
                #                              epoch, self._current_classes, self._epoch_num, self._device,
                #                              **self._loss_args) * self.args["b1"]
                #     if self._cur_task == 0:
                #         loss = loss * self.args["b2"]
                #         loss += loss_edl
                #     else:
                #         loss = loss_edl

                loss = F.cross_entropy(logits, targets.long())
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

    def load_checkpoint(self, filename, fold, only_thresh=False):
        if not only_thresh:
            trained_model = torch.load("{}/trained_model/{}_{}.pkl".format(filename, 0, fold))
            self._network.load_state_dict(trained_model["model_state_dict"])
            self._network.to(self._device)

        if not self._get_thresh and fold == 'final' and self._ood_method != "None":
            with open("{}/{}/ood_thresh_{}.txt".format(filename, self._ood_method, self._ood_thresh_type)) as f:
                ood_threshs = f.read().split('\n')
                ood_threshs = ood_threshs[self._cur_task]
                self._ood_thresh = [float(ood_threshs)]

    def _get_ood_thresh(self, loader):
        y_preds, y_true, id_y_prob = self._eval_maha(loader, self._init_protos, self._protos, val=True)
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
            y_preds, y_true, id_y_prob = self._eval_maha(loader, self._init_protos, self._protos)
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

    def _eval_maha(self, loader, init_means, class_means, val=False):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = self._maha_dist(vectors, init_means, class_means)
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance
        # prob = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))  # 距离最值归一化
        # prob = 1 - prob # 距离越大，得分越低
        # max_prob = np.amax(prob, axis=1)
        y_pred = np.argsort(scores, axis=1)[:, :self.topk]

        # cnn inference
        y_true, y_prob = [], []
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
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
        y_preds, y_true, y_prob = self._eval_maha(self.test_loader, self._init_protos, self._protos)

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

    def _maha_dist(self, vectors, init_means, class_means):
        vectors = torch.tensor(vectors).to(self._device)
        if self.args["tukey"] and self._cur_task > 0:
            vectors = self._tukeys_transform(vectors)
        maha_dist = []
        for class_index in range(self._total_classes):
            if self._cur_task == 0:
                dist = self._mahalanobis(vectors, init_means[class_index])
            else:
                if self.args["ncm"]:
                    dist = self._mahalanobis(vectors, class_means[class_index])
                elif self.args["full_cov"]:
                    if self.args["per_class"]:
                        if self.args["norm_cov"]:
                            dist = self._mahalanobis(vectors, class_means[class_index], self._norm_cov_mat[class_index])
                        elif self.args["shrink"]:
                            dist = self._mahalanobis(vectors, class_means[class_index], self._cov_mat_shrink[class_index])
                        else:
                            dist = self._mahalanobis(vectors, class_means[class_index], self._cov_mat[class_index])
                    else:
                        dist = self._mahalanobis(vectors, class_means[class_index], self._common_cov)
                elif self.args["diagonal"]:
                    if self.args["per_class"]:
                        dist = self._mahalanobis(vectors, class_means[class_index], self._diag_mat[class_index])
            maha_dist.append(dist)
        maha_dist = np.array(maha_dist)  # [nb_classes, N]
        return maha_dist

    def _mahalanobis(self, vectors, class_means, cov=None):
        if self.args["tukey"] and self._cur_task > 0:
            class_means = self._tukeys_transform(class_means)
        x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(class_means, p=2, dim=-1)
        if cov is None:
            cov = torch.eye(self._network.feature_dim)  # identity covariance matrix for euclidean distance
        inv_covmat = torch.linalg.pinv(cov).float().to(self._device)
        left_term = torch.matmul(x_minus_mu, inv_covmat)
        mahal = torch.matmul(left_term, x_minus_mu.T)
        return torch.diagonal(mahal, 0).cpu().numpy()

    def diagonalization(self, cov):
        diag = cov.clone()
        cov_ = cov.clone()
        cov_.fill_diagonal_(0.0)
        diag = diag - cov_
        return diag

    def shrink_cov(self, cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag * mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0])
        alpha1 = self.args["alpha1"]
        alpha2 = self.args["alpha2"]
        cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
        return cov_

    def normalize_cov(self):
        if self.args["shrink"]:
            cov_mat = self._cov_mat_shrink
        else:
            cov_mat = self._cov_mat
        norm_cov_mat = []
        for cov in cov_mat:
            sd = torch.sqrt(torch.diagonal(cov))
            cov = cov / (torch.matmul(sd.unsqueeze(1), sd.unsqueeze(0)))
            norm_cov_mat.append(cov)

        return norm_cov_mat

    def normalize_cov2(self, cov):
        diag = torch.diagonal(cov)
        norm = torch.linalg.norm(diag)
        cov = cov / norm
        return cov

    def _tukeys_transform(self, x):
        beta = self.args["beta"]
        x = torch.tensor(x)
        if beta == 0:
            return torch.log(x)
        else:
            return torch.pow(x, beta)

    def _extract_vectors_common_cov(self, loader):
        self._network.eval()
        vectors, covs = [], []
        for i, (_, _inputs, _) in enumerate(loader):
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            if i % 20 == 0:
                vecs = np.concatenate(vectors)
                if self.args["tukey"]:
                    vecs = self._tukeys_transform(vecs)
                covs.append(np.cov(vecs.T))
                vectors = []

        if len(vectors) > 4:
            vecs = np.concatenate(vectors)
            if self.args["tukey"]:
                vecs = self._tukeys_transform(vecs)
            covs.append(np.cov(vecs.T))

        cov = np.mean(covs, axis=0)
        return torch.tensor(cov)
