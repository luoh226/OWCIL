import sys
import logging
import copy

import numpy as np
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    if args["prefix"] == "train_efcil":
        dir_path = "logs/{}/{}/{}/{}/{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"]
        )
        logfilename = dir_path + "{}_{}_{}_{}_{}_{}_{}_{}".format(
            args["loss_type"],
            args["activate_type"],
            args["a1"],
            args["a2"],
            args["coef_type"],
            args["coef_start"],
            args["batch_size"],
            "wauto" if args["auto_aug"] else "woauto",
        )
    elif args["prefix"] == "separate_head":
        dir_path = "logs/{}/{}/{}/{}/{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"]
        )
        logfilename = dir_path + "{}_{}_{}".format(
            "seph{}".format(args["separate_type"]) if args["separate_head"] else "woseph",
            args["b1"],
            args["b2"]
        )
    elif args["prefix"] == "ccil":
        root_path = "logs/{}/{}/{}/{}/{}/{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"],
            args["seed"]
        )
        if args["eval_only"] or args["get_thresh"]:
            dir_path = root_path + "/{}".format(args["ood_method"])
            logfilename = dir_path + "/{}".format("get_ood_thresh" if args["get_thresh"] else "test_log")
        else:
            dir_path = root_path + "/trained_model"
            logfilename = dir_path + "/train_log"
    else:
        dir_path = "logs/{}/{}/{}/{}/{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"],
        )
        logfilename = dir_path + "{}_{}".format(
            args["seed"],
            args["convnet_type"],
        )

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    args["save_path"] = root_path
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["k_fold"],
        args["ood_ratio"],
    )
    if not args["eval_only"] and args["k_fold"] > 1:
        model_list = [factory.get_model(args["model_name"], args) for _ in range(args["k_fold"])]
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve, edl_curve, task_curve, avg_acc = {"top1": [], "top5": []}, {"top1": [], "top5": []}, {
        "top1": [], "top5": []}, {"top1": [], "top5": []}, {}
    ood_thresh_curve, ood_score_curve, forget_ratio_curve, f1_score_curve, overall_score_curve, acc_curve, id_acc_curve, multi_acc_curve = [], [], [], [], [], [], [], []
    for task in range(data_manager.nb_tasks):
        logging.info("=" * 50)
        logging.info("=" + " " * 20 + "task: " + str(task) + " " * 20 + "=")
        logging.info("=" * 50)
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))
        if args["k_fold"] > 1:
            # k fold cross validation
            if not args["eval_only"]:
                if not args["get_thresh"]:
                    for i in range(args["k_fold"]):
                        model_list[i].incremental_train(data_manager, i)
                        model_list[i].after_task(i)
                    # final train
                    model.incremental_train(data_manager)
                # get ood thresh
                elif args["get_thresh"]:
                    ood_threshs = []
                    for i in range(args["k_fold"]):
                        model_list[i]._eval_only = True
                        model_list[i].incremental_train(data_manager, i)
                        ood_thresh = model_list[i]._get_ood_thresh(model_list[i].val_loader)
                        ood_threshs.append(ood_thresh)
                        model_list[i].after_task(i)
                        logging.info("Task {} Fold {} ood threshold: {}".format(task, i, ood_thresh))
                    ood_threshs = np.array(ood_threshs)
                    max_ood_thresh = list(np.max(ood_threshs, axis=0))
                    model._eval_only = True
                    model.incremental_train(data_manager)
                    model._ood_thresh = max_ood_thresh
                    model.save_ood_thresh(model._save_path)
                    logging.info("Task {} ood threshold: {}".format(task, ood_threshs))
                    logging.info("Task {} max ood threshold: {}".format(task, max_ood_thresh))
            else:
                model.incremental_train(data_manager)
        else:
            model.incremental_train(data_manager)

        # inference & eval
        if args["eval_only"]:
            if args["prefix"] == "ccil":
                acc, id_acc, ood_score, forget_ratio, f1_score, overall_score, recall_curves, precision_curves, multi_acc = model.eval_task()

                acc_curve.append(acc)
                id_acc_curve.append(id_acc)
                # ood_thresh_curve.append(round(model._ood_thresh, 4))
                ood_score_curve.append(ood_score)
                forget_ratio_curve.append(forget_ratio)
                f1_score_curve.append(f1_score)
                overall_score_curve.append(overall_score)
                multi_acc_curve.append(multi_acc)

                logging.info("acc: {}, avg: {}".format(acc_curve, get_avg(acc_curve)))
                logging.info("id acc: {}, avg: {}".format(id_acc_curve, get_avg(id_acc_curve)))
                # logging.info('ood thresh: {}, avg: {}'.format(ood_thresh_curve, get_avg(ood_thresh_curve)))
                logging.info("ood_score: {}, avg: {}".format(ood_score_curve, get_avg(ood_score_curve)))
                logging.info("task-level forget ratio to init: {}, avg: {}".format(forget_ratio_curve, get_avg(forget_ratio_curve)))
                logging.info("task-level current classes f1 score: {}, avg: {}".format(f1_score_curve, get_avg(f1_score_curve)))
                logging.info("task-level overall score: {}, avg: {}".format(overall_score_curve, get_avg(overall_score_curve)))

                # if task == 1:
                if task == data_manager.nb_tasks - 1:
                    curves = [acc_curve, id_acc_curve, ood_score_curve,
                              forget_ratio_curve, f1_score_curve, overall_score_curve, recall_curves, precision_curves]
                    curves = save_data(curves, os.path.join(os.path.join(args["save_path"], args["ood_method"]), 'curves_data_{}_ood={}.txt'.format(args["ood_thresh_type"], args["ood_ratio"])))
                    # draw_curve(curves, os.path.join(args["save_path"], args["ood_method"]))
                    # save_data(multi_acc_curve, os.path.join(os.path.join(args["save_path"], args["ood_method"]), 'multi_acc_ood={}.txt'.format(args["ood_ratio"])), acc=True)
                    # draw_acc_curve(multi_acc_curve, os.path.join(os.path.join(args["save_path"], args["ood_method"]), 'multi_acc_ood={}.png'), args["ood_ratio"])
            else:
                cnn_accy, nme_accy, edl_accy, task_accy = model.eval_task()

                if nme_accy is None and edl_accy is not None:
                    if not args["eval_ccil"]:
                        logging.info("CNN: {}".format(cnn_accy["grouped"]))
                        logging.info("EDL: {}".format(edl_accy["grouped"]))
                        logging.info("TASK: {}\n".format(task_accy["grouped"]))

                    cnn_curve["top1"].append(cnn_accy["top1"])
                    cnn_curve["top5"].append(cnn_accy["top5"])
                    edl_curve["top1"].append(edl_accy["top1"])
                    edl_curve["top5"].append(edl_accy["top5"])
                    task_curve["top1"].append(task_accy["top1"])

                    logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                    logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
                    logging.info("EDL top1 curve: {}".format(edl_curve["top1"]))
                    logging.info("EDL top5 curve: {}".format(edl_curve["top5"]))
                    logging.info("TASK top1 curve: {}\n".format(task_curve["top1"]))

                    logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
                    logging.info("Average Accuracy (EDL): {}".format(sum(edl_curve["top1"]) / len(edl_curve["top1"])))
                    logging.info(
                        "Average Accuracy (TASK): {}\n".format(sum(task_curve["top1"]) / len(task_curve["top1"])))
                elif nme_accy is None and edl_accy is None:
                    if not args["eval_ccil"]:
                        logging.info("CNN: {}".format(cnn_accy["grouped"]))

                    cnn_curve["top1"].append(cnn_accy["top1"])
                    cnn_curve["top5"].append(cnn_accy["top5"])

                    logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                    logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

                    logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
                else:
                    if not args["eval_ccil"]:
                        logging.info("CNN: {}".format(cnn_accy["grouped"]))
                        logging.info("NME: {}\n".format(nme_accy["grouped"]))

                    cnn_curve["top1"].append(cnn_accy["top1"])
                    cnn_curve["top5"].append(cnn_accy["top5"])
                    nme_curve["top1"].append(nme_accy["top1"])
                    nme_curve["top5"].append(nme_accy["top5"])

                    logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                    logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
                    logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
                    logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

                    logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
                    logging.info("Average Accuracy (NME): {}\n".format(sum(nme_curve["top1"]) / len(nme_curve["top1"])))
        model.after_task()


def _set_device(args):
    device_type = args["device"]

    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def save_data(curves, title, acc=False):
    if not acc:
        # add avg
        for i in range(len(curves)):
            if i < 6:
                curves[i].append(get_avg(curves[i]))
    # save data
    with open(title, 'w') as f:
        for i in range(len(curves)):
            for val in curves[i]:
                f.write(str(val) + ' ')
            f.write('\n')
    return curves


def draw_acc_curve(curves, save_path, ood_ratio):
    fig_name = 'Accuracy, ood ratio = {}'.format(ood_ratio)
    plt.figure()
    plt.title(fig_name, fontsize=10, fontweight='bold')
    plt.xlabel('Val-set True Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('Test-set Accuracy', fontsize=12, fontweight='bold')
    plt.xticks([i for i in range(0, 101, 5)])
    for curve_o in curves:
        curve = [round(c * 100, 2) for c in curve_o]
        plt.plot([i for i in range(0, 101)], curve, linewidth=1.5)

    font = font_manager.FontProperties(family='Times new roman',  # 'Times new roman',
                                       weight='bold',
                                       style='normal', size=12)
    plt.legend(labels=['task={}'.format(i) for i in range(len(curves))], fontsize=12, prop=font)
    plt.grid(alpha=0.8, linestyle='--')
    plt.savefig(save_path.format(ood_ratio))


def draw_curve(curves, save_path):
    fig_name = 'global accuracy'
    plt.figure()
    plt.title(fig_name)
    plt.xlabel('task')
    plt.xticks(np.linspace(1, 10, 10), [str(i) for i in range(1, 11)])
    for i in range(0, 2):
        plt.plot(list(range(1, len(curves[i]))), curves[i][:-1])
    plt.legend(labels=['acc, avg={}'.format(curves[0][-1]), 'id-acc, avg={}'.format(curves[1][-1])])
    plt.savefig(os.path.join(save_path, fig_name + '.png'))
    # plt.show()

    fig_name = 'OOD metrics'
    plt.figure()
    plt.title(fig_name)
    plt.xlabel('task')
    plt.xticks(np.linspace(1, 10, 10), [str(i) for i in range(1, 11)])
    for i in range(2, 3):
        plt.plot(list(range(1, len(curves[i]))), curves[i][:-1])
    plt.legend(labels=['ood score, avg={}'.format(curves[2][-1])])
    plt.savefig(os.path.join(save_path, fig_name + '.png'))
    # plt.show()

    fig_name = 'forgetting ratio'
    plt.figure()
    plt.title(fig_name)
    plt.xlabel('task')
    plt.xticks(np.linspace(1, 10, 10), [str(i) for i in range(1, 11)])
    for i in range(3, 4):
        plt.plot(list(range(2, len(curves[i]))), curves[i][1:-1])
    plt.legend(labels=['task-level-to-init, avg={}'.format(curves[3][-1])])
    plt.savefig(os.path.join(save_path, fig_name + '.png'))
    # plt.show()

    fig_name = 'current classes f1 score'
    plt.figure()
    plt.title(fig_name)
    plt.xlabel('task')
    plt.xticks(np.linspace(1, 10, 10), [str(i) for i in range(1, 11)])
    for i in range(4, 5):
        plt.plot(list(range(1, len(curves[i]))), curves[i][:-1])
    plt.legend(labels=['task-level, avg={}'.format(curves[4][-1])])
    plt.savefig(os.path.join(save_path, fig_name + '.png'))
    # plt.show()

    fig_name = 'overall score'
    plt.figure()
    plt.title(fig_name)
    plt.xlabel('task')
    plt.xticks(np.linspace(1, 10, 10), [str(i) for i in range(1, 11)])
    for i in range(5, 6):
        plt.plot(list(range(2, len(curves[i]))), curves[i][1:-1])
    plt.legend(labels=['task-level, avg={}'.format(curves[5][-1])])
    plt.savefig(os.path.join(save_path, fig_name + '.png'))
    # plt.show()

    fig_name = 'task-level recall curves'
    plt.figure()
    plt.title(fig_name)
    plt.xlabel('task')
    plt.xticks(np.linspace(1, 10, 10), [str(i) for i in range(1, 11)])
    for curve in curves[6]:
        start_task = 1
        tmp = []
        for c in curve:
            if c is not None:
                tmp.append(c)
            else:
                start_task += 1
        plt.plot(list(range(start_task, start_task + len(tmp))), tmp)
    plt.legend(labels=['task {}'.format(i) for i in range(1, 11)])
    plt.savefig(os.path.join(save_path, fig_name + '.png'))
    # plt.show()

    fig_name = 'task-level precision curves'
    plt.figure()
    plt.title(fig_name)
    plt.xlabel('task')
    plt.xticks(np.linspace(1, 10, 10), [str(i) for i in range(1, 11)])
    for curve in curves[7]:
        start_task = 1
        tmp = []
        for c in curve:
            if c is not None:
                tmp.append(c)
            else:
                start_task += 1
        plt.plot(list(range(start_task, start_task + len(tmp))), tmp)
    plt.legend(labels=['task {}'.format(i) for i in range(1, 11)])
    plt.savefig(os.path.join(save_path, fig_name + '.png'))
    # plt.show()


def get_avg(list, pos=4):
    _list = [i for i in list if i is not None]
    if len(_list) > 0:
        return round(sum(_list) / len(_list), pos)
    else:
        return 0.0
