import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

datasets = 'cifar100'  # tinyimagenet200    cifar100
init_cls = 50
increment = 10
seed_list = [1993]
ood_ratios = [1.0]
ood_ids = np.array([0, 1])
ood_methods = list(np.array(['None', 'MSP', 'ENERGY'])[ood_ids])
method_id = np.array([0, 1, 2, 3, 4])  # 0,1,4 # 2, 4
cil_methods = list(np.array(['PASS', 'IL2A', 'SSRE', 'FeTrIL', 'FeCAM'])[method_id])
# thr_types = ['TPR95', 'MID']
thr_types = ['MID']
color = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange'])[method_id]
ood_color = np.array(['paleturquoise', 'cyan', 'c', 'darkcyan', 'darkslategray'])
linestyle = np.array(['-', '-', '-', '-', '-', '-', '-'])
# marker = np.array(['.', '*', '+', 'x', '^', 's', 'v'])
marker = np.array(['^', 's'])
root_path = '../logs/'

ood_score = []
curves_info = []
metric_avg = {'Accuracy Var': [], 'Forgetting Ratio Var': [], 'Task-level F1 Score Var': []}


def avg_result(ori_results):
    seed_num = len(seed_list)
    result_num = int(len(ori_results) / seed_num)
    results = []
    for i in range(result_num):
        result = ori_results[i].copy()
        for j in range(1, seed_num):
            for k in range(len(result)):
                result[k] += ori_results[i + j * result_num][k]
        result = [round(r / seed_num, 4) for r in result]
        results.append(result)

    return results


def plot_curve(curves, title='figure', axs=None, subfig=0):
    assert len(curves) > 0
    for c in curves:
        metric_avg[title + " Var"].append(c[-1] * 100)
    start = 1 if title == 'Accuracy' or title == 'Task-level F1 Score' else 2
    num_classes = 200 if datasets == 'tinyimagenet200' else 100
    if init_cls != 10:
        start -= 1
        num_classes -= init_cls
    T = int(num_classes / increment)
    legend = []
    if isinstance(axs, np.ndarray):
        ax = axs[subfig]
    else:
        ax = axs
    ax.set_xlabel('Task', fontsize=10, fontweight='bold')
    ax.set_ylabel(title, fontsize=10, fontweight='bold')
    star = 100
    stop = 0
    result0 = []
    result1 = []
    for i, curve_o in enumerate(curves):
        curve = [round(c * 100, 2) for c in curve_o]
        cil_id = cil_methods.index(curves_info[i]['cil'])
        ood_id = ood_methods.index(curves_info[i]['ood'])
        thr_id = thr_types.index(curves_info[i]['thr_type'])
        ax.plot(np.linspace(start, T, T - start + 1), curve[:-1], color=color[cil_id],
                linestyle=linestyle[ood_id], linewidth=1.5, marker=marker[ood_id], markersize=6)
        # ood_ratio_id = ood_ratios.index(curves_info[i]['ood_ratio'])
        # ax.plot(np.linspace(start, T, T - start + 1), curve[:-1], color=ood_color[ood_ratio_id],
        #         linestyle=linestyle[ood_id], linewidth=1.5, marker=marker[ood_id], markersize=6)
        if ood_methods[ood_id] == 'None':
            legend.append('{}: {}'.format(cil_methods[cil_id], curve[-1]))
        else:
            # legend.append('{}-{}, ood:{}'.format(cil_methods[cil_id], ood_methods[ood_id], ood_score[i][0]))
            # legend.append('{}-{}'.format(cil_methods[cil_id], 'ours'))
            # legend.append('OR={}'.format(ood_ratios[ood_ratio_id]))
            # legend.append('{}-{}: {}'.format(cil_methods[cil_id], thr_types[thr_id], curve[-1]))
            legend.append('{}-ours: {}'.format(cil_methods[cil_id], curve[-1]))
        # if ood_id == 0:
        #     result0.append(curve[-1])
        # else:
        #     result1.append(curve[-1])
        star = min(star, min(curve[:-1]))
        stop = max(stop, max(curve[:-1]))
    # for i in range(len(result0)):
    #     print(result0[i], result1[i])
    ax.set_xticks(np.linspace(start, T, T - start + 1))
    y_gap = 5.0
    ytick = [i * y_gap for i in range(int(star / y_gap), int(stop / y_gap) + 2)]
    # print(ytick)
    ax.set_yticks(ytick)
    ax.grid(alpha=0.8, linestyle='--')
    return legend


# def plot_all_curve(acc, forgeting, plasticity):


if __name__ == '__main__':
    acc = []
    forgetting = []
    f1 = []
    score = []
    for seed in seed_list:
        for ood_ratio in ood_ratios:
            for cil in cil_methods:
                for ood in ood_methods:
                    # flag = 1
                    for thr_type in thr_types:
                        # if ood == 'None' and flag == 0:
                        #     continue
                        # flag = 0
                        with open(os.path.join(root_path,
                                               '{}/{}/{}/{}/ccil/{}/{}/curves_data_{}_ood={}.txt'.format(cil.lower(),
                                                                                                     datasets,
                                                                                                     0 if init_cls == 10 else init_cls,
                                                                                                     increment,
                                                                                                     seed, ood,
                                                                                                     thr_type,
                                                                                                     ood_ratio))) as f:
                            curves_data = f.read().split('\n')[:-1]
                            acc.append([float(c) for c in curves_data[0].split(' ')[:-1]])
                            ood_score.append([float(curves_data[2].split(' ')[-2])])
                            forgetting.append([float(c) for c in curves_data[3].split(' ')[1:-1]])
                            f1.append([float(c) for c in curves_data[4].split(' ')[:-1]])
                            score.append([float(c) for c in curves_data[5].split(' ')[1:-1]])
                            curve_info = {"seed": seed, "ood_ratio": ood_ratio, "ood": ood, "cil": cil, "thr_type": thr_type}
                            curves_info.append(curve_info)
                            print(curve_info)

    font = font_manager.FontProperties(family='Times new roman',  # 'Times new roman',
                                       weight='bold',
                                       style='normal', size=12)

    ood_score = avg_result(ood_score)
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    legend = plot_curve(avg_result(acc), 'Accuracy', axs, 0)
    plot_curve(avg_result(forgetting), 'Forgetting Ratio', axs, 1)
    plot_curve(avg_result(f1), 'Task-level F1 Score', axs, 2)
    metric_var = {}
    for k, v in metric_avg.items():
        metric_var[k] = np.var(v)
    print(metric_var)
    axs[1].legend(legend, bbox_to_anchor=(0.5, -(0.23 + 0.034 * len(cil_methods) * len(ood_ratios))),
                  loc='lower center', ncol=len(cil_methods) * len(ood_ratios), fontsize=10,
                  prop=font)

    save_path = os.path.join(root_path, 'visualization')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'All.png'), dpi=400, bbox_inches='tight')
    # plot_all_curve(avg_result(acc), avg_result(forgetting), avg_result(f1))
    # plot_curve(avg_result(score), 'Overall Score')
