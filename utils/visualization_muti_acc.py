import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

datasets = 'cifar100'  # tinyimagenet200
increment = 10
seed_list = [1993]
ood_ratios = [1.0]
ood_ids = np.array([1, 2])
ood_methods = list(np.array(['None', 'MSP', 'ENERGY'])[ood_ids])
method_id = np.array([0, 1])  # 0,1,4 # 2, 4
cil_methods = list(np.array(['EwC', 'LwF', 'PASS', 'IL2A', 'SSRE', 'FeTrIL', 'FeCAM'])[method_id])
root_path = '../logs/'

curves_info = []

def draw_acc_curve(curves, save_path, ood_ratio):
    fig_name = 'Accuracy, ood ratio = {}'.format(ood_ratio)
    plt.figure()
    plt.title(fig_name, fontsize=12, fontweight='bold')
    plt.xlabel('Val-set True Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('Test-set Accuracy', fontsize=12, fontweight='bold')
    plt.xticks([i for i in range(0, 101, 5)])
    for curve_o in curves:
        curve = [round(c * 100, 2) for c in curve_o]
        plt.plot([i for i in range(0, 101)], curve, linewidth=1.5)
    plt.axvline(x=95, linewidth=1.5, linestyle='--', color='k')

    font = font_manager.FontProperties(family='Times new roman',  # 'Times new roman',
                                       weight='bold',
                                       style='normal', size=8)
    plt.legend(labels=['task={}'.format(i) for i in range(len(curves))], fontsize=8, prop=font)
    plt.grid(alpha=0.8, linestyle='--')
    plt.savefig(save_path.format(ood_ratio), dpi=200, bbox_inches='tight')

def draw_final_acc_curve(curves, ood_ratio):
    fig_name = 'Accuracy, ood ratio = {}'.format(ood_ratio)
    plt.figure()
    plt.title(fig_name, fontsize=12, fontweight='bold')
    plt.xlabel('Val-set True Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('Test-set Accuracy', fontsize=12, fontweight='bold')
    plt.xticks([i for i in range(0, 101, 5)])
    for i, curve_o in enumerate(curves):
        curve = [round(c * 100, 2) for c in curve_o]
        plt.plot([i for i in range(0, 101)], curve, linewidth=1.5)
    plt.axvline(x=95, linewidth=1.5, linestyle='--', color='k')

    font = font_manager.FontProperties(family='Times new roman',  # 'Times new roman',
                                       weight='bold',
                                       style='normal', size=8)
    labels = []
    for info in curves_info:
        labels.append("{}-{}".format(info["cil"], info["ood"]))
    plt.legend(labels=labels, fontsize=8, prop=font)
    plt.grid(alpha=0.8, linestyle='--')
    plt.savefig(os.path.join(root_path, 'visualization/Multi_Acc_ood={}.png'.format(ood_ratio)), dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    final_acc = []
    for seed in seed_list:
        for ood_ratio in ood_ratios:
            for cil in cil_methods:
                for ood in ood_methods:
                    file_name = os.path.join(root_path,
                                           '{}/{}/0/{}/ccil/{}/{}/'.format(cil.lower(), datasets,
                                                                                                 increment,
                                                                                                 seed, ood))
                    with open(os.path.join(file_name, 'multi_acc_ood={}.txt'.format(ood_ratio))) as f:
                        curves_data = f.read().split('\n')[:-1]
                        for i in range(len(curves_data)):
                            curves_data[i] = [float(c) for c in curves_data[i].split(' ')[:-1]]

                        curve_info = {"seed": seed, "ood_ratio": ood_ratio, "ood": ood, "cil": cil}
                        curves_info.append(curve_info)
                        print(curve_info)
                        final_acc.append(curves_data[-1])
                        draw_acc_curve(curves_data, os.path.join(file_name, 'multi_acc_ood={}.png'), ood_ratio)

    draw_final_acc_curve(final_acc, ood_ratios[0])
