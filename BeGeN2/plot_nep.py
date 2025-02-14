#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/21 16:40
# @Author  : 兵
# @email    : 1747193328@qq.com

import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import font_manager
# 添加字体
font_path = '/home/lzd/soft/bin/times.ttf'
font_manager.fontManager.addfont(font_path)
# 更新 rcParams
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.figsize': (8.5, 6),
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
    'lines.linewidth': 1.5,
    'lines.markersize': 10,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
     'xtick.labelsize': 20,  # 设置 x 轴刻度标签的字体大小为 20
    'ytick.labelsize': 20,

})





if os.path.exists("stress_test.out"):
    os.remove("stress_test.out")

if os.path.exists("stress_train.out"):
    os.remove("stress_train.out") 

# 字典储存文件名
Config = [
    {"name": "energy", "unit": "eV/atom"},
    {"name": "force", "unit": "eV/A"},
    {"name": "virial", "unit": "eV/atom"},
    {"name": "stress", "unit": "GPa"},
]


def plot_loss_result(axes: plt.Axes):
    # 画loss图
    loss = np.loadtxt("loss.out")
    axes.loglog(loss[:, 1:7],label=['Total', 'L1-regularization','L2-regularization', 'Energy-train','Force-train', 'Virial-train'])
    axes.set_xlabel('Generation/100',fontsize=20, fontweight='bold')
    axes.set_ylabel('Loss',fontsize=20, fontweight='bold')
    axes.set_title('(a)',fontsize=20, fontweight='bold')

    # # 测试集画图
    # if np.any(loss[7:10] != 0):
        # axes.loglog(loss[:, 7:10], label=['Energy-test', 'Force-test', 'Virial-test'])
        
    axes.legend(ncol=2)

#plot_train_result(axes, config,col=color)
def plot_train_result(axes: plt.Axes, config: dict,col: str, tit:str):
    types = ["train", "test"]
    colors = ['deepskyblue', 'orange']
    xys = [(0.1, 0.8), (0.4, 0.1)]
    for i in range(2):
        data_type = types[i]
        color = colors[i]
        xy = xys[i]
        if not os.path.exists(f"{config['name']}_{data_type}.out"):
            continue
        data = np.loadtxt(f"{config['name']}_{data_type}.out")
        min_value = np.min(data)
        max_value = np.max(data)
        index = data.shape[1] // 2
        plt.scatter(data[:, index:], data[:, :index], marker='o', edgecolors=color, facecolors=f'{col}', label=data_type, s=100)        
        axes.plot(np.linspace(min_value, max_value, num=10), np.linspace(min_value, max_value, num=10), '-', color='k')
        rmse = np.sqrt(mean_squared_error(data[:, :index], data[:, index:]))
        r2 = r2_score(data[:, :index], data[:, index:])
        #axes.text(xy[0], xy[1], f'RMSE={1000 * rmse:.3f}(m{config["unit"]})\n$R^2$={r2:.3f}',
                  #transform=axes.transAxes)
        axes.text(xy[0], xy[1], f'RMSE={1000 * rmse:.3f}(m{config["unit"]})',fontsize=20, fontweight='bold',
                  transform=axes.transAxes)
    axes.set_title(f'{tit}',fontsize=20, fontweight='bold')
    handles, labels = axes.get_legend_handles_labels()
    label_dict = dict(zip(labels, handles))
    axes.legend(label_dict.values(), label_dict, frameon=False, ncol=2, columnspacing=1)
    axes.set_xlabel(f'DFT {config["name"]} ({config["unit"]})',fontsize=20, fontweight='bold')
    axes.set_ylabel(f'NEP {config["name"]} ({config["unit"]})',fontsize=20, fontweight='bold')


if __name__ == '__main__':
    out_num = len(glob.glob("*.out"))
    test_out_num = len(glob.glob("*test.out"))
    titles = ['(a)','(b)','(c)','(d)']
    rows = 2 if out_num >= 4 else 1   # 如果 out_num 是 4 或更多，则将 rows 设置为 2
    cols = (out_num - test_out_num) // rows + (out_num - test_out_num) % rows  # 2 

    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    grids = fig.add_gridspec(rows, cols)    # 2 行 2 列的网格布局，可以在其中放置最多4个子图

    if os.path.exists("loss.out"):
        axes_index = 0        
        axes = fig.add_subplot(grids[axes_index])
        axes_index += 1
        plot_loss_result(axes)

    else:
        axes_index = 0
        
    color_1ist = ['red', 'blue','green']
    title_list = ['(b)','(c)','(d)']
    for config in Config:
        if not os.path.exists(f"{config['name']}_train.out"):
            continue
        color = color_1ist[axes_index-1]
        title = title_list[axes_index-1]
        axes = fig.add_subplot(grids[axes_index])
        plot_train_result(axes, config,col=color,tit=title)
        axes_index += 1
        
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)        # 子图与图像边缘的距离
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3) # 子图之间的距离
    plt.savefig("FIG2.png", dpi=300)
