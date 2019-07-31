import os
import json

import torch

from opt import create_opt
from preprocessing import generate_k_fold
from add_fake_label import add_fake_label
from train import train
from evaluate import final_test
from utils import write_csv, vote


if __name__ == '__main__':

    opt = create_opt()

    # -------------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------------
    # 指明使用第几块显卡
    # 当前代码不支持无显卡环境
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['cuda']

    # 设置随机数的种子
    # 当前代码只使用了pytorch内的随机数
    if opt['seed'] is not None:
        torch.manual_seed(opt['seed'])

    # 创建保存结果的目录
    # 所有生成的文件都将存放在opt['out_dir']里
    os.system('rm -rf %s'%opt['out_dir'])
    os.makedirs(opt['out_dir'])

    # 保存opt
    # 便于复现实验结果
    with open(os.path.join(opt['out_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)

    # -------------------------------------------------------------------------
    # 交叉验证
    # -------------------------------------------------------------------------
    # 准备交叉验证的.csv文件
    generate_k_fold(opt['train_file'], opt['label_dir'], opt['fold'])
    if opt['use_fake_label']:
        add_fake_label(opt['fake_dir'], opt['label_dir'], opt['fold'])

    # 训练
    acc_list, result_list = [], []
    for fold_idx in range(opt['fold']):
        print('%dth fold:'%fold_idx)

        # 准备路径
        output = os.path.join(opt['out_dir'], str(fold_idx))
        train_file = os.path.join(opt['label_dir'], 'train_%d.csv'%fold_idx)
        valid_file = os.path.join(opt['label_dir'], 'valid_%d.csv'%fold_idx)

        # 训练
        acc, model = train(opt['train_dir'], train_file, valid_file, output,
                           opt['train_bs'], opt['num_workers'], 
                           opt['num_epochs'], opt['max_N'], opt['lr_list'],
                           opt['augment'], opt)
        acc_list.append(acc)

        # 保存测试结果
        result, _ = final_test(model, opt['test_dir'], opt['test_file'])
        result_list.append(result)
        write_csv(os.path.join(output, 'result.csv'), result)

        print('')

    # 打印在验证集上的平均准确率
    print('total accuracy:', sum(acc_list) / opt['fold'])

    #投票
    final_result = vote(result_list)

    # 保存投票后的结果
    write_csv(os.path.join(opt['out_dir'], 'final_result.csv'), final_result)
