import os
import time

from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model
from evaluate import final_test


def train(img_dir, train_file, valid_file, output,
          train_bs, num_workers,
          num_epochs, max_N, lr_list,
          augment=False, opt=None):
    # opt在这个函数里仅表示数据增强的超参数

    start = time.time()

    # 创建保存结果的文件夹
    os.system('rm -rf %s'%output)
    os.makedirs(output)

    # 创建训练用的数据集和dataloader
    train_set = Dataset(img_dir, train_file, augment, opt)
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)

    # 创建模型
    model = Model(opt=opt)

    epoch = 0
    train_acc_list, valid_acc_list = [], []
    for lr in lr_list:
        # 调整学习率
        model.reset_lr(lr)
        print('set lr to %.6f'%lr)

        # 每次调整学习率后，重新计算当前有多少个epoch准确率未上升
        patience = 0

        while True:
            st = time.time()

            # 训练一个完整的epoch
            for idx, (img, gt) in enumerate(train_loader):
                loss = model.train(img, gt)
                print(epoch, idx, loss, end='\r')
            print('')

            # 计算训练集和验证集的准确率
            _, train_acc = final_test(model, img_dir, train_file)
            _, valid_acc = final_test(model, img_dir, valid_file)

            # 保存模型
            model.save(os.path.join(output, 'latest_model.pth'))
            if len(train_acc_list) == 0 or train_acc > max(train_acc_list):
                model.save(os.path.join(output, 'best_model.pth'))

            # 计算已经连续多少个epoch训练集的准确率没有上升了
            if len(train_acc_list) == 0 or train_acc > max(train_acc_list):
                patience = 0
            else:
                patience += 1

            epoch += 1
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

            et = time.time()
            print('train:', train_acc, 'valid:', valid_acc, '%ds'%int(et-st))

            if patience == max_N or epoch >= num_epochs:
                break

    avg_valid_acc = sum(valid_acc_list[-max_N:]) / max_N
    print('average valid accuracy in last %d epochs:'%max_N, avg_valid_acc)

    end = time.time()
    print('time:', '%ds'%int(end - start))

    return avg_valid_acc, model
