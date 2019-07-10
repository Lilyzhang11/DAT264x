import os
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Model
import time
import torch


def train(img_dir, train_file, valid_file, output,
          train_bs, valid_bs, num_epochs, max_N,
          lr_list,
          test_dir=None, test_file=None):
    os.system('rm -rf %s'%output)
    os.makedirs(output)
    train_set = Dataset(img_dir, train_file, True)
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True,
                              num_workers=int(train_bs*1.5), pin_memory=True,
                              drop_last=True)
    v_train_set = Dataset(img_dir, train_file)
    v_loader_train = DataLoader(v_train_set, batch_size=valid_bs, shuffle=False,
                                num_workers=int(valid_bs*1.5), pin_memory=True,
                                drop_last=False)
    valid_set = Dataset(img_dir, valid_file)
    v_loader_valid = DataLoader(valid_set, batch_size=valid_bs, shuffle=False,
                                num_workers=int(valid_bs*1.5), pin_memory=True,
                                drop_last=False)
    if test_dir is not None and test_file is not None:
        test_set = Dataset(test_dir, test_file, True)
        test_loader = DataLoader(test_set, batch_size=train_bs, shuffle=True,
                                 num_workers=int(train_bs*1.5), pin_memory=True,
                                 drop_last=True)
    model = Model(lr_list[0])
    lr_idx = 0

    valid_acc_list = []
    for epoch in range(num_epochs):
        st = time.time()
        # train
        '''
        for idx, (img, gt) in enumerate(train_loader):
            loss = model.train(img, gt)
            print(epoch, idx, loss, end='\r')
            # break
        '''
        idx = 0
        train_iter = iter(train_loader)
        test_iter = iter(test_loader)
        while True:
            try:
                # if torch.rand(1).item() < 0.5 or lr_idx == 0:
                if True:
                    img, gt = next(train_iter)
                    loss = model.train(img, gt)
                else:
                    img, _ = next(test_iter)
                    loss = model.train(img)
            except StopIteration:
                break
            print(epoch, idx, loss, end='\r')
            idx += 1
        print('')
        # valid
        _, train_acc = model.test(v_loader_train, train_file)
        _, valid_acc = model.test(v_loader_valid, valid_file)
        et = time.time()
        print('train:', train_acc, 'valid:', valid_acc, '%ds'%int(et-st))
        # save
        model.save(output+'latest_model.pth')
        if len(valid_acc_list) == 0 or valid_acc > max(valid_acc_list):
            N = 0
            model.save(output+'best_model.pth')
        else:
            N += 1
            if N == max_N:
                if lr_idx == len(lr_list)-1:
                    break
                else:
                    N = 0
                    lr_idx += 1
                    model.load(output+'best_model.pth')
                    model.reset_lr(lr_list[lr_idx])
                    print('reset lr to %f'%lr_list[lr_idx])
        valid_acc_list.append(valid_acc)

    return max(valid_acc_list), model


if __name__ == '__main__':

    train_bs = 16
    valid_bs = 16

    num_epochs = 100

    train_set = Dataset('data/processed/train/', 'data/processed/5-fold/train_0.csv')
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True,
                              num_workers=int(train_bs*1.5), pin_memory=True, drop_last=True)
    train_test_loader = DataLoader(train_set, batch_size=valid_bs, shuffle=False,
                                   num_workers=int(valid_bs*1.5), pin_memory=True, drop_last=False)
    valid_set = Dataset('data/processed/train/', 'data/processed/5-fold/valid_0.csv')
    valid_loader = DataLoader(valid_set, batch_size=valid_bs, shuffle=False,
                              num_workers=int(valid_bs*1.5), pin_memory=True, drop_last=False)

    model = Model()

    valid_acc_list = []
    for epoch in range(num_epochs):

        for idx, (img, label) in enumerate(train_loader):
            loss = model.train(img, label)
            print(epoch, idx, loss)
        print()

        _, train_acc = model.test(train_test_loader, 'data/processed/5-fold/train_0.csv')
        _, valid_acc = model.test(valid_loader, 'data/processed/5-fold/valid_0.csv')
        print('%d epoch, train: %.4f, valid: %.4f'%(epoch, train_acc, valid_acc))

        if len(valid_acc_list) == 0 or valid_acc > max(valid_acc_list):
            N = 0
            model.save('best_model.pth')
        else:
            N += 1
            if N == 5:
                break
        valid_acc_list.append(valid_acc)
        model.save('latest_model.pth')

    print('train finished.')
