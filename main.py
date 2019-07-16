from train import train
from evaluate import final_test
from preprocessing import read_csv
import csv
import os
import numpy as np
import torch


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(1)


root = 'data/processed/'
fold = 10
train_bs = 50
valid_bs = 50
num_epochs = 100
max_N = 5
lr_list = [3e-4, 3e-5, 3e-6, 3e-7]


def func(label):
    num = [0] * 3
    for item in label:
        num[item] += 1
    return np.array(num).argmax()


if __name__ == '__main__':

    os.system('rm -rf result/')

    acc_list = []
    for i in range(fold):
        print('%dth fold:'%i)
        train_file = root+'%d-fold/train_%d.csv'%(fold, i)
        valid_file = root+'%d-fold/valid_%d.csv'%(fold, i)
        test_file = root+'submission_format.csv'
        output = 'result/%d/'%i
        '''
        acc, model = train(root+'train/', train_file, valid_file, output,
                           train_bs, valid_bs, num_epochs, max_N,
                           lr_list)
        '''
        acc, model = train(root, train_file, valid_file, output,
                           train_bs, valid_bs, num_epochs, max_N,
                           lr_list,
                           root, test_file)
        acc_list.append(acc)
        model.load(output+'best_model.pth')
        final_test(model, root, test_file, valid_bs, output)
        print('')
    print('total accuracy:', sum(acc_list)/fold)

    data = []
    for i in range(fold):
        data.append(read_csv('result/%d/result.csv'%i))
    result = []
    for i in range(len(data[0])):
        label = []
        for j in range(fold):
            label.append(int(data[j][i][1]))
        label = func(label)
        result.append([data[0][i][0], str(label)])
    with open('result/final_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file_id', 'accent'])
        for item in result:
            writer.writerow(item)
