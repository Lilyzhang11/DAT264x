import os

from utils import read_csv, write_csv


def get_fake_label(fake_list):
    # 读取fake_list里指向的.csv文件
    fake_data = []
    for fake_file in fake_list:
        fake_data.append(read_csv(fake_file))

    # 获取fake_data里预测结果全一致的样本
    fake_label = []
    for i in range(len(fake_data[0])):
        equal = True
        for j in range(1, len(fake_data)):
            if fake_data[0][i] != fake_data[j][i]:
                equal = False
                break
        if equal:
            fake_label.append(fake_data[0][i])

    return fake_label


def count(data):
    # 计算三类图片分别有多少张图片
    # 输入的数据格式应该和read_csv读取到的list的格式相同
    num = [0] * 3
    for _, accent in data:
        num[int(accent)] += 1
    return num


def add_fake_label(fake_dir, label_dir, fold):
    #     本函数将读取某次训练的结果，将其k次交叉验证的预测结果完全相同的样本
    # 当作是有标记样本，加入到新的交叉验证的train_*.csv中

    # fake_dir的目录结构:
    # fake_dir-0-result.csv
    #         -1-result.csv
    #         -2-result.csv
    #         ......

    # label_dir的目录结构:
    # label_dir-train_0.csv
    #          -train_1.csv
    #          -train_2.csv
    #          ......

    fake_list = []
    for i in range(fold):
        fake_list.append(os.path.join(fake_dir, str(i), 'result.csv'))

    fake_label = get_fake_label(fake_list)

    # 截取整百个
    num_each_accent = min(count(fake_label)) // 100 * 100

    # new_label保存的是截取整百个，而且各个类别数量相等的数据
    num = [0] * 3
    new_label = []
    for file_id, accent in fake_label:
        if num[int(accent)] < num_each_accent:
            new_label.append([file_id, accent])
            num[int(accent)] += 1

    # 将new_label加入到label_dir中的train_*.csv中
    for i in range(fold):
        target = os.path.join(label_dir, 'train_%d.csv'%i)
        label = read_csv(target)
        label = label + new_label
        write_csv(target, label)
