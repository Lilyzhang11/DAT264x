import os

from utils import read_csv, write_csv


def classified_by_label(label_file):
    # 将label_file按照类别进行分类
    image_name = [[], [], []]
    data = read_csv(label_file)
    for file_id, accent in data:
        image_name[int(accent)].append(file_id)
    # 此时image_name的格式是：
    # [['10000', ...], ['10001', ...], ['10002', ...]]
    return image_name


def generate_k_fold(label_file, output_dir, fold):
    # 用label_file进行fold次划分，用于交叉验证
    # 每次划分，都有1/fold的样本作为验证集
    # 划分后的train_*.csv和valid_*.csv保存在output_dir中

    image_name = classified_by_label(label_file)

    # 1折的大小
    size = len(image_name[0]) // fold

    # 生成保存k-fold的.csv的文件夹
    os.system('rm -rf %s'%output_dir)
    os.makedirs(output_dir)

    # 生成k-fold
    for i in range(fold):
        train_list, valid_list = [], []
        for j in range(3):
            for k, file_id in enumerate(image_name[j]):
                if k in range(i * size, (i+1) * size):
                    valid_list.append([file_id, str(j)])
                else:
                    train_list.append([file_id, str(j)])

        write_csv(os.path.join(output_dir, 'train_%d.csv'%i), train_list)
        write_csv(os.path.join(output_dir, 'valid_%d.csv'%i), valid_list)


if __name__ == '__main__':

    def check_order(data):
        image_name = [[], [], []]
        for file_id, accent in data:
            image_name[int(accent)].append(int(file_id))
        for i in range(3):
            for j in range(len(image_name[i]) - 1):
                assert(image_name[i][j+1] > image_name[i][j])

    label_file = 'data/processed/train_labels.csv'
    output_dir = 'data/processed/10-fold/'
    fold = 10

    generate_k_fold(label_file, output_dir, fold)

    # 检查
    for idx in range(fold):
        train_data = read_csv(os.path.join(output_dir, 'train_%d.csv'%idx))
        check_order(train_data)
        valid_data = read_csv(os.path.join(output_dir, 'valid_%d.csv'%idx))
        check_order(valid_data)
        data1 = train_data + valid_data
        data1.sort()
        data2 = read_csv(label_file)
        assert(len(train_data) + len(valid_data) == len(data2))
        for i in range(len(data1)):
            assert(data1[i][0] == data2[i][0])
            assert(data1[i][1] == data2[i][1])
