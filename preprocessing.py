import os
import csv


# 存放解压后的两个文件夹及两个.csv文件的目录
root = 'data/processed/'

# k-fold交叉验证
fold = 10


def read_csv(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for idx, (file_id, accent) in enumerate(reader):
            if idx == 0:
                continue
            data.append([file_id, accent])
    return data


if __name__ == '__main__':

    os.system('rm -rf data/processed/classified_by_label/')

    # 把类别为i(i=1,2,3)的图片复制到root/classified_by_label/i/下
    for i in range(3):
        os.makedirs(os.path.join(root, 'classified_by_label', str(i)), exist_ok=True)
    with open(os.path.join(root, 'train_labels.csv'), 'r') as f:
        reader = csv.reader(f)
        for idx, (file_id, accent) in enumerate(reader):
            if idx == 0:
                continue
            if int(file_id) < 20000:
                source_path = os.path.join(root, 'train', file_id+'.png')
            else:
                source_path = os.path.join(root, 'test', file_id+'.png')
            target_path = os.path.join(root, 'classified_by_label', accent, file_id+'.png')
            os.system('cp %s %s'%(source_path, target_path))
            print(idx)

    # 生成5-fold的训练集和验证集
    image_name = []
    for i in range(3):
        target_dir = os.path.join(root, 'classified_by_label', str(i))
        name_list = os.listdir(target_dir)
        name_list.sort()
        # random.shuffle(name_list)
        image_name.append(name_list)
    size = len(image_name[0])
    os.makedirs(os.path.join(root, '%d-fold'%fold), exist_ok=True)
    for i in range(fold):
        train_list = []
        valid_list = []
        for j in range(3):
            for k in range(size):
                if k in range(i * (size//fold), (i+1) * (size//fold)):
                    valid_list.append([image_name[j][k][:-4], str(j)])
                else:
                    train_list.append([image_name[j][k][:-4], str(j)])
        with open(os.path.join(root, '%d-fold'%fold, 'train_%d.csv'%i), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['file_id', 'accent'])
            for item in train_list:
                writer.writerow(item)
        with open(os.path.join(root, '%d-fold'%fold, 'valid_%d.csv'%i), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['file_id', 'accent'])
            for item in valid_list:
                writer.writerow(item)

    # 验证
    '''
    for i in range(fold):
        train_data = read_csv(os.path.join(root, '%d-fold'%fold, 'train_%d.csv'%i))
        valid_data = read_csv(os.path.join(root, '%d-fold'%fold, 'valid_%d.csv'%i))
        data1 = train_data + valid_data
        data1.sort()
        data2 = read_csv(os.path.join(root, 'train_labels.csv'))
        for i in range(len(data1)):
            assert(data1[i][0] == data2[i][0]), (data1[i][0], data2[i][0])
            assert(data1[i][1] == data2[i][1]), (data1[i][1], data2[i][1])
    '''
