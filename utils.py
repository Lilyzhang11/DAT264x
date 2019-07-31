import csv

import numpy as np


def read_csv(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for idx, (file_id, accent) in enumerate(reader):
            if idx == 0:
                continue
            data.append([file_id, accent])
    return data


def write_csv(file_path, data):
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file_id', 'accent'])
        for item in data:
            writer.writerow(item)


def vote(results):
    # 对多次交叉验证的结果进行投票
    final_result = []
    for img_idx in range(len(results[0])):
        file_id = results[0][img_idx][0]

        # 通过投票得到accent
        num = [0] * 3
        for fold_idx in range(len(results)):
            label = int(results[fold_idx][img_idx][1])
            num[label] += 1
        accent = np.array(num).argmax()

        final_result.append([file_id, accent])
    return final_result


def compute_diff(file1, file2):
    # 计算两个.csv文件中有多少label是相同的
    # 返回相同的比例
    data1 = read_csv(file1)
    data2 = read_csv(file2)

    n = 0
    for i in range(len(data1)):
        if data1[i][1] == data2[i][1]:
            n += 1

    return n / len(data1)


if __name__ == '__main__':

    print(compute_diff(
        'save/0.9126-0.9234/final_result.csv',
        'result/final_result.csv'
    ))
