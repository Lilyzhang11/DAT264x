from torch.utils.data import DataLoader

from dataset import Dataset
from utils import read_csv


def final_test(model, img_dir, label_file, bs=128, num_workers=32):
    # 此函数用输出的模型，对img_dir和label_file描述的图片集合进行测试
    # 最终返回的是预测的结果及和label_file对比的准确率

    # 构建由img_dir和label_file描述的数据集
    dataset = Dataset(img_dir, label_file)
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=num_workers,
                            shuffle=False, pin_memory=True, drop_last=False)

    gt = read_csv(label_file)

    # 对整个数据集进行预测
    result = []
    idx = 0
    for inp, _ in dataloader:
        out = model.test(inp)
        for i in range(out.size(0)):
            result.append([gt[idx][0], str(out[i].item())])
            idx += 1

    # 计算准确率
    n = 0
    for i in range(len(gt)):
        if gt[i][1] == result[i][1]:
            n += 1
    acc = n / len(gt)

    return result, acc
