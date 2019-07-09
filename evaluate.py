import csv
from torch.utils.data import DataLoader
from dataset import Dataset


def generate_result(result, output_path):
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file_id', 'accent'])
        for file_id, accent in result:
            writer.writerow([file_id, accent])


def compute_accuracy(pred_file, gt_file):
    pred = {}
    with open(pred_file, 'r') as f:
        reader = csv.reader(f)
        for idx, (file_id, accent) in enumerate(reader):
            if idx == 0:
                continue
            pred[file_id] = accent

    gt = {}
    with open(gt_file, 'r') as f:
        reader = csv.reader(f)
        for idx, (file_id, accent) in enumerate(reader):
            if idx == 0:
                continue
            gt[file_id] = accent

    n, N = 0, 0
    for key in pred:
        if pred[key] == gt[key]:
            n += 1
        N += 1

    return n / N


def final_test(model, img_dir, label_file, bs, output):
    test_set = Dataset(img_dir, label_file)
    loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=int(bs*1.5),
                        pin_memory=True, drop_last=False)
    pred, _ = model.test(loader, label_file)
    with open(output+'result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file_id', 'accent'])
        for item in pred:
            writer.writerow(item)


if __name__ == '__main__':

    print(compute_accuracy('data/processed/train.csv', 'data/processed/train.csv'))
