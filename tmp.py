import csv
from preprocessing import read_csv


root = 'save/result_score_0.8376/'

data = []
for i in range(10):
    file = root + str(i) + '/result.csv'
    data.append(read_csv(file))

n = 0
ret = []
for i in range(len(data[0])):
    flag = True
    label = []
    for j in range(10):
        label.append(int(data[j][i][1]))
    for j in range(1, 10):
        if label[j] != label[0]:
            flag = False
            break
    if flag:
        ret.append([data[0][i][0], str(label[0])])

num = [0] * 3
for item in ret:
    num[int(item[1])] += 1
N = min(num) // 100 * 100
print(N)
new_ret = []
num = [0] * 3
for item in ret:
    if num[int(item[1])] < N:
        new_ret.append(item)
        num[int(item[1])] += 1

for i in range(10):
    data = read_csv('data/processed/10-fold/train_%d.csv'%i)
    data = data + new_ret
    with open('data/processed/10-fold/train_%d.csv'%i, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file_id', 'accent'])
        for item in data:
            writer.writerow(item)
