import torch
import torchvision
from preprocessing import read_csv


class Model(object):

    def __init__(self, lr):
        '''
        self.net = torch.nn.Sequential(
            torch.nn.Linear(128*173, 3),
            # torch.nn.ReLU(),
            # torch.nn.Linear(1024, 3)
        ).cuda()
        '''
        self.net = resnet18(3).cuda()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def reset_lr(self, lr):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def train(self, inp, gt):
        inp, gt = inp.cuda(), gt.cuda()
        # inp = inp.view(-1, 128*173)
        out = self.net(inp)
        loss = (out - gt).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sparse_loss(self, y):
        y_ = (y == y.max(dim=1, keepdim=True)[0]).float()
        return (y - y_).pow(2).mean()

    def test(self, loader, label_file):
        self.net.eval()
        with torch.no_grad():
            # 读取gt
            gt = read_csv(label_file)

            # 预测
            pred = []
            n = 0
            for idx, (img, _) in enumerate(loader):
                img = img.cuda()
                out = self.net(img)
                out = out.argmax(dim=1)
                for i in range(out.size(0)):
                    pred.append([gt[n][0], str(out[i].item())])
                    n += 1
                # print(idx)

            # 计算准确率
            n = 0
            for i in range(len(gt)):
                if (gt[i][1] == pred[i][1]):
                    n += 1
            acc = n / len(gt)
        self.net.train()

        return pred, acc

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


def alexnet(k):
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=k, bias=True)
    return model


def resnet18(k):
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=k, bias=True)
    return model


def resnet34(k):
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=k, bias=True)
    return model


def resnet50(k):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=k, bias=True)
    return model


def resnet101(k):
    model = torchvision.models.resnet101(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=k, bias=True)
    return model


def resnet152(k):
    model = torchvision.models.resnet152(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=k, bias=True)
    return model


def squeezenet1_1(k):
    model = torchvision.models.squeezenet1_1(pretrained=True)
    model.classifier[1] = torch.nn.Conv2d(512, k, kernel_size=(1,1), stride=(1,1))
    return model


if __name__ == '__main__':

    '''
    model = get_model()
    x = torch.rand(100, 3, 128, 173)
    y = model(x)
    print(y.size())
    '''

    inp = torch.rand(10, 1, 128, 173)
    gt = torch.rand(10, 3)
    model = Model()
    print(model.train(inp, gt))
    print(model.test(['data/processed/train/10000.png']))
