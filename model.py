import torch

from net import create_net


class Model(object):

    def __init__(self, lr=3e-4, opt={}):
        # 如果opt中缺少某个关键字，就将default_opt中的值放入opt中
        default_opt = self.create_default_opt()
        for key in default_opt:
            if key not in opt:
                opt[key] = default[key]
        self.opt = opt

        self.net = create_net(self.opt['net']).cuda()

        self.loss_func = self.set_loss_func(self.opt)

        self.optimizer = self.create_optimizer(lr, opt)

        # 根据测试结果，猜测测试集合的三个类别比例是1:0.75:1
        # 但将这个作为loss的权重以后，准确率并没有提升
        # self.weight = torch.tensor([1., 0.75, 1.]).cuda()

    def create_default_opt(self):
        # 构建默认的opt
        opt = {}
        opt['net']          = 'resnet18'
        opt['loss']         = 'cross_entropy'
        opt['optimizer']    = 'adam'
        opt['weight_decay'] = 0.
        return opt

    # -------------------------------------------------------------------------
    # loss相关
    # -------------------------------------------------------------------------
    def set_loss_func(self, opt):
        # 根据opt['loss']设置损失函数
        if opt['loss'] == 'mse':
            return self.mse
        elif opt['loss'] == 'cross_entropy':
            return self.cross_entropy
        else:
            assert(False), 'loss function should be mse/cross_entropy'

    def mse(self, x, y):
        # 均方差损失函数
        # x和y都用one-hot的向量表示
        return (x - y).pow(2).mean()

    def cross_entropy(self, x, y):
        # 交叉熵损失函数
        # x和y都用one-hot的向量表示
        y = y.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(x, y)

        # 加入weight是因为观察到测试集的三类图片数量可能不相等
        # 因为实验结果无效，所以弃用
        # loss = torch.nn.functional.cross_entropy(x, y, self.weight)

        return loss

    # 用于无监督的稀疏性loss，弃用
    # def sparse_loss(self, y):
    #     y_ = (y == y.max(dim=1, keepdim=True)[0]).float()
    #     return (y - y_).pow(2).mean()

    # -------------------------------------------------------------------------
    # 优化器相关
    # -------------------------------------------------------------------------
    def create_optimizer(self, lr, opt):
        # 根据opt['optimizer']创建优化器
        if opt['optimizer'] == 'adam':
            return torch.optim.Adam(self.net.parameters(), lr=lr,
                                    weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'rmsprop':
            return torch.optim.RMSprop(self.net.parameters(), lr=lr,
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'sgd':
            return torch.optim.SGD(self.net.parameters(), lr=lr,
                                   weight_decay=opt['weight_decay'])
        else:
            assert(False), 'optimizer should be adam/rmsprop/sgd'

    def reset_lr(self, lr):
        # 重置优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # -------------------------------------------------------------------------
    # 训练和测试的接口
    # -------------------------------------------------------------------------
    def train(self, inp, gt=None):
        # 考虑到可以使用稀疏性loss训练，所以让gt可以为None
        # 但实验效果是该loss没有作用，所以目前gt不能为None

        self.net.train()

        inp = inp.cuda()

        out = self.net(inp)

        if gt is None:
            # 实验效果是稀疏性loss没有作用，所以不应该进入此分支
            # loss = self.sparse_loss(out)
            assert(False), 'sparse_loss is not being used'
        else:
            gt = gt.cuda()
            loss = self.loss_func(out, gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, inp):
        # 此函数内部已经实现了gpu和cpu之间的数据传递
        # 输入和输出均是cpu上的数据
        self.net.eval()
        with torch.no_grad():
            out = self.net(inp.cuda())
            out = out.cpu().argmax(dim=1)
        return out

    # -------------------------------------------------------------------------
    # 保存和加载模型的接口
    # -------------------------------------------------------------------------
    def save(self, path):
        # 此函数不会自己创建文件夹
        # path路径中的文件夹需要已经存在
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
