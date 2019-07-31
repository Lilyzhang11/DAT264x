import torch
import torchvision


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


def densenet121(k):
    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(in_features=1024, out_features=k, bias=True)
    return model


def shufflenet_v2_x0_5(k):
    model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
    model.fc = torch.nn.Linear(in_features=1024, out_features=k, bias=True)
    return model


def resnext50_32x4d(k):
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=k, bias=True)
    return model


def mobilenet_v2(k):
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=k, bias=True)
    return model


def resnet18_dropout(k):
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=512, out_features=k, bias=True)
    )
    return model


def create_net(name):
    if name == 'alexnet':
        return alexnet(3)
    elif name == 'resnet18':
        return resnet18(3)
    elif name == 'resnet34':
        return resnet34(3)
    elif name == 'resnet50':
        return resnet50(3)
    elif name == 'resnet101':
        return resnet101(3)
    elif name == 'resnet152':
        return resnet152(3)
    elif name == 'squeezenet1_1':
        return squeezenet1_1(3)
    elif name == 'densenet121':
        return densenet121(3)
    elif name == 'shufflenet_v2_x0_5':
        return shufflenet_v2_x0_5(3)
    elif name == 'resnext50_32x4d':
        return resnext50_32x4d(3)
    elif name == 'mobilenet_v2':
        return mobilenet_v2(3)
    elif name == 'resnet18_dropout':
        return resnet18_dropout(3)
    else:
        assert(False)
