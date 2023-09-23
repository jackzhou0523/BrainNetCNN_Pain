import torch
import torch.nn.functional as F

from read_dataset import train, test_loader


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, example, bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, example, num_classes=3):
        super(BrainNetCNN, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(1, 46, example, bias=True)
        self.e2econv2 = E2EBlock(46, 92, example, bias=True)
        self.E2N = torch.nn.Conv2d(92, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 512, (self.d, 1))
        self.dense1 = torch.nn.Linear(512,256)
        # self.bn1 = torch.nn.BatchNorm1d(num_features=256)
        self.dense2 = torch.nn.Linear(256,32)
        # self.dense2_5 = torch.nn.Linear(64, 32)
        self.dense3 = torch.nn.Linear(32,2)


    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x))
        out = F.leaky_relu(self.e2econv2(out))
        out = F.leaky_relu(self.E2N(out))
        out = F.dropout(F.leaky_relu(self.N2G(out)), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out)), p=0.5)
        # out = self.dense1(out)
        # out = F.dropout(F.leaky_relu(self.bn1(out)), p=0.5)
        # out = F.dropout(torch.tanh(self.dense2(out)), p=0.5)
        # out = self.dense2(out)
        out = F.dropout(F.leaky_relu(self.dense2(out)), p=0.5)
        # out = F.dropout(torch.tanh(self.dense2_5(out)), p=0.5)
        # out = F.softmax(self.dense3(out), dim=1)
        out = F.leaky_relu(self.dense3(out))
        # out = F.dropout(self.dense3(out), p=0.5)
        return out

# from torchsummary import summary
# summary(net, input_size=(1, 108, 108))

if __name__ == '__main__':
    train_features, train_labels = next(iter(test_loader))
    model = BrainNetCNN(train_features)
    input = torch.ones(8,1,108,108)
    output = model(input)
    print(output.shape)


# from torchsummary import summary
# summary(model, (3, 224, 224))