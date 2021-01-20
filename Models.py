import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.autograd import Variable
import r2plus1d

class Net(nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.dataset = dataset

        if self.dataset == 'hmdb':
            self.num_class = 51
        elif self.dataset == 'ucf':
            self.num_class = 101
        elif self.dataset == 'kinetics':
            self.num_class = 400

        # the plain IG65 models 359 (32 clips), and 487 (8 clips) classes.
        self.net = r2plus1d.r2plus1d_34_32_kinetics(num_classes=400)
        self.prepare_basemodel(pretrained_weights='./data/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth')


    def forward(self, x):
        output = self.net(x)

        return output

    def prepare_basemodel(self, pretrained_weights):
        state_dicts = torch.load(pretrained_weights)
        self.net.load_state_dict(state_dicts, strict=False)
        print ('loading weights from r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth')

        # for keys, values in state_dicts.items():
        #     if 'num_batches_tracked' in keys:
        #         state_dicts[keys] = torch.tensor([0])


        # self.net.load_state_dict(state_dicts, strict=False)
        # print ('loading weights from '+ pretrained_weights)
        # self.net.fc = nn.Linear(in_features=512, out_features=self.num_class, bias=True)
        # print ('finetuned to ' + self.dataset)



if __name__ == '__main__':
    print('hahaha')
    fake_data = torch.randn(2, 3, 32, 112, 112)
    net = Net(dataset='kinetics')
    print (net(fake_data).size())


