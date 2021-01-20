import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter

class Selective_kernel(nn.Module):
    def __init__(self, frame=4):
        super(Selective_kernel, self).__init__()
        self.frame = frame
        self.conv1dim_low = nn.Sequential(
            nn.Conv3d(128, 512, kernel_size=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv1dim_mid = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv1dim_high = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.compact_descriptor = nn.Sequential(
            nn.Linear(512, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.compact_high = nn.Sequential(
            nn.Linear(512, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.compact_mid = nn.Sequential(
            nn.Linear(512, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.compact_low = nn.Sequential(
            nn.Linear(512, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # obtain feature for each channel
        self.multihead_high = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=1),
            nn.ReLU(),
        )
        self.multihead_mid = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=1),
            nn.ReLU(),
        )
        self.multihead_low = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=1),
            nn.ReLU(),
        )
        # compute correlation between fusion and invidual
        self.attention_low = nn.Sequential(
            nn.Linear(64, 1),
            # nn.Sigmoid(),
        )
        self.attention_mid = nn.Sequential(
            nn.Linear(64, 1),
            # nn.Sigmoid(),
        )
        self.attention_high = nn.Sequential(
            nn.Linear(64, 1),
            # nn.Sigmoid(),
        )


    def forward(self, low, mid ,high):
        high_resize = F.interpolate(self.conv1dim_high(high), size=[self.frame, 28, 28], mode='nearest')
        low_resize = self.conv1dim_low(low)
        mid_resize = F.interpolate(self.conv1dim_mid(mid), size=[self.frame, 28, 28], mode='nearest')

        fusion = high_resize + mid_resize + low_resize
        fusion_pooling = fusion.mean(-1).mean(-1).mean(-1)
        compact_feature = self.compact_descriptor(fusion_pooling)

        high_pooling = high_resize.mean(-1).mean(-1).mean(-1)
        high_feature = self.compact_high(high_pooling)
        high_feature = self.multihead_high(high_feature.unsqueeze(1))

        mid_pooling = mid_resize.mean(-1).mean(-1).mean(-1)
        mid_feature = self.compact_mid(mid_pooling)
        mid_feature = self.multihead_mid(mid_feature.unsqueeze(1))

        low_pooling = low_resize.mean(-1).mean(-1).mean(-1)
        low_feature = self.compact_low(low_pooling)
        low_feature = self.multihead_low(low_feature.unsqueeze(1))

        compact_feature = torch.cat(512*[compact_feature], dim=1).view(-1, 512, 32)

        high_correlation = torch.cat([high_feature, compact_feature], dim=2)
        high_correlation = self.attention_high(high_correlation).squeeze(-1)
        mid_correlation = torch.cat([mid_feature, compact_feature], dim=2)
        mid_correlation = self.attention_mid(mid_correlation).squeeze(-1)
        low_correlation = torch.cat([low_feature, compact_feature], dim=2)
        low_correlation = self.attention_low(low_correlation).squeeze(-1)

        attention = torch.cat([low_correlation, mid_correlation, high_correlation], dim=0).view(3, -1, 512).transpose(0, 1)
        attention = F.softmax(attention, dim=1)
        output_low = attention[:, 0, :].view(-1, 512, 1, 1, 1) * low_resize
        output_mid = attention[:, 1, :].view(-1, 512, 1, 1, 1) * mid_resize
        output_high = attention[:, 2, :].view(-1, 512, 1, 1, 1) * high_resize
        output = output_low + output_mid + output_high

        return output



class Selective_position(nn.Module):
    def __init__(self, frame=4):
        super(Selective_position, self).__init__()
        self.frame = frame
        self.conv1dim_low = nn.Sequential(
            nn.Conv3d(128, 512, kernel_size=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv1dim_mid = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv1dim_high = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.compact_descriptor = nn.Sequential(
            nn.Linear(28*28, 49),
            # nn.BatchNorm1d(49),
            nn.ReLU(),
        )

        self.compact_high = nn.Sequential(
            nn.Linear(28*28, 49),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.compact_mid = nn.Sequential(
            nn.Linear(28*28, 49),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.compact_low = nn.Sequential(
            nn.Linear(28*28, 49),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.multihead_high = nn.Sequential(
            nn.Conv1d(1, 28*28, kernel_size=1),
            nn.ReLU(),
        )
        self.multihead_mid = nn.Sequential(
            nn.Conv1d(1, 28*28, kernel_size=1),
            nn.ReLU(),
        )
        self.multihead_low = nn.Sequential(
            nn.Conv1d(1, 28*28, kernel_size=1),
            nn.ReLU(),
        )

        self.attention_low = nn.Linear(98, 1)
        self.attention_mid = nn.Linear(98, 1)
        self.attention_high = nn.Linear(98, 1)

    def forward(self, low, mid, high):
        high_resize = F.interpolate(self.conv1dim_high(high), size=[self.frame, 28, 28], mode='nearest')
        low_resize = self.conv1dim_low(low)
        mid_resize = F.interpolate(self.conv1dim_mid(mid), size=[self.frame, 28, 28], mode='nearest')

        fusion = high_resize + mid_resize + low_resize
        fusion_pooling = fusion.mean(1).mean(1).view(-1, 28*28)
        compact_feature = self.compact_descriptor(fusion_pooling)

        high_pooling = high_resize.mean(1).mean(1).view(-1, 28*28)
        high_feature = self.compact_high(high_pooling)
        high_feature = self.multihead_high(high_feature.unsqueeze(1))

        mid_pooling = mid_resize.mean(1).mean(1).view(-1, 28*28)
        mid_feature = self.compact_mid(mid_pooling)
        mid_feature = self.multihead_mid(mid_feature.unsqueeze(1))

        low_pooling = low_resize.mean(1).mean(1).view(-1, 28*28)
        low_feature = self.compact_low(low_pooling)
        low_feature = self.multihead_low(low_feature.unsqueeze(1))

        compact_feature = torch.cat((28*28)*[compact_feature], dim=1).view(-1, 28*28, 49)

        high_correlation = torch.cat([high_feature, compact_feature], dim=2)
        high_correlation = self.attention_high(high_correlation).squeeze(-1)
        mid_correlation = torch.cat([mid_feature, compact_feature], dim=2)
        mid_correlation = self.attention_mid(mid_correlation).squeeze(-1)
        low_correlation = torch.cat([low_feature, compact_feature], dim=2)
        low_correlation = self.attention_low(low_correlation).squeeze(-1)

        attention = torch.cat([low_correlation, mid_correlation, high_correlation], dim=0).view(3, -1, 28*28).transpose(0, 1)
        attention = F.softmax(attention, dim=1)
        output_low = attention[:, 0, :].view(-1, 1, 1, 28, 28) * low_resize
        output_mid = attention[:, 1, :].view(-1, 1, 1, 28, 28) * mid_resize
        output_high = attention[:, 2, :].view(-1, 1, 1, 28, 28) * high_resize
        output = output_low + output_mid + output_high

        return output

class Kernel_Position(nn.Module):
    def __init__(self, frame=4):
        super(Kernel_Position, self).__init__()
        self.kernel = Selective_kernel(frame=frame) # 4 is for 8 frame, 16 is for 32 frame
        self.position = Selective_position(frame=frame)
        # self.alpha = Parameter(torch.ones(1))
        # self.beta = Parameter(torch.ones(1))


    def forward(self, low, mid, high):
        output_kernel = self.kernel(low, mid, high)
        output_position = self.position(low, mid, high)
        output = output_kernel + output_position

        return output


if __name__ == '__main__':
    torch.manual_seed(1)
    print ('haha')
    fake_data_low = Variable(torch.randn(2, 128, 16, 28, 28)).cuda()
    fake_data_mid = Variable(torch.randn(2, 256, 8, 14, 14)).cuda()
    fake_data_high = Variable(torch.randn(2, 512, 4, 7, 7)).cuda()
    net = Kernel_Position(frame=16).cuda()
    output = net(fake_data_low, fake_data_mid, fake_data_high)
    print (output.size())
