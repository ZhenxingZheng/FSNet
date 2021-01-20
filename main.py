from opts import parser
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
from Models import Net
import torch.backends.cudnn as cudnn
import time
import os
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import math
from GradualWarmupScheduler import GradualWarmupScheduler
import r2plus1d
from tqdm import tqdm


def main():
    global args, best_prec1
    cudnn.benchmark = True
    args = parser.parse_args()


    if not os.path.exists (args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(args.score_dir):
        os.mkdir(args.score_dir)
    strat_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log = open(os.path.join(args.log_dir, strat_time + '.txt'), 'w')
    print (args.description)
    log.write(args.description + '\n')
    log.flush()
    print ('=======================Experimental Settings=======================\n')
    log.write('=======================Experimental Settings=======================\n')
    log.flush()
    print ('using_Dataset:{0}  stride:{1}  epochs:{2}  frames:{3}'.format(args.dataset, args.stride, args.epoch, args.frames))
    log.write('using_Dataset:{0}  stride:{1}  epochs:{2}  frames:{3}'.format(args.dataset, args.stride, args.epoch, args.frames) + '\n')
    log.flush()
    print ('===================================================================\n')
    log.write('===================================================================\n')
    log.flush()

    from dataset import VideoDataset

    train_loader = torch.utils.data.DataLoader(
            VideoDataset(root=args.root, list=args.train_video_list, num_frames=args.frames,
                         time_stride=args.stride, split='train'),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        VideoDataset(root=args.root, list=args.test_video_list, num_frames=args.frames,
                     time_stride=args.stride, split='val'),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        VideoDataset(root=args.root, list=args.test_video_list, num_frames=args.frames,
                     time_stride=args.stride, split='3crop'),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # for data, label in test_loader:
    #     print(data.size())
    #     break

    net = Net(dataset=args.dataset)
    net = torch.nn.DataParallel(net).cuda()


    # print(net.module.net.aggregation)


    if args.cross:
        net.load_state_dict(torch.load('./model/2020-06-06 09:16:4523.pkl'))
        print('loading weights from 2020-06-06 09:16:4523!')
        if args.dataset == 'hmdb':
            num_class = 51
        if args.dataset == 'ucf':
            num_class = 101
        net.module.net.fc = nn.Linear(512, num_class, bias=False).cuda()


    if args.get_scores:
        net.load_state_dict(torch.load('./model/2020-06-16 08:23:33_best.pkl'))
        print ('loading weights successfully 2020-06-16 08:23:33_best')


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=net.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=8)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=8, after_scheduler=scheduler_cosine)
    description_list = 'the testing list is' + args.test_video_list
    print (description_list)
    log.write(description_list)

    best_prec1 = 0
    for epoch in range(args.epoch):
        if not args.get_scores:
            scheduler_cosine.step()
            description_learning = 'the learning rate is ' + str(optimizer.state_dict()['param_groups'][0]['lr'])
            print (description_learning)
            train(train_loader, net, criterion, optimizer, epoch, log, args)
            # torch.save(net.state_dict(), os.path.join(args.model_dir, strat_time + str(epoch) + '.pkl'))

            if epoch % args.eval_freq == 0:
                with torch.no_grad():
                    prec1 = test(val_loader, net, epoch, log, args)
                    if prec1 > best_prec1:
                        best_prec1 = prec1
                        torch.save(net.state_dict(), os.path.join(args.model_dir, strat_time + '_best.pkl'))
                        # df = pd.DataFrame(mat[1:])
                        # df.to_excel(os.path.join(args.score_dir, strat_time + '.xlsx'))

        else:
            with torch.no_grad():
                print ('Begin Get Scores')
                gets(test_loader, net, epoch, args)
                print ('Done')
                break


def train(train_loader, net, criterion, optimizer, epoch, log, args):
    net.train()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()

    start = time.time()
    for step, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        output = net(input)
        loss = criterion(output, target)
        prec1, prec5 = compute_accuracy(output.data, target.data, topk=(1, 5))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        loss = loss / args.accumulation_steps
        loss.backward()

        losses.update(loss.item(), input.size(0))
        if ((step + 1) % args.accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() -start)
        start = time.time()

        if (step + 1) % args.print_freq == 0:
            NowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            output = ('Now Time {0}  Epoch:{1} || Step:{2}'
                      ' || Loss:{loss.avg:.4f}'
                      ' || Time:{batch_time.avg:.3f}'.format(NowTime, epoch, step + 1, loss=losses, batch_time=batch_time))
            print (output)
            log.write(output + '\n')
            log.flush()

    accuracy = ('Epoch:{0} || Prec@1: {top1.avg:.3f} || Prec@5: {top5.avg:.3f}').format(epoch + 1, top1=top1, top5=top5)
    print (accuracy)
    log.write(accuracy + '\n')
    log.flush()



def test(val_loader, net, epoch, log, args):
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if args.dataset == 'hmdb':
        mat = np.zeros((1, 51))
    elif args.dataset == 'ucf':
        mat = np.zeros((1, 101))

    for input, target in tqdm(val_loader):

        input = input.cuda()
        target = target.cuda()
        output = net(input)
        prec1, prec5 = compute_accuracy(output.data, target.data, topk=(1, 5))
        # mat = np.vstack((mat, output.cpu().data.view(1, -1).numpy()))

        top1.update(prec1)
        top5.update(prec5)


    NowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    accuracy = (
    'Testing Phrase ==>> Now Time {0} Epoch:{1} || Best Accuracy:{2} || Prec@1: {top1.avg:.3f} || Prec@5: {top5.avg:.3f}').format(
        NowTime, epoch, max(best_prec1, top1.avg), top1=top1, top5=top5, )
    print (accuracy)
    log.write(accuracy + '\n')
    log.flush()
    return top1.avg


def gets(test_loader, net, epoch, args):
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if args.dataset == 'hmdb':
        mat = np.zeros((1, 51))
    elif args.dataset == 'ucf':
        mat = np.zeros((1, 101))
    elif args.dataset == 'kinetics':
        mat = np.zeros((1, 400))

    for step, (input, target) in enumerate(test_loader):
        input = Variable(input.squeeze(0)).cuda()
        target = Variable(target).cuda(async=True)

        output = net(input)
        output = torch.mean(output, dim=0, keepdim=True)
        prec1, prec5 = compute_accuracy(output.data, target.data, topk=(1, 5))
        mat = np.vstack((mat, output.cpu().data.view(1, -1).numpy()))
        print ('The Testing Number {0} is {1} and overall accuracy is {top1.avg:.3f}'.format(step, prec1, top1=top1))

        top1.update(prec1)
        top5.update(prec5)
    NowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    accuracy = ('Testing Phrase ==>> Now Time {0} Epoch:{1} || Best Accuracy:{2} || Prec@1: {top1.avg:.3f} || Prec@5: {top5.avg:.3f}').format(NowTime, epoch, max(best_prec1, top1.avg), top1=top1, top5=top5, )
    print (accuracy)
    df = pd.DataFrame(mat[1:])
    df.to_excel(args.description + '.xlsx')



def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1)
    corrrect = pred.eq(target.view(-1, 1).expand_as(pred))

    store = []
    for k in topk:
        corrrect_k = corrrect[:,:k].float().sum()
        store.append(corrrect_k * 100.0 / batch_size)
    return store


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    if epoch in args.lr_step:
        args.learning_rate = args.learning_rate * 0.1

    # lr = 0.5 * (1 + math.cos(epoch * math.pi / args.epoch)) * args.learning_rate
    # lr = lr * 0.1 ** (epoch // lr_step)
    print ('the learning rate is changed to {0}'.format(args.learning_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate

# def adjust_learning_rate(optimizer, epoch, args):
#     lr = 0.5 * (1 + math.cos(epoch * math.pi / args.epoch)) * args.learning_rate
#     print ('the learning rate is changed to {0}'.format(lr))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


if __name__ == '__main__':
    main()
