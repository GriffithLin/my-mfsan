"""
@author: lin ming
"""
from __future__ import print_function

import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance



#
#####
#


# import torch
# import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
# import data_loader
import mfsan_model as mfsan
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 32
iteration = 10000
lr = [0.001, 0.01]
momentum = 0.9
cuda = True
# seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "./dataset/"
source1_name = "webcam"
source2_name = 'dslr'
target_name = "amazon"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.manual_seed(seed)
# if cuda:
#     torch.cuda.manual_seed(seed)

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
# source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
# target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
# target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    # get_train_transform 把 transform给封装了
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)

    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

# source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
# source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
# train_target_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
# target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

    train_source1_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source[0], args.target, train_transform, val_transform)
    
    train_source2_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source[1], args.target, train_transform, val_transform)

    train_source1_loader = DataLoader(train_source1_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
                                
    train_source2_loader = DataLoader(train_source2_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.workers, drop_last=True)

    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    target_test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    # source1_iter = iter(source1_loader)
    # source2_iter = iter(source2_loader)
    # target_iter = iter(test_loader)

    train_source1_iter = ForeverDataIterator(train_source1_loader)
    train_source2_iter = ForeverDataIterator(train_source2_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)



    # model = models.MFSAN(num_classes=31)
    # if cuda:
    #     model.cuda()

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    # pool_layer = nn.Identity() if args.no_pool else None
    model = mfsan.MFSAN(backbone, num_classes,finetune=not args.scratch).to(device)
    if cuda:
        model.cuda()

    train(model, train_source1_iter, train_source2_iter, train_target_iter, target_test_loader)
    


def train(model, source1_iter, source2_iter, target_iter, target_test_loader):

    correct = 0

    optimizer = torch.optim.SGD(model.get_parameters(args.lr), 
        # 设置其他参数学习率、动量和L2权重衰减
        lr=args.lr, momentum=momentum, weight_decay=l2_decay)
    
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    for i in range(1, iteration + 1):
        model.train()

        # optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        # optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        # optimizer.param_groups[2]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        # optimizer.param_groups[3]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        # optimizer.param_groups[4]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        

        source_data, source_label = next(source1_iter)[:2]
        target_data = next(target_iter)[0]

        if cuda:
            # print(source_data)
            source_data, source_label = source_data.cuda(), source_label.cuda()
            # print(source_data)
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        

        

        optimizer.zero_grad()


        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))



        source_data, source_label = next(source2_iter)[:2]
        target_data = next(target_iter)[0]


        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        
        
        
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=2)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

        # if i % (log_interval * 20) == 0:
            t_correct = test(model, target_test_loader)
            if t_correct > correct:
                correct = t_correct
            print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

def test(model, target_test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        #TODO 第三个返回是啥
        for data, target,_ in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2 = model(data, mark = 0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)

            pred = (pred1 + pred2) / 2
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}'.format(correct1, correct2))
    return correct

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MFSAN for multisource Unsupervised Domain Adaptation')
# dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')

        #√
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')

        
     #action  就是--no-hflip 之后不用指定参数，  存为true  
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    #TLlib的设置
    # parser.add_argument('--train-resizing', type=str, default='default')
    # parser.add_argument('--val-resizing', type=str, default='default')
    # parser.add_argument('--norm-mean', type=float, nargs='+',
    #                     default=(0.485, 0.456, 0.406), help='normalization mean')
    # parser.add_argument('--norm-std', type=float, nargs='+',
    #                     default=(0.229, 0.224, 0.225), help='normalization std')

    #改为mfsan原来算法的设置
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0, 0, 0), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(1, 1, 1), help='normalization std')
    parser.add_argument('--train-resizing', type=str, default='ran.crop')
    parser.add_argument('--val-resizing', type=str, default='res.')

        #这两个参数 和原文不一样,但是指定train-resizing = ran.crop时，用不到
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')

# model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
        #??
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
        #
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
# training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")


    args = parser.parse_args()
    main(args)




    