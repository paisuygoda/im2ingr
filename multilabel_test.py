import os
import time
import random
import numpy as np
from data_loader import ImagerLoader
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
import torchwordemb
from args import get_parser
from trijoint import norm
import pickle


class ingr_mult(nn.Module):
    def __init__(self):
        super(ingr_mult, self).__init__()
        if opts.preModel == 'resNet50':

            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
            self.visionMLP = nn.Sequential(*modules)

            self.final_fc = nn.Sequential(
                nn.Linear(opts.imfeatDim, opts.numActiveIngrs),
                nn.Sigmoid(),
            )

        else:
            raise Exception('Only resNet50 model is implemented.')

        if opts.semantic_reg:
            self.semantic_branch = nn.Linear(opts.imfeatDim, opts.numClasses)

    def forward(self, x):  # we need to check how the input is going to be provided to the model

        y = self.visionMLP(x)
        y = y.view(y.size(0), -1)
        y_label = self.final_fc(y)

        if opts.semantic_reg:
            sem_class = self.semantic_branch(y)
            # final output
            output = [y_label, sem_class]
        else:
            # final output
            output = [y_label]
        return output


# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

def main():
    gpus = ','.join(map(str, opts.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model = ingr_mult()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=range(len(opts.gpu)))
    model.final_fc = torch.nn.DataParallel(model.final_fc, device_ids=range(len(opts.gpu)))
    model.cuda()

    # define loss function (criterion) and optimizer
    # cosine similarity between embeddings -> input1, input2, target
    # cosine_crit = nn.CosineEmbeddingLoss(0.1).cuda()
    # cosine_crit = nn.CosineEmbeddingLoss(0.1)
    crit = nn.MultiLabelSoftMarginLoss()

    if opts.semantic_reg:
        weights_class = torch.Tensor(opts.numClasses).fill_(1)
        weights_class[0] = 0  # the background class is set to 0, i.e. ignore
        # CrossEntropyLoss combines LogSoftMax and NLLLoss in one single class
        class_crit = nn.CrossEntropyLoss(weight=weights_class).cuda()
        # we will use two different criteria
        criterion = [crit, class_crit]
    else:
        criterion = crit

    # optimizer - with lr initialized accordingly
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=opts.lr)

    if opts.resume:
        print("Resume not implemented")
        exit(1)

    best_val = float('inf')

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # preparing the training loader
    test_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
                     transforms.Compose([
                         transforms.Scale(256),  # rescale the image keeping the original aspect ratio
                         transforms.CenterCrop(256),  # we get only the center of that rescaled
                         transforms.RandomCrop(224),  # random crop within the center crop
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalize,
                     ]), data_path=opts.data_path, partition='test', sem_reg=opts.semantic_reg,
                     ingrW2V=opts.ingrW2V, multilabel=True),
        batch_size=opts.batch_size, shuffle=True,
        num_workers=opts.workers, pin_memory=True)
    print('Test loader prepared.')

    # run epochs
    test(test_loader, model, criterion)


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cos_losses = AverageMeter()
    if opts.semantic_reg:
        cls_losses = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        if torch.equal(target[0], torch.LongTensor([-1])) or (len(input[0]) < opts.batch_size):
            continue

        input_img = torch.autograd.Variable(input[0]).cuda()

        target_labels = np.zeros((opts.batch_size, opts.numActiveIngrs))
        for j, one_pic in enumerate(input[1]):
            for item in one_pic.long():
                try:
                    target_labels[j][item] = 1.0
                except:
                    pass
            target_labels[j][0] = 0
        ans_label = torch.autograd.Variable(torch.Tensor(target_labels)).cuda()

        # compute output
        output = model(input_img)

        target_var = list()
        for j in range(len(target)-2):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # compute loss
        if opts.semantic_reg:
            target_cls = torch.autograd.Variable(target[1])
            cos_loss = criterion[0](output[0], ans_label, target_var[0])
            cls_loss = criterion[1](output[1], target_cls)
            # combined loss
            loss = opts.cos_weight * cos_loss + opts.cls_weight * cls_loss

            # measure performance and record losses
            cos_losses.update(cos_loss.data, input[0].size(0))
            cls_losses.update(cls_loss.data, input[0].size(0))
        else:
            loss = criterion(output[0], ans_label)
            # measure performance and record loss
            cos_losses.update(loss.data[0], input[0].size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i==0:
            data0 = output[0].data.cpu().numpy()
            data1 = ans_label.data.cpu().numpy()
            data2 = target[-2]
            data3 = target[-1]
        else:
            data0 = np.concatenate((data0,output[0].data.cpu().numpy()),axis=0)
            data1 = np.concatenate((data1,ans_label.data.cpu().numpy()),axis=0)
            data2 = np.concatenate((data2,target[-2]),axis=0)
            data3 = np.concatenate((data3,target[-1]),axis=0)

    if opts.semantic_reg:
        print('* Test loss {losses:.4f}'.format(losses=cos_losses.avg))
    else:
        print('* Test loss {losses:.4f}'.format(losses=cos_losses.avg))

    with open(opts.path_results+'img_embeds.pkl', 'wb') as f:
        pickle.dump(data0, f)
    with open(opts.path_results+'rec_embeds.pkl', 'wb') as f:
        pickle.dump(data1, f)
    with open(opts.path_results+'img_ids.pkl', 'wb') as f:
        pickle.dump(data2, f)
    with open(opts.path_results+'rec_ids.pkl', 'wb') as f:
        pickle.dump(data3, f)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = opts.snapshots + 'model_multilabel_e%03d_v-%.3f.pth.tar' % (state['epoch'], state['best_val'])
    if is_best:
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


if __name__ == '__main__':
    main()
