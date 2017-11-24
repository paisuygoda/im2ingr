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

class ingr_mult(nn.Module):
    def __init__(self):
        super(ingr_mult, self).__init__()
        if opts.preModel == 'resNet50':

            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
            self.visionMLP = nn.Sequential(*modules)

            self.visual_embedding = nn.Sequential(
                nn.Linear(opts.imfeatDim, opts.embDim),
                nn.Tanh(),
            )

            self.recipe_embedding = nn.Sequential(
                nn.Linear(opts.irnnDim * 2 + opts.srnnDim, opts.embDim, opts.embDim),
                nn.Tanh(),
            )

        else:
            raise Exception('Only resNet50 model is implemented.')

        self.stRNN_ = stRNN()
        self.ingRNN_ = ingRNN()
        self.table = TableModule()

        if opts.semantic_reg:
            self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, x, y1, y2, z1, z2):  # we need to check how the input is going to be provided to the model
        # recipe embedding
        recipe_emb = self.table([self.stRNN_(y1, y2), self.ingRNN_(z1, z2)], 1)  # joining on the last dim
        recipe_emb = self.recipe_embedding(recipe_emb)
        recipe_emb = norm(recipe_emb)

        # visual embedding
        visual_emb = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)

        if opts.semantic_reg:
            visual_sem = self.semantic_branch(visual_emb)
            recipe_sem = self.semantic_branch(recipe_emb)
            # final output
            output = [visual_emb, recipe_emb, visual_sem, recipe_sem]
        else:
            # final output
            output = [visual_emb, recipe_emb]
        return output

        # Tweaked im2recipe model for ingredient retrieval


# =============================================================================
parser = get_parser()
opts = parser.parse_args()


# =============================================================================

def main():
    gpus = ','.join(map(str, opts.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model = ingr_mult()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=range(len(opts.gpu)))
    model.cuda()

    # define loss function (criterion) and optimizer
    # cosine similarity between embeddings -> input1, input2, target
    cosine_crit = nn.CosineEmbeddingLoss(0.1).cuda()
    # cosine_crit = nn.CosineEmbeddingLoss(0.1)
    if opts.semantic_reg:
        weights_class = torch.Tensor(opts.numClasses).fill_(1)
        weights_class[0] = 0  # the background class is set to 0, i.e. ignore
        # CrossEntropyLoss combines LogSoftMax and NLLLoss in one single class
        class_crit = nn.CrossEntropyLoss(weight=weights_class).cuda()
        # we will use two different criteria
        criterion = [cosine_crit, class_crit]
    else:
        criterion = cosine_crit

    # # creating different parameter groups
    vision_params = list(map(id, model.visionMLP.parameters()))
    base_params = filter(lambda p: id(p) not in vision_params, model.parameters())

    # optimizer - with lr initialized accordingly
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.visionMLP.parameters(), 'lr': opts.lr * opts.freeVision}
    ], lr=opts.lr * opts.freeRecipe)

    if opts.resume:
        if os.path.isfile(opts.resume):
            print("=> loading checkpoint '{}'".format(opts.resume))
            checkpoint = torch.load(opts.resume)
            opts.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opts.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            best_val = float('inf')
    else:
        best_val = float('inf')

        # models are save only when their loss obtains the best value in the validation
    valtrack = 0

    print('There are %d parameter groups' % len(optimizer.param_groups))
    print('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])
    print('Initial vision params lr: %f' % optimizer.param_groups[1]['lr'])

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    cudnn.benchmark = True

    # preparing the training loader
    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
                     transforms.Compose([
                         transforms.Scale(256),  # rescale the image keeping the original aspect ratio
                         transforms.CenterCrop(256),  # we get only the center of that rescaled
                         transforms.RandomCrop(224),  # random crop within the center crop
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalize,
                     ]), data_path=opts.data_path, partition='train', sem_reg=opts.semantic_reg, ingrW2V=opts.ingrW2V),
        batch_size=opts.batch_size, shuffle=True,
        num_workers=opts.workers, pin_memory=True)
    print('Training loader prepared.')

    # preparing validation loader
    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
                     transforms.Compose([
                         transforms.Scale(256),  # rescale the image keeping the original aspect ratio
                         transforms.CenterCrop(224),  # we get only the center of that rescaled
                         transforms.ToTensor(),
                         normalize,
                     ]), data_path=opts.data_path, sem_reg=opts.semantic_reg, partition='val', ingrW2V=opts.ingrW2V),
        batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=True)
    print('Validation loader prepared.')

    # run epochs
    for epoch in range(opts.start_epoch, opts.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % opts.valfreq == 0 and epoch != 0:
            val_loss = validate(val_loader, model, criterion)

            # check patience
            if val_loss >= best_val:
                valtrack += 1
            else:
                valtrack = 0
            if valtrack >= opts.patience:
                # we switch modalities
                opts.freeVision = opts.freeRecipe;
                opts.freeRecipe = not (opts.freeVision)
                # change the learning rate accordingly
                adjust_learning_rate(optimizer, epoch, opts)
                valtrack = 0

            # save the best model
            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'valtrack': valtrack,
                'freeVision': opts.freeVision,
                'curr_val': val_loss,
            }, is_best)

            print('** Validation: %f (best) - %d (valtrack)' % (best_val, valtrack))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cos_losses = AverageMeter()
    if opts.semantic_reg:
        img_losses = AverageMeter()
        rec_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # compute output
        output = model(input_var[0], input_var[1], input_var[2])
        output = model(input_var[0], input_var[1], input_var[2])

        # compute loss
        if opts.semantic_reg:
            cos_loss = criterion[0](output[0], output[1], target_var[0])
            img_loss = criterion[1](output[2], target_var[1])
            rec_loss = criterion[1](output[3], target_var[2])
            # combined loss
            loss = opts.cos_weight * cos_loss + \
                   opts.cls_weight * img_loss + \
                   opts.cls_weight * rec_loss

            # measure performance and record losses
            cos_losses.update(cos_loss.data, input[0].size(0))
            img_losses.update(img_loss.data, input[0].size(0))
            rec_losses.update(rec_loss.data, input[0].size(0))
        else:
            loss = criterion(output[0], output[1], target_var[0])
            # measure performance and record loss
            cos_losses.update(loss.data[0], input[0].size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if opts.semantic_reg:
        print(optimizer.param_groups[1]['lr'])
        print(optimizer.param_groups[0]['lr'])
        print('Epoch: {0}\t'
              'cos loss {cos_loss.val:.4f} ({cos_loss.avg:.4f})\t'
              'img Loss {img_loss.val:.4f} ({img_loss.avg:.4f})\t'
              'rec loss {rec_loss.val:.4f} ({rec_loss.avg:.4f})\t'
              'vision ({visionLR}) - recipe ({recipeLR})\t'.format(
            epoch, cos_loss=cos_losses, img_loss=img_losses,
            rec_loss=rec_losses, visionLR=float(optimizer.param_groups[1]['lr']),
            recipeLR=float(optimizer.param_groups[0]['lr'])))
    else:
        print('Epoch: {0}\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'vision ({visionLR}) - recipe ({recipeLR})\t'.format(
            epoch, loss=cos_losses, visionLR=optimizer.param_groups[1]['lr'],
            recipeLR=optimizer.param_groups[0]['lr']))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    cos_losses = AverageMeter()
    if opts.semantic_reg:
        img_losses = AverageMeter()
        rec_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j], volatile=True).cuda())
        target_var = list()
        for j in range(len(target) - 2):  # we do not consider the last two objects of the list
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j], volatile=True))

        # compute output
        output = model(input_var[0], input_var[1], input_var[2])

        if i == 0:
            data0 = output[0].data.cpu().numpy()
            data1 = output[1].data.cpu().numpy()
            data2 = target[-2]
            data3 = target[-1]
        else:
            data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
            data1 = np.concatenate((data1, output[1].data.cpu().numpy()), axis=0)
            data2 = np.concatenate((data2, target[-2]), axis=0)
            data3 = np.concatenate((data3, target[-1]), axis=0)

    medR, recall = rank(opts, data0, data1, data2)
    print('* Val medR {medR:.4f}\t'
          'Recall {recall}'.format(medR=medR, recall=recall))

    return medR


def rank(opts, img_embeds, rec_embeds, rec_ids):
    random.seed(opts.seed)
    type_embedding = opts.embtype
    im_vecs = img_embeds
    instr_vecs = rec_embeds
    names = rec_ids

    # Sort based on names to always pick same samples for medr
    idxs = np.argsort(names)
    names = names[idxs]

    # Ranker
    N = opts.medr
    idxs = range(N)

    glob_rank = []
    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    for i in range(10):

        ids = random.sample(range(0, len(names)), N)
        im_sub = im_vecs[ids, :]
        instr_sub = instr_vecs[ids, :]
        ids_sub = names[ids]

        # if params.embedding == 'image':
        if type_embedding == 'image':
            sims = np.dot(im_sub, instr_sub.T)  # for im2recipe
        else:
            sims = np.dot(instr_sub, im_sub.T)  # for recipe2im

        med_rank = []
        recall = {1: 0.0, 5: 0.0, 10: 0.0}

        for ii in idxs:

            name = ids_sub[ii]
            # get a column of similarities
            sim = sims[ii, :]

            # sort indices in descending order
            sorting = np.argsort(sim)[::-1].tolist()

            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)

            if (pos + 1) == 1:
                recall[1] += 1
            if (pos + 1) <= 5:
                recall[5] += 1
            if (pos + 1) <= 10:
                recall[10] += 1

            # store the position
            med_rank.append(pos + 1)

        for i in recall.keys():
            recall[i] = recall[i] / N

        med = np.median(med_rank)
        # print "median", med

        for i in recall.keys():
            glob_recall[i] += recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i] / 10

    return np.average(glob_rank), glob_recall


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = opts.snapshots + 'model_e%03d_v-%.3f.pth.tar' % (state['epoch'], state['best_val'])
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


def adjust_learning_rate(optimizer, epoch, opts):
    """Switching between modalities"""
    # parameters corresponding to the rest of the network
    optimizer.param_groups[0]['lr'] = opts.lr * opts.freeRecipe
    # parameters corresponding to visionMLP
    optimizer.param_groups[1]['lr'] = opts.lr * opts.freeVision

    print('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])
    print('Initial vision lr: %f' % optimizer.param_groups[1]['lr'])

    # after first modality change we set patience to 3
    opts.patience = 3


if __name__ == '__main__':
    main()
