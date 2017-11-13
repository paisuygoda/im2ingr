import torch.utils.data as data
from PIL import Image
import os
import pickle
import numpy as np
import lmdb
import torch
import torchwordemb

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        # print path
        return Image.new('RGB', (224,224), 'white')

def resize(img):
    w,h = img.size
    if w<h:
        ratio = float(h)/float(w)
        img = img.resize((256,int(256*ratio)))
    else:
        ratio = float(w)/float(h)
        img = img.resize((int(256*ratio),256))

    return img

class ImagerLoader(data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None,
                 loader=default_loader,square=False,data_path=None,partition=None,sem_reg=None,ingrW2V=None):
        ingr_id, _ = torchwordemb.load_word2vec_bin(ingrW2V)
        self.ingr_id = ingr_id

        if data_path==None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition=partition

        with open(os.path.join(data_path,partition+'_images.p'),'rb') as f:
            self.ids = pickle.load(f)

        self.square  = square

        self.imgPath = img_path
        self.mismtch = 0.1
        self.maxInst = 20
        with open(os.path.join(data_path,'ingredients_dict.p'),'rb') as f:
            self.ingr_dic = pickle.load(f)
        with open(os.path.join(data_path,'recipe_class.p'),'rb') as f:
            self.recipe_class = pickle.load(f)

        if sem_reg is not None:
            self.semantic_reg = sem_reg
        else:
            self.semantic_reg = False

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        recipeId  = self.ids[index][:-4]
        # we force 10 percent of them to be a mismatch
        if self.partition=='train':
            match = np.random.uniform() > self.mismtch
        elif self.partition=='val' or self.partition=='test':
            match = True 
        else:
            raise Exception('Partition name not well defined')

        target = match and 1 or -1

        # image
        if target==1:
                path = self.imgPath + recipeId + '.jpg'

        else:
                # we randomly pick one non-matching image
                all_idx = range(len(self.ids))
                rndindex = np.random.choice(all_idx)
                while rndindex == index:
                    rndindex = np.random.choice(all_idx) # pick a random index

                rndId = self.ids[rndindex][:-4]
                path = self.imgPath + rndId + '.jpg'

        # ingredients
        ingrs = []
        for item in self.ingr_dic[recipeId]['ingr']:
            if item in self.ingr_id:
                ingrs.append(self.ingr_id[item])
            else:
                ingrs.append(0)
        igr_ln = len(ingrs)
        if len(ingrs) < 50:
            ingrs.append([0]*(50-len(ingrs)))
        ingrs = torch.LongTensor(ingrs)

        # load image
        img = self.loader( path )

        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        print(img.size)

        rec_class = self.recipe_class[self.ingr_dic[recipeId]['dish']]

        if target == -1:
            img_class = self.recipe_class[self.ingr_dic[rndId]['dish']]
            img_id = rndId
        else:
            img_class = self.recipe_class[self.ingr_dic[recipeId]['dish']]
            img_id = recipeId

        # output
        if self.partition=='train':
            if self.semantic_reg:
                return [img, ingrs, igr_ln], [target, img_class, rec_class]
            else:
                return [img, ingrs, igr_ln], [target]
        else:
            if self.semantic_reg:
                return [img, ingrs, igr_ln], [target, img_class, rec_class, img_id, recipeId]
            else:
                return [img, ingrs, igr_ln], [target, img_id, recipeId]

    def __len__(self):
        return len(self.ids)
