import json
#import lmdb
import pickle
import os
from numpy import random
from PIL import Image
import csv
import sys
# import torchwordemb

def calc_f(tp,fp,fn):
    precision = tp/(tp*fp)
    recall = tp/(tp+fn)
    f = 2 * precision * recall / (precision + recall)
    print('presision = ', precision, '\nrecall = ', recall, '\nf-measure = ', f)
    return


def look_json(path):
    with open(path, 'r') as f:
        file = json.load(f)

    print(file[0])

"""
def look_lmdb(path):
    lmdb_env = lmdb.open(path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    lmdb_txn = lmdb_env.begin(write=False)
    lmdb_cursor = lmdb_txn.cursor()

    count = 0
    for key, value in lmdb_cursor:
        print("key = ", key, ", \nvalue = ", pickle.loads(value), "\n")
        count += 1
        if count > 10:
            break
"""


def look_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)

    if type(file) is list:
        for key, value in enumerate(file):
            print(value)
            if key > 10:
                break
    elif type(file) is dict:
        print("key?")
        s = input()
        print(type(s))
        if s in file:
            print(file[s])
        else:
            print(file)
    else:
        print(file)


def look_txt(path):
    with open(path, 'r') as f:
        file = f
        inglist = []
        for line in file:
            ing = line.split(" ")
            inglist.append(ing[0])

        n = 2
        while n > 1:
            print("ing id?")
            n = int(input()) - 1
            try:
                print(inglist[n])
            except:
                break


def img_sep(path):
    directory = os.listdir(path)
    train_list  =[]
    test_list   =[]
    val_list    =[]
    for pic in directory:
        seed = random.rand()
        if seed < 0.7 and len(train_list) < 42000:
            train_list.append(pic)
        elif seed < 0.85 and len(test_list) < 9000:
            test_list.append(pic)
        elif len(val_list) < 9000:
            val_list.append(pic)
        elif len(train_list) < 42000:
            train_list.append(pic)
        else:
            test_list.append(pic)

    print("len(train) = ", len(train_list))
    print("len(test)  = ", len(test_list))
    print("len(val)   = ", len(val_list))

    with open("train_images.p", 'wb') as f:
        pickle.dump(train_list,f)
    with open("test_images.p", 'wb') as f:
        pickle.dump(test_list,f)
    with open("val_images.p", 'wb') as f:
        pickle.dump(val_list,f)


def look_image(path):
    pic = Image.open(path)
    pic.show()


def ingr_max():
    count = 0
    max_ingr = 0
    recipeid = 0
    for line in open('data/Rakuten/recipe02_material_20160112.txt', 'r', encoding="utf-8"):
        count += 1.0
        proceeding = count / 5274990.0 * 100.0
        sys.stdout.write("\r%f%%" % proceeding)
        linelist = line.split()
        if recipeid == 0:
            recipeid = linelist[0]
            ingrlist = 0
        elif not linelist[0] == recipeid:
            if ingrlist > max_ingr:
                max_ingr = ingrlist
            recipeid = linelist[0]
            ingrlist = 0
        ingrlist += 1

    print("max ingr = ", max_ingr)

def ontrogy():

    tsv = csv.reader(open("data/synonym.tsv", "r", encoding="utf-8"), delimiter='\t')
    dic = {}
    word = ""
    count = 1
    id_dic = {}
    for row in tsv:
        if row[0] == "調理器具":
            break
        if row[1] is not word:
            word = row[1]
            id_dic[word] = count
            count += 1
        dic[row[2]] = row[1]

    with open('data/ontrogy_ingrcls.p', mode='wb') as f:
        pickle.dump(dic, f)
    with open('data/ingr_id.p', mode='wb') as f:
        pickle.dump(id_dic, f)
    print(count-1)

"""
def look_bin():
    name, vec = torchwordemb.load_word2vec_bin("data/vocab.bin")
    print(name['*'])
"""
print("MODE? (1 = json, 2 = image, 3 = pickle, 4 = text, 5 = img separation, \n\t6 = recipe_ingr, 7 = ontrogy)")
m = input()
print("PATH?")
path = input()
if m == "1":
    look_json(path)
elif m == '2':
    look_image('data/images/1040003832.jpg')
elif m == "3":
    if path == 'R':
        path = 'data/ingredients_dict.p'
    look_pickle(path)
elif m == '4':
    look_txt("data/vocab.txt")
elif m == '5':
    img_sep("data/images/")
elif m == '6':
    ingr_max()
elif m == '7':
    ontrogy()
else:
    print("Bad input mode")
