# -*- coding: utf-8 -*-
import pickle
# import MeCab
import sys
import re
import sys

"""
def process_outline():
    data = []

    for line in open('data/Rakuten/recipe01_all_20170118.txt', 'r', encoding="utf-8"):
        linelist = line.split()
        linedict = {"id": linelist[0], "title": linelist[5], "dish": linelist[9]}
        data.append(linedict)

    with open('data/outline_dict.p', mode='wb') as f:
        pickle.dump(linedict, f)

    tagger = MeCab.Tagger("-Owakati")
    textlist = []
    for i, d in enumerate(data):
        result = tagger.parse(d["title"])
        t = result + " * * * * *"
        textlist.append(t)
        p = float(i)/float(len(data)) * 100.0
        sys.stdout.write("\r%f%%" % p)
        sys.stdout.flush()
    print("")

    text = " ".join(textlist)

    f = open('data/titles.txt', 'w')
    f.write(text)
    f.close()
"""

def process_ingredients():
    data = []
    id = 0
    for line in open('data/Rakuten/recipe02_material_20160112.txt', 'r', encoding="utf-8"):
        linelist = line.split()
        if id == 0:
            id = linelist[0]
            linedict = {"id": linelist[0], "ingredient": []}
        elif not linelist[0] == id:
            data.append(linedict)
            id = linelist[0]
            linedict = {"id": linelist[0], "ingredient": []}
        text = re.sub('[◎●Ａ　ABＢ■○①②③☆★※＊*▽▼▲△◆◇・()（）]', '', linelist[1])
        linedict['ingredient'].append(text)
    data.append(linedict)

    with open('data/ingredients_dict.p', mode='wb') as f:
        pickle.dump(linedict, f)

    textlist = []
    for i, d in enumerate(data):
        result = ' '.join(d["ingredient"])
        t = result + "\n * * * * *"
        textlist.append(t)
        p = float(i) / float(len(data)) * 100.0
        sys.stdout.write("\r%f%%" % p)
        sys.stdout.flush()
    print("")

    text = " ".join(textlist)

    f = open('data/ingredients.txt', 'w')
    f.write(text)
    f.close()


def combine_texts():

    textlist = []
    for line in open('data/ingredients.txt', 'r', encoding="utf-8"):
        textlist.append(line)
    textlist.append("\n * * * * *")
    print("Done ingr")
    for line in open('data/titles.txt', 'r', encoding="utf-8"):
        textlist.append(line)

    text = " ".join(textlist)
    f = open('data/ingredients_titles.txt', 'w')
    f.write(text)
    f.close()


def data_dict():
    data = {}
    recipeid = 0
    count = 0.0

    for line in open('data/Rakuten/recipe01_all_20170118.txt', 'r', encoding="utf-8"):
        linelist = line.split('\t')
        data[linelist[0]] = {"title": linelist[5], "dish": linelist[9]}

    for line in open('data/Rakuten/recipe02_material_20160112.txt', 'r', encoding="utf-8"):
        count += 1.0
        proceeding = count / 5274990.0 * 100.0
        sys.stdout.write("\r%f%%" % proceeding)
        linelist = line.split()
        if recipeid == 0:
            recipeid = linelist[0]
            ingrlist = []
        elif not linelist[0] == recipeid:
            data[recipeid]['ingr'] = ingrlist
            recipeid = linelist[0]
            ingrlist = []
        text = re.sub('[◎●Ａ　ABＢ■○①②③☆★※＊*▽▼▲△◆◇・()（）]', '', linelist[1])
        ingrlist.append(text)
        data[recipeid]['ingr'] = ingrlist

    with open('data/ingredients_dict.p', mode='wb') as f:
        pickle.dump(data, f)


def class_id_set():

    recipe_class = {}
    recipe_id = 1
    id2text = []
    for line in open('data/Rakuten/recipe01_all_20170118.txt', 'r', encoding="utf-8"):
        linelist = line.split('\t')
        dish = linelist[9]
        dish_class = linelist[3]
        if dish_class not in recipe_class:
            recipe_class[dish_class] = recipe_id
            recipe_id += 1
    print(recipe_id)
    with open('data/recipe_class.p', mode='wb') as f:
        pickle.dump(recipe_class, f)

    with open('data/recipe_id2recipe_text.p', mode='wb') as f:
        pickle.dump(id2text, f)

class_id_set()