import pickle
import json
import numpy as np
import sys

with open("data/det_ingrs.json", 'r') as f:
    ingr_list = json.load(f)
with open("results/img_embeds.pkl", 'r') as f:
    img_embeds = pickle.load(f)
with open("results/img_ids.pkl", 'r') as f:
    img_ids = pickle.load(f)
with open("results/rec_embeds.pkl", 'r') as f:
    rec_embeds = pickle.load(f)
with open("results/rec_ids.pkl", 'r') as f:
    rec_ids = pickle.load(f)

ingr_dic = {}
num = float(len(ingr_list))
for i, recipe in enumerate(ingr_list):
    proceeding = float(i) / num * 100 * 0.5
    sys.stdout.write("\r%f" % proceeding)
    sys.stdout.flush()
    ings = []
    for item in recipe['ingredients']:
        ings.append(item['text'])
    ingr_dic[recipe['id']] = ings

tp = 0
fp = 0
fn = 0
for qid, query_emb in enumerate(img_embeds):
    proceeding = float(qid) * 0.5 + 50.0
    sys.stdout.write("\r%f" % proceeding)
    sys.stdout.flush()
    if proceeding == 100:
        break
    min = 9999.0
    id = -1
    for i, rec_emb in enumerate(rec_embeds):
        dist = np.linalg.norm(query_emb - rec_emb)
        if dist < min and (i is not qid):
            min = dist
            id = i
    if id == -1:
        print("all embeds are too far!")
        exit(1)
    query_id = img_ids[qid]
    result_id = rec_ids[id]

    query = ingr_dic[query_id]
    result = ingr_dic[result_id]

    TP = []
    for item in query:
        if item in result:
            TP.append(item)

    tp += len(TP)
    fp += len(result) - tp
    fn += len(query) - tp

precision = float(tp)/float(tp+fp)
recall = float(tp)/float(tp+fn)
f = 2 * precision * recall / (precision + recall)
print('\npresision = ', precision, '\nrecall = ', recall, '\nf-measure = ', f)