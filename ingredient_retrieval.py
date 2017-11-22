import pickle
import json
import numpy as np
import sys

with open("data/ingredients_dict.p", 'rb') as f:
    ingr_dic = pickle.load(f)
with open("results/img_embeds.pkl", 'rb') as f:
    img_embeds = pickle.load(f)
with open("results/img_ids.pkl", 'rb') as f:
    img_ids = pickle.load(f)
with open("results/rec_embeds.pkl", 'rb') as f:
    rec_embeds = pickle.load(f)
with open("results/rec_ids.pkl", 'rb') as f:
    rec_ids = pickle.load(f)

tp = 0
fp = 0
fn = 0
for qid, query_emb in enumerate(img_embeds):
    proceeding = qid
    sys.stdout.write("\r%d" % proceeding)
    sys.stdout.flush()
    if proceeding >= 1000:
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
    query_id = str(img_ids[qid])
    result_id = str(rec_ids[id])

    query = ingr_dic[query_id]['ingr']
    result = ingr_dic[result_id]['ingr']

    TP = []
    for item in query:
        if item in result:
            TP.append(item)

    tp += len(TP)
    fp += len(result) - len(TP)
    fn += len(query) - len(TP)

precision = float(tp)/float(tp+fp)
recall = float(tp)/float(tp+fn)
f = 2 * precision * recall / (precision + recall)
print('\npresision = ', precision, '\nrecall = ', recall, '\nf-measure = ', f)
