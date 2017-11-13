import torchwordemb
import pickle

ingr_id, _ = torchwordemb.load_word2vec_bin("data/vocab.bin")

with open('data/ingredients_dict.p', 'rb') as f:
    ingr_dic = pickle.load(f)
with open('data/test_images.p', 'rb') as f:
    ids = pickle.load(f)

for index in range(5):
    print("CASE: ", index)
    recipeId = ids[index][:-4]
    print("recipeid = ", recipeId)

    for item in ingr_dic[recipeId]['ingr']:
        print(item)
        if item not in ingr_id:
            print("but it's not in dict!")