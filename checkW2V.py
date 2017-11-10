from gensim.models import Word2Vec

model = Word2Vec.load("data/W2V_only_ingr.model")

results = model.most_similar(positive=["りんご"], negative=["みかん"], topn=10)

for result in results:
    print(result[0], '\t', result[1])