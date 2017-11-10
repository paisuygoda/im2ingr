# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging
import sys


def trainW2V():
    # 進捗表示用
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Word2Vecの学習に使用する分かち書き済みのテキストファイルの準備
    sentences = word2vec.Text8Corpus('data/ingredients_titles.txt')

    # Word2Vecのインスタンス作成
    # sentences : 対象となる分かち書きされているテキスト
    # size      : 出力するベクトルの次元数
    # min_count : この数値よりも登場回数が少ない単語は無視する
    # window    : 一つの単語に対してこの数値分だけ前後をチェックする
    model = word2vec.Word2Vec(sentences, size=300, min_count=20, window=5)

    # 学習結果を出力する
    model.save("data/W2V_ingr_title.model")
    return model

if len(sys.argv) > 1:
    try:
        model = word2vec.Word2Vec.load(sys.argv[1])
    except ValueError:
        print(sys.argv[1], " NOT exist!")
        exit(1)

else:
    model = trainW2V()

vocab = model.wv.vocab
print("Writing to data/only_vocab.txt")
f = open('data/only_vocab.txt', 'w')
f.write("\n".join(vocab))
f.close()

print("Writing to data/vocab.txt")
model.wv.save_word2vec_format("data/vocab.txt")

print("Writing to data/vocab.bin")
model.wv.save_word2vec_format("data/vocab.bin", binary=True)
