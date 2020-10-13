from sklearn.utils import murmurhash3_32
import numpy as np
import os
import re
from tqdm import tqdm
from collections import Counter

from utils.utils import mecab_tokenizer
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from database.doc_db import docdb
from tqdm import tqdm


class TfidfBuilder():

    docs = []
    names = []

    NAME2IDX = {}
    IDX2NAME = {}
    IDX2DOC = {}

    grow = []
    gcol = []
    gdata = []
    matrix = None
    freq = None

    n = 3

    def __init__(self, num_buckets=2**22, tokenizer=None):
        super().__init__()
        self.num_buckets = num_buckets
        self.tokenizer = tokenizer
        self.read_data()
        self.build_tfidf()

    def read_data(self):
        for root, dirs, files in os.walk("./lyrics", topdown=False):
            for name in files:
                self.names.append(name)
                self.docs.append(self.tokenize_ngram(re.sub("\n\t", "", open(
                    os.path.join(root, name), 'r').read().lower())))
                # if len(self.docs) > 1:
                #     return

    def ngrams(self, tokens):

        ngrams = [(s, e + 1)
                  for s in range(len(tokens))
                  for e in range(s, min(s + self.n, len(tokens)))]

        ngrams = [' '.join(tokens[s:e]) for (s, e) in ngrams]

        return ngrams

    def tokenize_ngram(self, text):
        if self.tokenizer:
            text = self.tokenizer(text)
        else:
            text = text.split()
        return self.ngrams(text)

    def hash(self, token):
        return murmurhash3_32(token) % self.num_buckets

    def read_wiki(self):
        for doc_id in tqdm(docdb.get_doc_ids()):
            self.names.append(doc_id)

            self.docs.append(
                re.sub("[^A-z0-9 ]", "", docdb.get_doc_text(doc_id).lower()))

            if len(self.names) > 2000000:
                break

    def get_doc_freq(self, mat):
        binary = (mat > 0).astype(int)
        return np.array(binary.sum(axis=0)).squeeze()

    def build_tfidf(self):
        for i in tqdm(range(len(self.names))):
            self.NAME2IDX[self.names[i]] = i
            self.IDX2NAME[i] = self.names[i]
            self.IDX2DOC[i] = self.docs[i]

            counter = Counter(self.docs[i])
            col = [self.hash(word)
                   for word in list(counter.keys())]
            data = [counter[word] for word in list(counter.keys())]

            self.grow.extend([i] * len(col))
            self.gcol.extend(col)
            self.gdata.extend(data)

        matrix = csr_matrix((self.gdata, (self.grow, self.gcol)),
                            (len(self.names), self.num_buckets))
        freq = self.get_doc_freq(matrix)
        self.freq = freq
        idfs = np.log((matrix.shape[0] - freq + 0.5) / (freq + 0.5))
        idfs[idfs < 0] = 0
        idfs = sp.diags(idfs, 0)
        tfs = matrix.log1p()
        tfidfs = tfs.dot(idfs)
        self.matrix = tfidfs

    def text2vec(self, query):
        print(self.tokenize_ngram(re.sub("\n\t", "", query.lower())))
        query = [self.hash(word) for word in
                 self.tokenize_ngram(re.sub("\n\t", "", query.lower()))]
        wids_unique, wids_counts = np.unique(query, return_counts=True)
        tfs = np.log1p(wids_counts)

        freq_w_in_query = self.freq[wids_unique]

        idfs = np.log(
            (self.matrix.shape[0] - freq_w_in_query + 0.5) / (freq_w_in_query + 0.5))
        idfs[idfs < 0] = 0
        data = np.multiply(tfs, idfs)
        col = wids_unique
        row = [0] * len(data)
        spvec = csr_matrix(
            (data, (row, col)), shape=(1, self.num_buckets)
        )

        return spvec

    def get_nearest(self, query, k=1):
        vec = self.text2vec(query)
        res = vec.dot(self.matrix.T).toarray().squeeze()
        o = np.argsort(-res)
        return [(i, float(res[i])) for i in o[:k]]


if __name__ == "__main__":
    a = TfidfBuilder(tokenizer=mecab_tokenizer)

    while True:
        text = input("> ")
        print([(a.IDX2NAME[i], score)
               for (i, score) in a.get_nearest(text, 10)])
