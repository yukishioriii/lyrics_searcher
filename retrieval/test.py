from sklearn.utils import murmurhash3_32
import numpy as np
import os
import re
from tqdm import tqdm
from collections import Counter

from utils.utils import mecab_tokenizer
from scipy.sparse import csr_matrix
import scipy.sparse as sp


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

    def __init__(self, num_buckets=2**19):
        super().__init__()
        self.num_buckets = num_buckets
        self.read_data()
        self.build_tfidf()

    def read_data(self):
        for root, dirs, files in os.walk("../lyrics", topdown=False):
            for name in files:
                self.names.append(name)
                self.docs.append(mecab_tokenizer(re.sub("\n\t", "", open(
                    os.path.join(root, name), 'r').read())))
                if len(self.docs) > 10000:
                    return

    def hash(self, token):
        return murmurhash3_32(token) % self.num_buckets

    def get_doc_freq(self, mat):
        binary = (mat > 0).astype(int)
        return np.array(binary.sum(axis=0)).squeeze()

    def build_tfidf(self):
        for i in range(len(self.names)):
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
        tfs = matrix
        # tfs = matrix.log1p()
        tfidfs = tfs.dot(idfs)
        self.matrix = tfidfs

    def text2vec(self, query):

        query = [self.hash(word) for word in list(
            mecab_tokenizer(re.sub("\n\t", "", query.lower())))]
        wids_unique, wids_counts = np.unique(query, return_counts=True)
        tfs = wids_counts
        # tfs = np.log1p(wids_counts)

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
        b = [self.hash(word) for word in list(
            mecab_tokenizer(re.sub("\n\t", "", query.lower())))]
        a = self.matrix.T[b]
        c = vec.T[b]
        res = vec.dot(self.matrix.T).toarray().squeeze()
        o = np.argsort(-res)
        return [(i, float(res[i])) for i in o[:k]]


if __name__ == "__main__":
    a = TfidfBuilder()
    while True:
        
        text = input("> ")
        print([(a.IDX2NAME[i], score) for (i, score) in a.get_nearest(text, 10)])
