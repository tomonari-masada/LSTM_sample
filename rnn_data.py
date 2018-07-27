import torch
import torch.nn as nn

class Corpus:

    def __init__(self, stem, min_len=2, max_len=10):
        self.train = open('train.' + stem + '.txt', 'r', encoding='utf-8')
        self.valid = open('valid.' + stem + '.txt', 'r', encoding='utf-8')
        self.test = open('test.' + stem + '.txt', 'r', encoding='utf-8')
        self.min_len = min_len
        self.max_len = max_len
        f = open('voc.' + stem + '.txt', 'r', encoding='utf-8')
        lines = f.read().strip().split('\n')
        vocab = [ line.split()[0] for line in lines ]
        self.BOS = len(vocab)
        vocab.append('BOS')
        self.EOS = len(vocab)
        vocab.append('EOS')
        f.close()
        self.vocab = vocab

    def __del__(self):
        self.train.close()
        self.valid.close()
        self.test.close()

    def rewind(self, f):
        f.seek(0)

    def read_seq(self, f, test=False):
        while True:
            for line in f:
                seq = [ int(x) - 1 for x in line.strip().split() ]
                if len(seq) >= self.min_len and len(seq) <= self.max_len:
                    yield [self.BOS] + seq + [self.EOS]
            if test: return
            f.seek(0)

    def _get_batch(self, f, batch_size=10, test=False):
        batch = list()
        target = list()
        for seq in self.read_seq(f, test=test):
            batch.append(seq[:-1])
            target.append(seq[1:])
            if len(batch) == batch_size:
                yield batch, target
                batch = list()
                target = list()
        yield batch, target

    def get_batch(self, f, batch_size=10, test=False):
        for batch, target in self._get_batch(f, batch_size=batch_size, test=test):
            perm = sorted(range(len(batch)), key=lambda k: len(batch[k]), reverse=True)
            batch = [ batch[i] for i in perm ]
            target = [ target[i] for i in perm ]
            yield batch, target
        
