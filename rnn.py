import sys
import io
import os
import time
import math
from datetime import datetime
from socket import gethostname
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence

from rnn_data import Corpus
from rnn_model import MyRNN

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
os.system('cp ' + sys.argv[0]
          + ' ./backup/' + sys.argv[0].split('.')[0]
          + '.`hostname`.`date +%Y%m%d%H%M`.py')
"""

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

torch.manual_seed(123)
np.random.seed(123)
if use_cuda:
    torch.cuda.manual_seed(123)

corpus = Corpus('seq.dblp.20180724._wiki_._tp3_._len2-30_._1978-2017_',
                min_len=2, max_len=30)
    
hidden_size = 500
n_layers = 2

lr = 0.0001
batch_size = 10
#clip = 0.25

show_interval = 100
valid_interval = 500
generate_interval = 500
lr_update_interval = 10000
log_interval = 10000
#----

output_size = len(corpus.vocab)

model = MyRNN(output_size, hidden_size, n_layers, dropout=0.5)
model.to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train(batch_size=100):
    model.train()
    hidden = model.initHidden(batch_size)
    batch, target = next(corpus.get_batch(corpus.train, batch_size=batch_size))
    embedded = pack_sequence([model.embedding(torch.tensor(s).to(device)) for s in batch])
    target = pack_sequence([torch.tensor(s).to(device) for s in target])
    optimizer.zero_grad()
    output, hidden = model(embedded, hidden)
    loss = criterion(output, target.data)
    loss.backward()
    optimizer.step()
    #torch.nn.utils.clip_grad_norm(model.parameters(), clip)
    return loss.cpu()

def perplexity(batch_size=100, test_size=10000):
    model.eval()
    hidden = model.initHidden(batch_size)
    cnt = 0
    loss_sum = 0
    len_sum = 0
    for batch, target in corpus.get_batch(corpus.valid, batch_size=batch_size, test=True):
        embedded = pack_sequence([model.embedding(torch.tensor(s).to(device)) for s in batch])
        target = pack_sequence([torch.tensor(s).to(device) for s in target])
        output, hidden = model(embedded, hidden)
        loss = criterion(output, target.data).cpu().detach()
        loss_sum += loss.item() * target.data.size(0)
        len_sum += target.data.size(0)
        cnt += batch_size
        if cnt >= test_size:
            corpus.rewind(corpus.valid)
            break
    return math.exp(loss_sum / len_sum)

def generate():
    model.eval()
    hidden = model.initHidden(1)
    print('## ', end='')
    input = torch.zeros(1, 1).long().to(device)
    word_id = corpus.BOS
    cnt = 0
    while True:
        batch = [[word_id]]
        embedded = pack_sequence([model.embedding(torch.tensor(s).to(device)) for s in batch])
        output, hidden = model(embedded, hidden)
        word_weights = output.squeeze().data.exp()
        word_id = torch.multinomial(word_weights, 1)[0]
        if word_id == corpus.EOS:
            break
        print(corpus.vocab[word_id], end=' ')
        cnt += 1
        if cnt >= corpus.max_len:
            break
    print(flush=True)

if __name__ == "__main__":
    print('# hidden_size {} | n_layers {}'.format(hidden_size, n_layers), flush=True)
    try:
        start_time = time.time()
        for i in range(1, 1000001):
            loss = train(batch_size)
            if i % show_interval == 0:
                elapsed = time.time() - start_time
                print('# {:d} {:.3f} {:.1f}'.format(i, loss.item(), elapsed), flush=True)
            if i % valid_interval == 0:
                print('# ppl {:.3f} lr {}'.format(perplexity(batch_size), lr), flush=True)
            if i % generate_interval == 0:
                for _ in range(10):
                    generate()
            if i % lr_update_interval == 0:
                lr *= 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if i % log_interval == 0:
                save_file = 'param.' + gethostname() + '.' \
                            + datetime.now().strftime('%Y%m%d%H')
                with open(save_file, 'wb') as f:
                    torch.save(model.state_dict(), f)
    except KeyboardInterrupt:
        print('#', '-' * 30)
