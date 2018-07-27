import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers, dropout=0.5):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, embedded, hidden):
        output, hidden = self.rnn(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        x = next(self.parameters()).detach()
        return x.new_zeros((self.n_layers, batch_size, self.hidden_size))
