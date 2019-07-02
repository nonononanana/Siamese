from torch import nn
from torch.autograd import Variable

import torch



class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.embed_size = config['model']['embed_dim']
        self.batch_size = config['model']['batch_size']
        self.hidden_size = config['model']['encoder']['hidden_size']
        self.num_layers = config['model']['encoder']['num_layers']
        self.bidir = config['model']['encoder']['bidirectional']
        if self.bidir:
            self.direction = 2
        else: self.direction = 1
        self.dropout = config['model']['encoder']['dropout']
        self.embedding = config['embedding_matrix']
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, dropout=self.dropout,
                            num_layers=self.num_layers, bidirectional=self.bidir)

    def initHiddenCell(self):
        rand_hidden = Variable(torch.randn(self.direction * self.num_layers, self.batch_size, self.hidden_size)).cuda()
        rand_cell = Variable(torch.randn(self.direction * self.num_layers, self.batch_size, self.hidden_size)).cuda()
        return rand_hidden, rand_cell

    def forward(self, input, hidden, cell):
        #if torch.cuda.is_available():
         #   input = input.cuda()
            #hidden = hidden.cuda()
            # = cell.cuda()
        input = self.embedding(input).view(self.batch_size, 1, -1)
        # print (input.shape)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell


# Siamese network with word-by-word encoder.
class Siamese(nn.Module):
    def __init__(self, config):
        super(Siamese, self).__init__()
        self.encoder = LSTMEncoder(config)

        self.input_dim = 2 * self.encoder.direction * self.encoder.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim/2)),
            nn.Linear(int(self.input_dim/2), 2),
        )

    def forward(self, s1, s2):
        if torch.cuda.is_available():
            s1 = [s.cuda() for s in s1]
            s2 = [s.cuda() for s in s2]
        # init hidden and cell
        h1, c1 = self.encoder.initHiddenCell()
        h2, c2 = self.encoder.initHiddenCell()

        for i in range(len(s1)):
            v1, h1, c1 = self.encoder(s1[i], h1, c1)

        for i in range(len(s2)):
            v2, h2, c2 = self.encoder(s2[i], h2, c2)

        #
        output = self.classifier(torch.cat((v1, v2), 2))

        return output




