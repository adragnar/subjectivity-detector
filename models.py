import torch
import torch.nn as nn
import torch.nn.functional as F

def get_word_vector_matrix(sent, vocab, embedding):
    '''Takes in a string sentance, outputs a in-order matrix of each word's vector representation'''
    # sent = sent.split(" ")
    # for i in range(len(sent)):
    #     sent[i] = vocab.stoi[sent[i]]
    sent = embedding(torch.LongTensor(sent)).long()
    return sent

class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        ######

        # 4.1 YOUR CODE HERE
        self.vocab = vocab
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.embedding_dim,1)
        ######

    def forward(self, sentance, lengths=None):

        ######

        # 4.1 YOUR CODE HERE
        x = self.embedding(sentance)
        x = x.mean(1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

        # for i in range(sentance.shape(0)):
        #     x = sentance[i,:]
        #     x = get_word_vector_matrix(x,self.vocab, self.embedding)
        #     x = sum(x)/sentance.shape[1]
        #     x = self.fc1(x)
        #     x = F.sigmoid(x)
        #     predictions = torch.cat((predictions,x),0)
        # return predictions
        ######

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()

        # 4.2 YOUR CODE HERE
        self.vocab = vocab
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.GRU = nn.GRU(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, 1)


    def forward(self, x, lengths=None):

        # 4.2 YOUR CODE HERE
        x = x.permute(1,0)
        x = self.embedding(x)
        #x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
        output, hidden = self.GRU(x)
        hidden = (hidden.permute(1,0,2)).squeeze(1)
        x = self.fc1(hidden)
        x = torch.sigmoid(x)
        return x

# class RNN(nn.Module):
#     def __init__(self, embedding_dim, vocab, hidden_dim):
#         super(RNN, self).__init__()
#
#         ######
#
#         # 4.2 YOUR CODE HERE
#         self.vocab = vocab
#         self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.GRU = nn.GRU(self.embedding_dim, self.hidden_dim)
#         self.fc1 = nn.Linear(self.embedding_dim, 1)
#         ######
#
#     def forward(self, sent, lengths=None):
#
#         ######
#
#         # 4.2 YOUR CODE HERE
#         predictions = torch.tensor([])
#         h0 = torch.zeros(self.hidden_dim)
#         sent = self.embedding(sent)
#         packed = torch.nn.utils.rnn.pack_padded_sequence(sent, self.embedding_dim)
#         outputs, hidden = self.gru(packed, h0)
#         print(3)
#         # for i in range(sent.shape[0]):
#         #     x = sent[i, :]
#         #     x = get_word_vector_matrix(x, self.vocab, self.embedding)
#         #     for j in range (len(x)):
#         #         word = x[j, :]
#         #         if j == 0:
#         #             hn = self.GRU(word, h0)
#         #         else:
#         #             hn = self.GRU(word, hn)
#
#
#
#             #predictions = torch.cat((predictions, x), 0)
#         #return predictions
#         ######

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        ######

        # 4.3 YOUR CODE HERE
        self.vocab = vocab
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(1, 50, (2, self.embedding_dim)).float()
        self.conv2 = nn.Conv2d(1, 50, (4, self.embedding_dim)).float()
        self.fc1 = nn.Linear(100, 1)
        ######


    def forward(self, sentance, lengths=None):
        ######
        # 4.3 YOUR CODE HERE
        x = self.embedding(sentance)
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        maxpool1 = nn.MaxPool2d((x1.shape[2], x1.shape[3]))
        maxpool2 = nn.MaxPool2d((x2.shape[2], x2.shape[3]))
        x1 = maxpool1(x1)
        x2 = maxpool2(x2)
        x = torch.cat([x1, x2], 1).squeeze()
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x





        # for i in range(sentance.shape[0]):
        #     x = sentance[i, :]
        #     x = get_word_vector_matrix(x, self.vocab, self.embedding)
        #     x = (x.unsqueeze(1))
        #     x = x.float()
        #     results = self.conv1(x)
        #     results = self.maxpool(results)



        ######
