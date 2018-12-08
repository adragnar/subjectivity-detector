"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
nlp = spacy.load('en')

base_model = torch.load("model_base")
rnn_model = torch.load("model_rnn")
cnn_model = torch.load("model_cnn")

TEXT = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
LABEL = data.Field(sequential=False, use_vocab=False)

train, val, test = data.TabularDataset.splits(
    path="/Users/RobertAdragna/Documents/Third Year/Fall Term/MIE 324 - Introduction to Machine Intelligence/mie324/a4",
    train='train.tsv', validation='validation.tsv', test='test.tsv',
    format='tsv', skip_header=True,
    fields=[('text', TEXT), ('label', LABEL)])

TEXT.build_vocab(train)
vocab = TEXT.vocab
vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))

while True:
    sentence = input('Enter a sentence \n')
    sentence = nlp(sentence)
    tokens = []
    length = []

    for token in sentence:
        tokens.append(vocab.stoi[token.text])
    tokens_tensor = torch.LongTensor(tokens)
    tokens_tensor = tokens_tensor.view(tokens_tensor.shape[0], 1)
    length.append(tokens_tensor.shape[0])
    length_tensor = torch.LongTensor(length)
    tokens_tensor = tokens_tensor.permute(1,0)

    base_predict = base_model(tokens_tensor, length)
    rnn_predict = rnn_model(tokens_tensor, length)
    cnn_predict = cnn_model(tokens_tensor, length)
    base_predict = base_predict.detach().numpy()
    rnn_predict = rnn_predict.detach().numpy()
    cnn_predict = cnn_predict.detach().numpy()

    if base_predict > 0.5:
        print("Model baseline: subjective (", str(base_predict),")\n")
    else:
        print("Model baseline: objective (", str(base_predict), ")\n")

    if rnn_predict > 0.5:
        print("Model rnn: subjective (", str(rnn_predict),")\n")
    else:
        print("Model rnn: objective (", str(rnn_predict), ")\n")

    if cnn_predict > 0.5:
        print("Model cnn: subjective (", str(cnn_predict),")\n")
    else:
        print("Model cnn: objective (", str(cnn_predict), ")\n")



# # 3.2 Processing of the data
# TEXT = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
# LABEL = data.Field(sequential=False, use_vocab=False)
#
# train, val, test = data.TabularDataset.splits(
#     path="/Users/RobertAdragna/Documents/Third Year/Fall Term/MIE 324 - Introduction to Machine Intelligence/mie324/a4",
#     train='train.tsv', validation='validation.tsv', test='test.tsv',
#     format='tsv', skip_header=True,
#     fields=[('text', TEXT), ('label', LABEL)])
#
# TEXT.build_vocab(train)
# vocab = TEXT.vocab
# vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))

# while(1):
#     sentance = input("Enter a Sentance \n")
#     sentance = nlp(sentance)
#     sentance_list = []
#     for i, word in enumerate(sentance):
#         sentance_list.append(vocab.stoi[word.text])
#
#     sentance_list = torch.tensor(sentance_list).long()
#     sentance_list.unsqueeze(0)
#     length_tensor = torch.LongTensor(length)
#
#
#     base_model = torch.load("model_base")
#     rnn_model = torch.load("model_rnn")
#     cnn_model = torch.load("model_cnn")
#
#     base_predict = base_model.forward(sentance_list, lengths)
#     rnn_predict = rnn_model.forward(sentance_list, lengths)
#     cnn_predict = cnn_model.forward(sentance_list, lengths)
#
#     if base_predict > 0.5:
#         print("Model baseline: subjective (", str(base_predict),")\n")
#     else:
#         print("Model baseline: objective (", str(base_predict), ")\n")
#
#     if rnn_predict > 0.5:
#         print("Model rnn: subjective (", str(rnn_predict),")\n")
#     else:
#         print("Model rnn: objective (", str(rnn_predict), ")\n")
#
#     if cnn_predict > 0.5:
#         print("Model cnn: subjective (", str(cnn_predict),")\n")
#     else:
#         print("Model cnn: objective (", str(cnn_predict), ")\n")


