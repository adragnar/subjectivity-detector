import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
spacy_en = spacy.load('en')

import argparse
import os

import models

def evaluate(model, data_iter, loss_fnc):
    total_corr = 0
    cummu_loss = 0
    for i, batch in enumerate(data_iter):
        feats = batch.text[0].transpose(0, 1)
        lengths = batch.text[1]
        labels = batch.label

        predictions = model.forward(feats, lengths)
        loss = loss_fnc(input=predictions.squeeze().float(), target=labels.float())
        cummu_loss += loss

        corr = (predictions > 0.5).squeeze().float() == labels.float()
        total_corr += int(corr.sum())

    return (float(total_corr) / len(data_iter.dataset), float(cummu_loss / i))

def train_func(model, train_iter, val_iter, test_iter, eval_every, model_name):
    learn_rate = 0.001
    MaxEpochs = 25
    batch_size = 64

    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    leftover = 0
    step_list = []
    train_data_list = []
    val_data_list = []
    best_val_acc = 0
    for counter, epoch in enumerate(range(MaxEpochs)):
        tot_loss = 0
        prev_tot_loss = 0
        tot_corr = 0
        prev_tot_corr = 0

        train_acc = 0
        val_acc = 0
        test_acc = 0
        train_loss = 0
        val_loss = 0
        test_loss = 0

        ultmin_val_acc = 0
        for i, batch in enumerate(train_iter):
            feats = batch.text[0].transpose(0, 1)
            lengths = batch.text[1]
            labels = batch.label

            optimizer.zero_grad()
            predictions = model.forward(feats, lengths)

            batch_loss = loss_fnc(input=predictions.squeeze().float(), target=labels.float())
            tot_loss += batch_loss

            batch_loss.backward()
            optimizer.step()

            corr = (predictions > 0.5).squeeze().float() == labels.float()
            tot_corr += int(corr.sum())

            # Print training stats
            if (((i + leftover) % eval_every == 0) and ((i + leftover) != 0)):
                val_acc, val_loss = evaluate(model, val_iter, loss_fnc)
                train_acc = float((tot_corr - prev_tot_corr) / (eval_every * batch_size))
                train_loss = float((tot_loss - prev_tot_loss) / (eval_every * batch_size))
                print("Batch", i, ": Total correct in last", eval_every, "batches is", tot_corr - prev_tot_corr,
                      "out of ", eval_every * batch_size)
                print("Total training accurracy over last batches is ", train_acc)
                print("Total validation accurracy over last batches is ", val_acc, "\n")

                if val_acc > ultmin_val_acc:
                    ultimin_val_acc = val_acc
                    torch.save(model, "model_" + model_name)

                # Record relevant values
                if len(step_list) == 0:
                    step_list.append(0)
                else:
                    step_list.append(step_list[-1] + eval_every)

                train_data_list.append(train_acc)
                val_data_list.append(val_acc)
                prev_tot_corr = tot_corr
                prev_tot_loss = tot_loss
        print("epoch", str(counter), "complete")
    test_acc, test_loss = evaluate(model, test_iter, loss_fnc)

    f = open(model_name + "stats", "w+")
    f.write("final train acc: " + str(train_acc))
    f.write("final train loss: " + str(train_loss))
    f.write("final val acc: " + str(val_acc))
    f.write("final val loss: " + str(val_loss))
    f.write("final test acc: " + str(test_acc))
    f.write("final test loss: " + str(test_loss))
    f.close()


def main(args):
    ######

    # 3.2 Processing of the data
    TEXT = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
    LABEL = data.Field(sequential=False, use_vocab=False)

    train, val, test = data.TabularDataset.splits(path="/Users/RobertAdragna/Documents/Third Year/Fall Term/MIE 324 - Introduction to Machine Intelligence/mie324/a4",
                                                  train='train.tsv', validation='validation.tsv', test='test.tsv',
                                                  format='tsv', skip_header=True,
                                                  fields=[('text', TEXT), ('label', LABEL)])

    # train_itr = data.BucketIterator(train, 64, sort_key=lambda x: len(x.TEXT), sort_within_batch=True, repeat=False)
    # val_itr = data.BucketIterator(val, 64, sort_key=lambda x: len(x.TEXT), sort_within_batch=True, repeat=False)
    # test_itr = data.BucketIterator(test, 64, sort_key=lambda x: len(x.TEXT), sort_within_batch=True, repeat=False)

    ######
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False,
        batch_sizes=(64, 64, 64), device=-1)
    # train_iter, val_iter, test_iter = data.BucketIterator.splits(
    #     (train, val, test), sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False,
    #     batch_sizes=(64, 64, 64), device=-1)
    TEXT.build_vocab(train)
    vocab = TEXT.vocab
    vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))

    ######

    # 5 Training and Evaluation
    base_model = models.Baseline(100, vocab)
    rnn_model = models.RNN(100, vocab, 100)
    cnn_model = models.CNN(100, vocab, 50, (2,4))
    train_func(rnn_model, train_iter, val_iter, test_iter, 20, "rnn")
    ######




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)