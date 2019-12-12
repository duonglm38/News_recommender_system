from os import listdir
import json
import numpy as np
import torch


def load_data(file_path):
    idx2news = dict()
    for fn in listdir(file_path):
        with open(file_path + '/' + fn) as f:
            d = json.load(f)
            idx2news[int(d['Id'])] = {}
            idx2news[int(d['Id'])]['title'] = d['Title']
            idx2news[int(d['Id'])]['content'] = d['Content']
    return idx2news


def get_vocab(idx2news):
    vocab = set()
    for i in idx2news:
        news = idx2news[i]
        for token in news['title'].split():
            vocab.add(token)
        for token in news['content'].split():
            vocab.add(token)
    vocab = list(vocab)
    print('vocab size:', len(vocab))
    return vocab


def get_idx2word_word2idx(vocab):
    idx2word = {0: '<PAD>',
                 1: '<UNK>'}
    word2idx = {'<PAD>': 0,
                 '<UNK>': 1}
    for word in vocab:
        word2idx[word] = len(word2idx)
        idx2word[len(idx2word)] = word
    return idx2word, word2idx


def build_embedding(word2idx, idx2word, w2v_model, emb_dim=300):
    word2vec = {}
    for word in word2idx:
        if word in w2v_model:
            word2vec[word] = w2v_model[word]
    all_embedding = np.array(list(word2vec.values()))
    emb_mean = float(np.mean(all_embedding))
    emb_std = float(np.std(all_embedding))
    print('Embedding mean:', emb_mean)
    print('Embedding std:', emb_std)
    vocab_size = len(word2idx)
    embedding_matrix = torch.FloatTensor(vocab_size, emb_dim).normal_(emb_mean, emb_std)
    embedding_matrix[0] = torch.zeros((1, emb_dim))
    for i in range(2, vocab_size):
        word = idx2word[i]
        if word in word2vec:
            embedding_matrix[i] = torch.FloatTensor(word2vec[word])
    return embedding_matrix


def preprocess(idx2news, word2idx, args):
    idx2processedData = {}
    for i in idx2news:
        title = idx2news[i]['title']
        content = idx2news[i]['content']
        title_ids = [word2idx.get(w, 1) for w in title.split()]
        title_ids = title_ids[:args.max_seq_length]
        content_ids = [word2idx.get(w, 1) for w in content.split()]
        content_ids = content_ids[:args.max_seq_length]
        idx2processedData[i] = {'title': title_ids,
                                'content': content_ids}
    return idx2processedData

