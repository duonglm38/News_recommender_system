import pickle
import logging
import gensim
from utils import *
from model import *
from dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from scipy import spatial


logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def evaluate_baseline(args):
    with open(args.data_split_path + 'train_val_test.pkl', 'rb') as f:
        data = pickle.load(f)
    train_data, val_data, test_data = data['train'], data['valid'], data['test']
    
    with open('doc2vec/file.pkl', 'rb') as f:
        doc2vec = pickle.load(f)
    doc2vec = {int(k):v for k, v in doc2vec.items()}
    thres_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    record = 0
    choosen_thres = 0
    for thres in thres_list:
        val_preds = []
        val_labels = []
        for p in val_data:
            distance = 1 - spatial.distance.cosine(doc2vec[p[0]], doc2vec[p[1]])
            pred = 1 if distance > thres else 0
            val_preds.append(pred)
            val_labels.append(p[2])
        f1 = f1_score(val_labels, val_preds)
        acc = accuracy_score(val_labels, val_preds)
        p = precision_score(val_labels, val_preds)
        r = recall_score(val_labels, val_preds)
        logger.info('Valid, threshold: {}, precision: {}, recall: {}, f1: {}, accuracy: {}'.format(thres, p, r, f1, acc))
        if record < f1:
            record = f1
            choosen_thres = thres

    logger.info('Testing ...')            
    test_preds = []
    test_labels = []
    for p in test_data:
        distance = 1 - spatial.distance.cosine(doc2vec[p[0]], doc2vec[p[1]])
        pred = 1 if distance > choosen_thres else 0
        test_preds.append(pred)
        test_labels.append(p[2])
        
    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)
    p = precision_score(test_labels, test_preds)
    r = recall_score(test_labels, test_preds)

    logger.info('Test, thres_hold: {}, precision: {}, recall: {}, f1: {}, accuracy: {}'.format(choosen_thres, p, r, f1, acc))
