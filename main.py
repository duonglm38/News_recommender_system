import argparse
import random
import logging
import numpy as np
import torch
import sys
sys.path.insert(0, './')
from trainer import *
from baseline import *

manualSeed = 2019
print(manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


parser.add_argument('--data_path', default='./data')
parser.add_argument('--saved_model_path', default='./saved_model')
parser.add_argument('--data_split_path', default='./')
parser.add_argument('--w2v_path', default='./w2v/')

parser.add_argument('--model', default='LSTM', choices=['LSTM'])
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_seq_length', default=100, type=int)
parser.add_argument('--saved_model', default=True, type=bool)
parser.add_argument('--learning_rate', default=0.0005, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--patient_threshold', default=3, type=int)
parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'Adagrad', 'SGD', 'RMSProp'])

parser.add_argument('--rnn_hidden_size', default=300, type=int)
parser.add_argument('--doc_size', default=256, type=int)
parser.add_argument('--feature_dim', default=300, type=int)
parser.add_argument('--dropout1', default=0.5, type=float)
parser.add_argument('--dropout2', default=0.5, type=float)
parser.add_argument('--model_type', default=0, type=int, choices=[0, 1, 2, 3])  #0: doc2vec+lstm | 1:lstm | 2:title only | 3:content only
parser.add_argument('--using_gpu', default=True, type=bool)
parser.add_argument('--gen_vocab', default=False, type=bool)
parser.add_argument('--gen_w2v', default=False, type=bool)
parser.add_argument('--gen_embedding', default=False, type=bool)
parser.add_argument('--training', default=True, type=bool)
parser.add_argument('--baseline', default=False, type=bool)
parser.add_argument('--save_prediction', default=False, type=bool)

def main():
    args = parser.parse_args()
    if args.baseline:
        evaluate_baseline(args)
    else:
        device = torch.device('cuda:0' if args.using_gpu else 'cpu')
        args.device = device
        train(args)


if __name__ == '__main__':
    main()