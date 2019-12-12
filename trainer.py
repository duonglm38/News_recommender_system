import pickle
import logging
import gensim
from utils import *
from model import *
from dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train(args):
    with open(args.data_split_path + 'train_val_test.pkl', 'rb') as f:
        data = pickle.load(f)
    train_data, val_data, test_data = data['train'], data['valid'], data['test']

    idx2news = load_data(args.data_path)

    if args.gen_vocab:
        vocab = get_vocab(idx2news)
        with open('./vocab/vocab.txt', 'w') as f:
            f.write('\n'.join(vocab))
    else:
        with open('./vocab/vocab.txt') as f:
            vocab = f.read().splitlines()
    idx2word, word2idx = get_idx2word_word2idx(vocab)
    
    if args.gen_embedding:
        w2v = gensim.models.Word2Vec.load(args.w2v_path + "word2vec_NLP1.model")
        word_embedding = build_embedding(word2idx=word2idx, idx2word=idx2word, w2v_model=w2v, emb_dim=100)
        torch.save(word_embedding, './vocab/embedding.pt')
    else:
        word_embedding = torch.load('./vocab/embedding.pt')
    
    with open('doc2vec/file.pkl', 'rb') as f:
        doc2vec = pickle.load(f)
    
    idx2processedData = preprocess(word2idx=word2idx, idx2news=idx2news, args=args)
    train_dataset = MyDataset(args=args, data=train_data, idx2processedData=idx2processedData, doc2vec=doc2vec)
    val_dataset = MyDataset(args=args, data=val_data, idx2processedData=idx2processedData, doc2vec=doc2vec)
    test_dataset = MyDataset(args=args, data=test_data, idx2processedData=idx2processedData, doc2vec=doc2vec)
    logger.info('train, dev, test: {} {} {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
    
    if args.model_type == 0:
        c_fn = MyDataset.collate_fn
    elif args.model_type == 1:
        c_fn = MyDataset.nodoc2vec_collate_fn
    elif args.model_type == 2:
        c_fn = MyDataset.justtitle_collate_fn
    else:
        c_fn = MyDataset.justcontent_collate_fn
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=c_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=c_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=c_fn)

    model = Classifier(embedding=word_embedding, args=args)
    if args.using_gpu:
        model = model.to(args.device)

    loss_function = nn.NLLLoss()  # .to(args.device)
    optim_method = getattr(torch.optim, args.optimizer)
    optimizer = optim_method(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    record = -1
    if args.training:
        for epoch in range(args.num_epoch):
            for batch in train_dataloader:
                if args.using_gpu:
                    batch = tuple(t.to(args.device) for t in batch)
                inputs = {'title1': batch[0],
                          'content1': batch[1],
                          'length1': batch[2],
                          'title2': batch[3],
                          'content2': batch[4],
                          'length2': batch[5],
                          'labels': batch[6]}
                logits = model(inputs)
                loss = loss_function(logits, inputs['labels'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_loss, p, r, f1, acc = evaluate(model, val_dataloader, args)
            logger.info('epoch: {}, loss: {}, precision: {}, recall: {}, f1: {}, accuracy: {}'.format(epoch, val_loss,
                                                                                                      p, r, f1, acc))
            if record < f1:
                record = f1
                patient = 0
                if args.saved_model:
                    torch.save(model.state_dict(), args.saved_model_path + '/' + args.model + '.pt')
            else:
                patient += 1
                if patient > args.patient_threshold:
                    break
        logger.info('Training done!')
    logger.info('Testing ...')
    model.load_state_dict(torch.load(args.saved_model_path + '/' + args.model + '.pt'))
    val_loss, p, r, f1, acc = evaluate(model, test_dataloader, args)
    logger.info('Test, loss: {}, precision: {}, recall: {}, f1: {}, accuracy: {}'.format(val_loss, p, r, f1, acc))


def evaluate(model, val_dataloader, args):
    preds = None
    out_label_ids = None
    num_sample = 0
    loss = 0
    loss_function = nn.NLLLoss()
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            if args.using_gpu:
                batch = tuple(t.to(args.device) for t in batch)
            inputs = {'title1': batch[0],
                      'content1': batch[1],
                      'length1': batch[2],
                      'title2': batch[3],
                      'content2': batch[4],
                      'length2': batch[5],
                      'labels': batch[6]}
            logits = model(inputs)
            loss = loss_function(logits, inputs['labels'])
            loss += loss_function(logits, inputs['labels']).item() * inputs['labels'].shape[0]
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            num_sample += inputs['labels'].shape[0]
    model.train()
    pred_labels = np.argmax(preds, axis=1)
    if args.save_prediction:
        np.savetxt('./prediction/predict' + str(args.model_type) + '.txt', preds)
    acc = accuracy_score(out_label_ids, pred_labels)
    f1 = f1_score(out_label_ids, pred_labels)
    p = precision_score(out_label_ids, pred_labels)
    r = recall_score(out_label_ids, pred_labels)
    loss = loss / num_sample
    return loss, p, r, f1, acc